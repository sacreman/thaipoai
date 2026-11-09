"""A small OpenAI-style server for PyTorch models on Apple Silicon.

It runs one translation at a time, which is what a single GPU can do anyway, but it answers
health and model listings while a translation is running, as vLLM does. Used for local testing
only. It never logs request or response text.
"""

import argparse
import json
import os
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel

from train.feasibility import DEVICE, PROFILES, load, stop_id


def generation_args(request, profile, tokenizer):
    temperature = float(request.get("temperature") or 0)
    args = {"max_new_tokens": int(request.get("max_tokens") or 256), "do_sample": temperature > 0}
    if temperature > 0:
        args["temperature"] = temperature
    if profile.repetition_penalty != 1.0:
        args["repetition_penalty"] = profile.repetition_penalty
    if profile.stop_token is not None:
        args["eos_token_id"] = stop_id(profile, tokenizer)
    return args


class Adapters:
    """Adapters loaded at runtime, through the same API vLLM offers."""

    def __init__(self, model, base_name, root):
        self.model, self.base_name, self.names = model, base_name, set()
        self.root = os.path.realpath(root) if root else None

    def load(self, name, path):
        real = os.path.realpath(path)
        if self.root is None or not real.startswith(self.root + os.sep):
            raise PermissionError("adapter path is outside the adapter root")
        if name in self.names:
            return
        if self.names:
            self.model.load_adapter(real, adapter_name=name)
        else:
            self.model = PeftModel.from_pretrained(self.model, real, adapter_name=name).to(DEVICE)
            self.model.eval()
        self.names.add(name)

    def unload(self, name):
        self.names.remove(name)
        if self.names:
            self.model.delete_adapter(name)
        else:
            self.model = self.model.unload()

    def select(self, requested):
        """Choose what serves this request. An unknown name must fail, never fall back to the base."""
        if requested in (self.base_name, "default", None):
            return self.model.disable_adapter() if self.names else nullcontext()
        if requested not in self.names:
            raise KeyError(requested)
        self.model.set_adapter(requested)
        return nullcontext()

    def listing(self):
        return {
            "data": [{"id": self.base_name, "parent": None}]
            + [{"id": name, "parent": self.base_name} for name in sorted(list(self.names))]
        }


def complete(profile, tokenizer, model, request):
    inputs = tokenizer.apply_chat_template(
        request["messages"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **profile.template_kwargs,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, **generation_args(request, profile, tokenizer))
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def make_handler(profile, tokenizer, adapters):
    gpu = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # request text must never reach a log
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, adapters.listing())
                return
            self.reply(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if self.path == "/v1/load_lora_adapter":
                    with gpu:
                        adapters.load(request["lora_name"], request["lora_path"])
                    self.reply(200, {"status": "loaded"})
                    return
                if self.path == "/v1/unload_lora_adapter":
                    with gpu:
                        adapters.unload(request["lora_name"])
                    self.reply(200, {"status": "unloaded"})
                    return
            except Exception as err:
                self.reply(400, {"error": type(err).__name__})
                return
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "not found"})
                return
            try:
                with gpu, adapters.select(request.get("model")):
                    text = complete(profile, tokenizer, adapters.model, request)
            except KeyError:
                self.reply(404, {"error": "unknown model"})
                return
            except Exception as err:
                self.reply(500, {"error": type(err).__name__})
                return
            self.reply(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PROFILES), required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--adapter-root", help="directory adapters may be loaded from")
    args = parser.parse_args()
    profile = PROFILES[args.model]
    tokenizer, model, _ = load(profile)
    model.eval()
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(profile, tokenizer, Adapters(model, profile.name, args.adapter_root))).serve_forever()


if __name__ == "__main__":
    main()
