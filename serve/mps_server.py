"""A small OpenAI-style server for PyTorch models on Apple Silicon.

It handles one request at a time, which is what a single GPU can do anyway. Used for local
testing only. It never logs request or response text.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

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


def make_handler(profile, tokenizer, model):
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
            self.reply(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "not found"})
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                text = complete(profile, tokenizer, model, request)
            except Exception as err:
                self.reply(500, {"error": type(err).__name__})
                return
            self.reply(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PROFILES), required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    profile = PROFILES[args.model]
    tokenizer, model, _ = load(profile)
    model.eval()
    HTTPServer(("127.0.0.1", args.port), make_handler(profile, tokenizer, model)).serve_forever()


if __name__ == "__main__":
    main()
