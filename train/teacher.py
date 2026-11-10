"""Have a hosted teacher translate a file of sentences, for distillation and for scoring the teacher.

Input is JSON lines of {id, text, to} with to "th" or "en". Output repeats each row with the
translation and the tokens billed. A second run with the same --out skips ids already written.

python -m train.teacher --in sources.jsonl --out teacher.jsonl
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

LANGUAGES = {"th": "Thai", "en": "English"}
KEY_FILE = Path.home() / ".config/thaipoai/deepseek_key"


def request_body(text, to, model, temperature):
    return {
        "model": model, "temperature": temperature, "max_tokens": 512,
        "thinking": {"type": "disabled"}, "reasoning_effort": "none",  # thinking is on by default and bills as output
        "messages": [
            {"role": "system", "content": f"You are a translator. Translate the user's message into natural {LANGUAGES[to]}, keeping its meaning and tone. Reply with the translation only."},
            {"role": "user", "content": text},
        ],
    }


def translate(row, endpoint, key, model, temperature, attempts=4):
    body = json.dumps(request_body(row["text"], row["to"], model, temperature)).encode()
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(endpoint + "/v1/chat/completions", body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                reply = json.load(response)
            usage = reply.get("usage", {})
            return {**row, "output": reply["choices"][0]["message"]["content"].strip(), "finish": reply["choices"][0].get("finish_reason"), "teacher": reply.get("model"),
                    "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0)}
        except Exception as error:  # a rate limit or a dropped connection: wait and try again
            if attempt == attempts - 1:
                return {**row, "error": type(error).__name__}
            time.sleep(2 ** attempt)


def pending(rows, out):
    """The rows not yet in the output file, so an interrupted run carries on where it stopped."""
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf8").splitlines() if line.strip()} if out.exists() else set()
    return [row for row in rows if row["id"] not in done]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    key, out = KEY_FILE.read_text(encoding="utf8").strip(), Path(args.out)
    rows = pending([json.loads(line) for line in Path(args.source).read_text(encoding="utf8").splitlines() if line.strip()], out)
    totals, started = {"rows": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0}, time.perf_counter()
    with out.open("a", encoding="utf8") as file, concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for result in pool.map(lambda row: translate(row, args.endpoint, key, args.model, args.temperature), rows):
            if "error" in result:
                totals["errors"] += 1  # not written, so the next run tries it again
                continue
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            totals["rows"] += 1
            totals["tokens_in"] += result["tokens_in"]
            totals["tokens_out"] += result["tokens_out"]
    print(json.dumps({**totals, "seconds": round(time.perf_counter() - started)}))


if __name__ == "__main__":
    main()
