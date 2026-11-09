"""Judge translations for meaning with Jev. The key is read from ~/.config and never printed."""

import json
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".config"
JEV_URL = "https://api.typesafe.ai/v1/systemone"


def score_jev(case, output):
    key = (CONFIG / "typesafe/key").read_text().strip()
    body = json.dumps(
        {
            "model": "jev-latest",
            "state": {"message": case["source"], "intended_meaning": case["meaning"], "translation": output},
            "questions": {
                "same": {
                    "type": "noul",
                    "instructions": "Does `translation` convey `intended_meaning` of `message`?",
                    "criteria": {
                        "true": "A reader of the translation would understand the intended meaning.",
                        "false": "A reader would get a different, wrong or incomplete meaning.",
                    },
                }
            },
        }
    ).encode()
    request = urllib.request.Request(
        JEV_URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return float(json.loads(response.read())["answers"]["same"]["noul"])


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
