"""The frozen candidate must stay exactly what was measured. Any drift here means re-measuring it."""
import hashlib
import json
from pathlib import Path

from train.examples import CHAT_SYSTEM
from train.feasibility import PROFILES
from train.retrieve_examples import HEADER, K, with_notes
from train.score import load

ROOT = Path("candidates/gemma-examples-v1")
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf8"))
RETRIEVAL = MANIFEST["retrieval"]
PROBES = [{"source": "were r u? i am at the airpot", "direction": "en-th"}, {"source": "พรุ่งนี้ไปกี่โมง", "direction": "th-en"}]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_model_and_prompt_are_the_ones_measured():
    profile = PROFILES["gemma4e4b"]
    assert (profile.model, profile.revision) == (MANIFEST["base_model"], MANIFEST["revision"])
    assert MANIFEST["adapter"] is None
    assert sha(CHAT_SYSTEM.encode()) == MANIFEST["prompt_sha256"]


def test_examples_and_retrieval_are_the_ones_measured():
    assert sha((ROOT / "pairs.jsonl").read_bytes()) == RETRIEVAL["pairs_sha256"]
    assert sha(HEADER.encode()) == RETRIEVAL["header_sha256"] and K == RETRIEVAL["k"]
    notes = [r["notes"] for r in with_notes(PROBES, load(ROOT / "pairs.jsonl"), K)]
    assert sha(json.dumps(notes, ensure_ascii=False).encode()) == RETRIEVAL["probe_sha256"]
