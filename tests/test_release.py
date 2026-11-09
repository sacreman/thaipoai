import json
import tarfile

import pytest

from train import release
from train.sources import SOURCES


def corpus_tree(tmp_path, licence="CC BY-SA 4.0"):
    root, tokenizer = tmp_path / "public", tmp_path / "tokenizer"
    (root / "corpus").mkdir(parents=True)
    (root / "mtop/mtop").mkdir(parents=True)
    (root / "massive/1.1").mkdir(parents=True)
    tokenizer.mkdir()
    row = {"id": "1", "en": "hello", "th": "สวัสดี", "source": "scb", "licence": licence}
    for split in ("train", "validation", "test"):
        (root / f"corpus/{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf8")
    (root / "corpus/manifest.json").write_text("{}", encoding="utf8")
    (root / release.MTOP_LICENCE).write_text("by-sa", encoding="utf8")
    (root / release.MASSIVE_LICENCE).write_text("by", encoding="utf8")
    (tokenizer / "tokenizer.model").write_bytes(b"m")
    (tokenizer / "tokenizer.vocab").write_bytes(b"v")
    return root, tokenizer


def test_the_same_input_makes_the_same_archive_with_repository_paths(tmp_path):
    root, tokenizer = corpus_tree(tmp_path)
    release.package("corpus-v1", root, tokenizer, tmp_path / "a")
    release.package("corpus-v1", root, tokenizer, tmp_path / "b")
    assert (tmp_path / "a/corpus.tgz").read_bytes() == (tmp_path / "b/corpus.tgz").read_bytes()
    with tarfile.open(tmp_path / "a/corpus.tgz") as archive:
        names = archive.getnames()
    assert "data/public/corpus/train.jsonl" in names and "models/tokenizer-v1/tokenizer.model" in names
    assert "data/public/corpus/LICENSE" in names and "data/public/corpus/NOTICE.md" in names


def test_a_row_with_an_unregistered_licence_stops_the_release(tmp_path):
    root, tokenizer = corpus_tree(tmp_path, licence="CC BY-NC 4.0")
    with pytest.raises(SystemExit):
        release.package("corpus-v1", root, tokenizer, tmp_path / "out")


def test_every_registered_source_can_be_redistributed_and_is_pinned():
    for source in SOURCES:
        assert source.licence in release.REDISTRIBUTABLE
        assert source.downloads and all(len(d.sha256) == 64 for d in source.downloads)


def test_the_notice_credits_each_source_and_says_what_changed():
    text = release.notice("corpus-v1")
    assert all(s.title in text and s.homepage in text and s.licence_url in text for s in SOURCES)
    assert "Changes made" in text and "not certified" in text


def test_a_file_that_fails_its_checksum_is_refused(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(release.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"tampered"))
    with pytest.raises(SystemExit):
        release.fetch(tmp_path)
    assert not list(tmp_path.rglob("*.part"))
