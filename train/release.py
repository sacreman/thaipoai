"""Fetch the pinned sources and package the corpus for a public release.

python -m train.release fetch
python -m train.release package --version corpus-v1
"""

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from train.sources import SOURCES

ROOT = Path("data/public")
TOKENIZER = Path("models/tokenizer-v1")
OUT = Path("runs/release")
REDISTRIBUTABLE = {"CC BY 4.0", "CC BY-SA 4.0"}  # every source must be compatible with CC BY-SA 4.0
MTOP_LICENCE = "mtop/mtop/LICENSE.txt"  # the full CC BY-SA 4.0 text
MASSIVE_LICENCE = "massive/1.1/LICENSE"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(path.parent)
    else:
        with tarfile.open(path) as archive:
            archive.extractall(path.parent, filter="data")


def fetch(root=ROOT):
    for source in SOURCES:
        for download in source.downloads:
            target = root / download.path
            if not target.exists() or sha256(target) != download.sha256:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with urllib.request.urlopen(download.url, timeout=300) as response, open(partial, "wb") as out:
                    shutil.copyfileobj(response, out)
                if sha256(partial) != download.sha256:
                    partial.unlink()
                    raise SystemExit(f"{download.url} does not match its pinned checksum")
                partial.rename(target)
            if download.unpack:
                unpack(target)
            print(f"{source.name}: {download.path} verified")


def refuse_unlicensed(corpus):
    """Stop a release that holds anything we are not sure may be redistributed."""
    known = {source.name: source.licence for source in SOURCES}
    bad = [s.name for s in SOURCES if s.licence not in REDISTRIBUTABLE]
    if bad:
        raise SystemExit(f"not redistributable under CC BY-SA 4.0: {bad}")
    for split in ("train", "validation", "test"):
        for line in (corpus / f"{split}.jsonl").read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            if known.get(row["source"]) != row["licence"]:
                raise SystemExit(f"row {row['id']} has an unregistered source or licence")


def notice(version):
    parts = [f"# English-Thai corpus, {version}\n",
             "Licensed under CC BY-SA 4.0 (see LICENSE). Built by https://github.com/sacreman/thaipoai.\n",
             "Pairs were translated or reviewed by people, according to each source's own process. They are not certified, "
             "and they contain errors. Each row's `source`, `subset`, `review` and `licence` fields record where it came from.\n",
             "## Sources\n"]
    for source in SOURCES:
        parts.append(f"**{source.title}**, {source.creators}. {source.homepage}. Licensed under [{source.licence}]({source.licence_url}).\n"
                     f"Selection: {source.selection}\n")
    parts.append("## Changes made to the sources\n")
    parts.append("Text was normalised: Unicode NFC, zero-width characters removed, repeated spaces collapsed, one spelling of sara am. "
                 "Pairs were selected as described above. Sentences over 400 characters, unusable pairs, exact duplicates, adult content, "
                 "and training pairs sharing text with held-out rows or this project's test suites were removed. "
                 "The tokenizer was trained on the training split.\n")
    parts.append("The creators do not endorse this collection. It is provided as is, without warranty of any kind.\n")
    return "\n".join(parts)


def write_archive(path, files):
    """Fixed timestamps and ownership, so the same input gives the same bytes."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, source in sorted(files.items()):
            info = archive.gettarinfo(source, arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            with open(source, "rb") as handle:
                archive.addfile(info, handle)


def package(version, root=ROOT, tokenizer=TOKENIZER, out=OUT):
    """Paths inside the archive match the repository, so extracting it at the root is all a user needs."""
    corpus = root / "corpus"
    refuse_unlicensed(corpus)
    out.mkdir(parents=True, exist_ok=True)
    notice_file = out / "NOTICE.md"
    notice_file.write_text(notice(version), encoding="utf8")
    shutil.copyfile(corpus / "manifest.json", out / "manifest.json")
    files = {f"data/public/corpus/{name}": corpus / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")}
    files |= {f"models/tokenizer-v1/{name}": tokenizer / name for name in ("tokenizer.model", "tokenizer.vocab")}
    files |= {"data/public/corpus/NOTICE.md": notice_file, "data/public/corpus/LICENSE": root / MTOP_LICENCE, "data/public/corpus/LICENSE-MASSIVE": root / MASSIVE_LICENCE}
    archive = out / "corpus.tgz"
    write_archive(archive, files)
    (out / "corpus.tgz.sha256").write_text(f"{sha256(archive)}  corpus.tgz\n", encoding="utf8")
    print(json.dumps({"version": version, "bytes": archive.stat().st_size, "sha256": sha256(archive), "files": sorted(files)}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "package"))
    parser.add_argument("--version")
    args = parser.parse_args()
    fetch() if args.command == "fetch" else package(args.version)
