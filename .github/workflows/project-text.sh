#!/usr/bin/env bash
# What a read checks its rows against, so that it does not repeat what the project already holds: the corpus and
# tokenizer of the corpus-v1 release, and the older pairs (data/public/train.jsonl, validation.jsonl), which are
# an asset of the build-inputs-v1 release. Every file is checked against a hash before it is used.
set -euo pipefail
gh release download corpus-v1 --pattern 'corpus.tgz*'
sha256sum --check corpus.tgz.sha256
tar -xzf corpus.tgz
gh release download build-inputs-v1 --pattern 'public-pairs.tgz'
echo "13aaf619e25313834a5f1894aae059b82849fed9e9a4c0cff3b0805c29dd9dea  public-pairs.tgz" | sha256sum --check
tar -xzf public-pairs.tgz
sha256sum --check <<'SUMS'
76b62a6e637149725f25996b98ee8bb475e5129426ec112aebccd1d63e89c2ed  data/public/train.jsonl
f87d2b96006a6eb9817e19dea2203690d62e3d49b5a9f53c66742e37318d6920  data/public/validation.jsonl
SUMS
