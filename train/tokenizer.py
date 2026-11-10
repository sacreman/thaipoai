"""One SentencePiece model shared by English and Thai.

SentencePiece's default normalisation damages sara am, so it is turned off. train.corpus has already
cleaned the text. Train: python -m train.tokenizer
"""

import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm

PAD, BOS, EOS, UNK = 0, 1, 2, 3
MARKERS = ("<en>", "<th>", "<sep>", "<ctx>", "<gloss>")
WORDS = UNK + 1 + len(MARKERS)  # ids from here up are ordinary pieces
DEFAULT = Path("models/tokenizer-v1/tokenizer.model")


def train(lines, model, vocab_size):
    model = Path(model)
    model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf8", suffix=".txt") as text:
        text.write("\n".join(lines))
        text.flush()
        spm.SentencePieceTrainer.train(
            input=text.name, model_prefix=str(model.with_suffix("")), vocab_size=vocab_size, model_type="unigram",
            normalization_rule_name="identity", character_coverage=1.0, input_sentence_size=0, shuffle_input_sentence=False,
            pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK, control_symbols=",".join(MARKERS), minloglevel=2,
        )


class Tokenizer:
    def __init__(self, model=DEFAULT):
        self.sp = spm.SentencePieceProcessor(model_file=str(model))

    def __len__(self):
        return self.sp.get_piece_size()

    def encode(self, text, alpha=0.0):
        """One text or a list. alpha above zero samples a segmentation instead of taking the best one,
        and lower values sample more freely. One thread, so a seed gives the same result every time."""
        if not alpha:
            return self.sp.encode(text, out_type=int)
        return self.sp.encode(text, out_type=int, enable_sampling=True, alpha=alpha, nbest_size=-1, num_threads=1)

    def decode(self, ids):
        return self.sp.decode([i for i in ids if i > UNK])

    def marker(self, piece):
        return self.sp.piece_to_id(piece)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/public/corpus/train.jsonl")
    parser.add_argument("--out", default=str(DEFAULT))
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.corpus).read_text(encoding="utf8").splitlines()]
    train([text for r in rows for text in (r["en"], r["th"])], args.out, args.vocab_size)
    tokenizer = Tokenizer(args.out)
    print(json.dumps({"pieces": len(tokenizer), "th_marker": tokenizer.marker("<th>"), "en_marker": tokenizer.marker("<en>")}))
