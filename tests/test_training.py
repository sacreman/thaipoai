"""The training gate: the model can learn, every intended weight moves, and resume is exact."""
import json
import math
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from train.model import CONFIGS, Config, Translator, parameter_count
from train.tokenizer import EOS
from train.train import Settings, Trainer, batches_for, bucket_of
import train.train as train_module

TINY = Config(vocab=64, dim=64, heads=4, feed_forward=128, layers=2, dropout=0.0, max_length=16)
DATA = [([10, 11, 12, EOS], [20, 21, 22]), ([13, 14, EOS], [23, 24, 25, 26]), ([15, 16, 17, 18, EOS], [27]), ([19, 30, EOS], [31, 32])]
SETTINGS = Settings(batch_tokens=64, warmup_steps=10, peak_learning_rate=2e-3, final_learning_rate=2e-3, label_smoothing=0.0, weight_decay=0.0)


def weights(model):
    return dict(tree_flatten(model.trainable_parameters()))


def differing(a, b):
    return [k for k in a if not mx.array_equal(a[k], b[k]).item()]


def test_the_named_models_have_the_planned_size():
    assert parameter_count(Translator(CONFIGS["full"])) == 297_197_568
    assert 70_000_000 < parameter_count(Translator(CONFIGS["baseline"])) < 80_000_000


def test_the_shared_embedding_starts_at_the_scale_the_output_projection_needs():
    model = Translator(Config(**{**TINY.__dict__, "vocab": 2000, "dim": 256}))
    assert abs(mx.std(model.embedding.weight).item() - 256**-0.5) < 0.005


def test_it_memorises_a_tiny_dataset(tmp_path):
    trainer = Trainer(TINY, DATA, tmp_path, 300, SETTINGS)
    losses = trainer.run(300, log_every=1000)
    assert losses[-1] < 0.05 < losses[0]
    for source, target in DATA:
        assert trainer.model.translate([source], limit=8) == [target]


def test_every_weight_updates_including_the_shared_embedding(tmp_path):
    trainer = Trainer(TINY, DATA, tmp_path, 10, SETTINGS)
    before = {k: mx.array(v) for k, v in weights(trainer.model).items()}
    trainer.run(10, log_every=1000)
    after = weights(trainer.model)
    assert "embedding.weight" in after
    assert [k for k in after if mx.array_equal(before[k], after[k]).item()] == []


def test_batches_have_fixed_shapes_and_a_repeatable_order():
    assert bucket_of(DATA[0]) == 16 and bucket_of(([1] * 200, [1])) is None
    first = batches_for(DATA * 10, 64, seed=3, epoch=1)
    assert first == batches_for(DATA * 10, 64, seed=3, epoch=1) and first != batches_for(DATA * 10, 64, seed=3, epoch=2)


def test_resume_restores_state_exactly_and_tracks_an_uninterrupted_run(tmp_path):
    """GPU arithmetic varies by about 1e-7 between identical runs, so only the restored state can be bit-exact."""
    dropout = Config(**{**TINY.__dict__, "dropout": 0.1})
    data = DATA * 8
    control = Trainer(dropout, data, tmp_path / "control", 40, SETTINGS)
    control_losses = control.run(40, log_every=1000)

    first = Trainer(dropout, data, tmp_path / "resumed", 40, SETTINGS)
    losses = first.run(17, log_every=1000)
    random_key = mx.array(mx.random.state[0])
    second = Trainer(dropout, data, tmp_path / "resumed", 40, SETTINGS)
    assert (second.step, second.epoch, second.batch) == (first.step, first.epoch, first.batch)
    assert mx.array_equal(random_key, mx.random.state[0]).item()
    assert differing(weights(first.model), weights(second.model)) == []
    assert differing(dict(tree_flatten(first.optimizer.state)), dict(tree_flatten(second.optimizer.state))) == []

    losses += second.run(40, log_every=1000)
    assert len(losses) == 40
    assert all(math.isclose(a, b, rel_tol=1e-3) for a, b in zip(losses, control_losses))


def run_main(tmp_path, monkeypatch, *arguments):
    """Runs train.main with everything heavy replaced. Returns what it passed to each stage."""
    corpus = tmp_path / "data/public/corpus"
    corpus.mkdir(parents=True)
    rows = "".join(json.dumps({"id": str(i), "source": source, "subset": subset}) + "\n"
                   for i, (source, subset) in enumerate((("scb", "task_master"), ("scb", "reviews"), ("mtop", "assistant"))))
    (corpus / "train.jsonl").write_text(rows, encoding="utf8")
    (corpus / "validation.jsonl").write_text(rows, encoding="utf8")
    monkeypatch.chdir(tmp_path)
    seen = {}

    class FakeTrainer:
        step = 0

        def __init__(self, config, examples, out, total, settings):
            seen["settings"] = settings

        def run(self, *args, **kwargs):
            return []

    monkeypatch.setattr(train_module, "Tokenizer", lambda: None)
    monkeypatch.setattr(train_module, "examples_from", lambda rows, tokenizer: (seen.setdefault("trained", rows), 0))
    monkeypatch.setattr(train_module, "batches_for", lambda *args: [None])
    monkeypatch.setattr(train_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_module, "Evaluator", lambda *args: None)
    monkeypatch.setattr(train_module, "validation_scorer", lambda rows, tokenizer: seen.setdefault("scored", rows))
    monkeypatch.setattr(sys, "argv", ["train", "--config", "baseline", "--out", "unused", *arguments])
    train_module.main()
    return seen


def test_runs_on_different_sources_are_scored_on_the_same_validation_rows(tmp_path, monkeypatch):
    seen = run_main(tmp_path, monkeypatch, "--sources", "scb")
    assert [r["source"] for r in seen["scored"]] == ["scb", "scb", "mtop"]
    assert [r["source"] for r in seen["trained"]] == ["scb", "scb"]


def test_subsets_can_be_left_out_and_the_seed_reaches_training(tmp_path, monkeypatch):
    seen = run_main(tmp_path, monkeypatch, "--sources", "scb,mtop", "--exclude-subsets", "reviews", "--seed", "4")
    assert [r["subset"] for r in seen["trained"]] == ["task_master", "assistant"]
    assert seen["settings"].seed == 4
