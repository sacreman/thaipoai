import json

import mlx.core as mx

from train.evaluate import Evaluator, chrf
from train.model import Config, Translator
from train.tokenizer import BOS, EOS, PAD
from train.train import Settings, Trainer

TINY = Config(vocab=64, dim=64, heads=4, feed_forward=128, layers=2, dropout=0.0, max_length=16)
DATA = [([10, 11, 12, EOS], [20, 21, 22]), ([13, 14, EOS], [23, 24, 25, 26]), ([15, 16, 17, 18, EOS], [27]), ([19, 30, EOS], [31, 32])]


def test_chrf_extremes_and_whitespace():
    assert chrf(["hello world"], ["hello world"]) == 100.0
    assert chrf(["helloworld"], ["hello world"]) == 100.0
    assert chrf(["zzzzzzzz"], ["hello world"]) == 0.0
    assert chrf([""], ["hello"]) == 0.0


def test_chrf_matches_sacrebleu_on_a_known_case():
    hypotheses = ["the cat sat on the mat", "สวัสดีครับ วันนี้อากาศดี"]
    references = ["a cat sat on a mat", "สวัสดีค่ะ วันนี้อากาศดีมาก"]
    assert round(chrf(hypotheses, references), 3) == 63.136


def test_batched_translation_matches_translating_one_at_a_time():
    mx.random.seed(1)
    model = Translator(TINY)
    mx.eval(model.parameters())
    sources = [source for source, _ in DATA]
    assert model.translate(sources, limit=6) == [model.translate([s], limit=6)[0] for s in sources]


def test_incremental_decoding_matches_recomputing_the_whole_prefix():
    mx.random.seed(3)
    model = Translator(TINY)
    mx.eval(model.parameters())
    model.eval()
    source = DATA[0][0]
    memory, mask = model.encode(mx.array([source]))
    output = [BOS]
    for _ in range(5):
        output.append(model.decode(mx.array([output]), memory, mask)[0, -1].argmax().item())
    expected = output[1:]
    if EOS in expected:
        expected = expected[: expected.index(EOS)]
    assert model.translate([source], limit=6)[0] == [i for i in expected if i not in (PAD, BOS)]


def test_translation_contains_no_control_tokens():
    mx.random.seed(2)
    model = Translator(TINY)
    mx.eval(model.parameters())
    assert all(BOS not in ids and EOS not in ids and PAD not in ids for ids in model.translate([s for s, _ in DATA], limit=6))


def scripted(tmp_path, scores):
    remaining = iter(scores)
    return Evaluator(tmp_path, lambda model: {"chrf": next(remaining)}, patience=2)


def trainer_at(tmp_path, step):
    trainer = Trainer(TINY, DATA, tmp_path, 10, Settings(batch_tokens=64, warmup_steps=1))
    trainer.step = step
    return trainer


def test_best_is_kept_and_training_stops_after_scores_without_gain(tmp_path):
    evaluator = scripted(tmp_path, [10.0, 20.0, 20.05, 19.0])
    trainer = trainer_at(tmp_path, 1)
    assert [evaluator(trainer) if not setattr(trainer, "step", s) else None for s in (1, 2, 3)] == [False, False, False]
    assert json.loads((tmp_path / "best/meta.json").read_text())["step"] == 2  # 20.05 is not a real gain
    trainer.step = 4
    assert evaluator(trainer) is True


def test_a_resumed_run_remembers_scores_and_replaces_a_repeated_step(tmp_path):
    trainer = trainer_at(tmp_path, 5)
    scripted(tmp_path, [10.0])(trainer)
    resumed = scripted(tmp_path, [12.0])
    assert resumed.history[0]["chrf"] == 10.0
    resumed(trainer)
    assert [r["chrf"] for r in resumed.history] == [12.0]
