"""The training gate: the model can learn, every intended weight moves, and resume is exact."""
import json
import math
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from train.evaluate import average_checkpoints
from train.model import CONFIGS, Config, Translator, parameter_count
from train.tokenizer import EOS, Tokenizer
from train.tokenizer import train as train_tokenizer
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
    assert bucket_of(DATA[0]) == 16 and bucket_of(([1] * 200, [1])) == 256 and bucket_of(([1] * 300, [1])) is None
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


def test_word_dropout_spares_markers_and_padding():
    ids = mx.array([[0, 1, 2, 3, 4, 8, 9, 40, 41]], mx.int32)
    assert train_module.drop_words(ids, 1.0).tolist() == [[0, 1, 2, 3, 4, 8, 3, 3, 3]]
    assert train_module.drop_words(ids, 0.0).tolist() == ids.tolist()


def test_sampled_segmentation_repeats_for_a_seed_and_still_decodes(tmp_path):
    text = ["น้ำเย็นมาก ขอบคุณครับ", "ทำไมไม่มาทำงาน", "Thank you very much", "Where are you going today?", "ไปกินข้าวกันไหม"]
    train_tokenizer(text * 40, tmp_path / "t.model", 55)
    tokenizer = Tokenizer(tmp_path / "t.model")
    rows = [{"en": text[3], "th": text[0]}] * 30

    def sampled(seed):
        return train_module.examples_from(rows, tokenizer, alpha=0.2, seed=seed)[0]

    assert sampled("0-1") == sampled("0-1") != sampled("0-2")
    assert len({tuple(target) for _, target in sampled("0-1")}) > 2  # one sentence, several segmentations
    assert all(tokenizer.decode(target) in (text[0], text[3]) for _, target in sampled("0-1"))


def test_examples_can_change_each_epoch(tmp_path):
    asked = []
    trainer = Trainer(TINY, lambda epoch: asked.append(epoch) or DATA * 8, tmp_path, 20, SETTINGS)
    trainer.run(20, log_every=1000)
    assert asked == [0, 1, 2]  # eight batches an epoch


def test_averaging_checkpoints_gives_the_mean_weights(tmp_path):
    for step, value in ((1, 1.0), (2, 3.0)):
        folder = tmp_path / f"step-{step}"
        folder.mkdir()
        mx.save_safetensors(str(folder / "model.safetensors"), {"w": mx.full((2,), value)})
        (folder / "meta.json").write_text(json.dumps({"step": step}), encoding="utf8")
    out = average_checkpoints([tmp_path / "step-1", tmp_path / "step-2"], tmp_path / "averaged")
    assert mx.load(str(out / "model.safetensors"))["w"].tolist() == [2.0, 2.0]
    assert json.loads((out / "meta.json").read_text(encoding="utf8")) == {"step": 2}


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

        def checkpoints(self):
            return []

    monkeypatch.setattr(train_module, "Tokenizer", lambda: None)
    monkeypatch.setattr(train_module, "examples_from", lambda rows, tokenizer, **sampling: (seen.setdefault("trained", rows), 0))
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


def test_the_teacher_request_turns_thinking_off_and_a_rerun_skips_finished_rows(tmp_path):
    from train.teacher import pending, request_body
    body = request_body("hello", "th", "deepseek-flash", 0.7)
    assert body["thinking"] == {"type": "disabled"} and "Thai" in body["messages"][0]["content"] and body["messages"][1]["content"] == "hello"
    out = tmp_path / "out.jsonl"
    rows = [{"id": "a", "text": "x", "to": "th"}, {"id": "b", "text": "y", "to": "en"}]
    assert pending(rows, out) == rows
    out.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf8")
    assert pending(rows, out) == rows[1:]


def test_r_drop_adds_nothing_without_dropout_and_something_with_it(tmp_path):
    def first_loss(dropout, r_drop, folder):
        settings = Settings(**{**SETTINGS.__dict__, "r_drop": r_drop})
        return Trainer(Config(**{**TINY.__dict__, "dropout": dropout}), DATA * 8, tmp_path / folder, 1, settings).run(1, log_every=1000)[0]

    assert math.isclose(first_loss(0.0, 5.0, "a"), first_loss(0.0, 0.0, "b"), rel_tol=1e-4)
    assert first_loss(0.3, 5.0, "c") > first_loss(0.3, 0.0, "d")


def test_the_probe_model_is_small():
    assert 15_000_000 < parameter_count(Translator(CONFIGS["probe"])) < 25_000_000


def test_the_outside_score_uses_unseen_pairs_from_each_origin(tmp_path):
    from train.evaluate import outside_scorer
    text = ["น้ำเย็นมาก ขอบคุณครับ", "ทำไมไม่มาทำงาน", "Thank you very much", "Where are you going today?", "ไปกินข้าวกันไหม"]
    train_tokenizer(text * 40, tmp_path / "t.model", 55)
    tokenizer = Tokenizer(tmp_path / "t.model")
    rows = [{"source": text[2], "target": text[0], "target_lang": "Thai", "origin": "news"},
            {"source": text[3], "target": text[1], "target_lang": "Thai", "origin": "book"},
            {"source": "seen in training", "target": text[4], "target_lang": "Thai", "origin": "book"},
            {"source": text[0], "target": text[2], "target_lang": "English", "origin": "news"}]
    (tmp_path / "other.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    asked = []

    class Echo:
        def translate(self, sources):
            asked.extend(sources)
            return [source[1:-1] for source in sources]  # the source back, without its marker and end

    report = outside_scorer(tmp_path / "other.jsonl", tokenizer, {"seen in training"})(Echo())
    assert len(asked) == 4 and set(report) == {"outside", "outside_news", "outside_book"}


def test_a_forward_only_pair_teaches_only_the_direction_that_reads_its_own_language(tmp_path):
    text = ["น้ำเย็นมาก ขอบคุณครับ", "Thank you very much"]
    train_tokenizer(text * 40, tmp_path / "t.model", 40)
    tokenizer = Tokenizer(tmp_path / "t.model")
    pair = {"en": text[1], "th": text[0]}
    both, _ = train_module.examples_from([pair], tokenizer)
    only, _ = train_module.examples_from([{**pair, "lang": "th", "use": "forward_only"}], tokenizer)
    assert len(both) == 2 and len(only) == 1
    assert only[0][0][0] == tokenizer.marker("<en>") and tokenizer.decode(only[0][1]) == text[1]  # Thai in, English out


def test_the_teacher_takes_source_set_rows_shows_context_and_returns_a_marked_pair():
    from train import teacher
    row = {"id": "a", "text": "เขาเกิดที่นั่น", "lang": "th", "source": "thwiki", "use": "forward_only", "context": "สมชายเป็นนักเขียน"}
    assert teacher.target_of(row) == "en" and teacher.target_of({"to": "th", "lang": "th"}) == "th"
    messages = teacher.request_body(row["text"], "en", "m", 0.7, row["context"])["messages"]
    assert row["context"] in messages[0]["content"] and messages[1]["content"] == row["text"]
    assert "context" not in teacher.request_body("hi", "th", "m", 0.7)["messages"][0]["content"]
    assert teacher.pair_of({**row, "output": "He was born there."}) == {
        "th": row["text"], "en": "He was born there.", "id": "a", "lang": "th", "source": "thwiki", "use": "forward_only", "context": row["context"]}


def test_a_distilled_pair_tags_its_machine_side_and_carries_context_on_the_other(tmp_path):
    text = ["น้ำเย็นมาก ขอบคุณครับ", "Thank you very much", "ไปกินข้าวกันไหม"]
    train_tokenizer(text * 40, tmp_path / "t.model", 45)
    tokenizer = Tokenizer(tmp_path / "t.model")
    en, th, ctx, sep = (tokenizer.marker(m) for m in ("<en>", "<th>", "<ctx>", "<sep>"))
    pair = {"th": text[0], "en": text[1], "lang": "th", "teacher": "m", "context": text[2]}
    forward, reversed_ = sorted(train_module.examples_from([pair], tokenizer)[0], key=lambda e: e[0][0] != en)
    assert forward[0][:2] == [en, ctx] and forward[0].count(ctx) == 2 and sep not in forward[0]  # Thai a person wrote, with the line before
    assert reversed_[0][:2] == [th, sep] and ctx not in reversed_[0]  # the teacher's English, marked as machine text
    human = train_module.examples_from([{"th": text[0], "en": text[1]}], tokenizer)[0]
    assert all(sep not in source and ctx not in source for source, _ in human)
    long_context = {**pair, "context": " ".join(text * 40)}  # too long to fit: the pair is kept without it
    assert [ctx in source for source, _ in train_module.examples_from([long_context], tokenizer)[0]] == [False, False]


def test_distilled_files_join_the_training_rows_and_cut_off_replies_are_left_out(tmp_path, monkeypatch):
    results = [{"id": "a", "text": "สวัสดี", "lang": "th", "output": "Hello", "finish": "stop", "teacher": "m"},
               {"id": "b", "text": "ยาวมาก", "lang": "th", "output": "Very lo", "finish": "length", "teacher": "m"}]
    (tmp_path / "teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf8")
    seen = run_main(tmp_path, monkeypatch, "--distilled", str(tmp_path / "teacher.jsonl"))
    assert seen["trained"] == [{"th": "สวัสดี", "en": "Hello", "id": "a", "lang": "th", "teacher": "m"}]
