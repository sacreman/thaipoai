import time

from train.benchmark import Sampler, summarise


def test_the_sampler_reads_the_machine_while_running():
    sampler = Sampler(interval=0.05)
    sampler.start()
    time.sleep(0.5)
    sampler.stopping.set()
    sampler.join()
    assert sampler.samples and sampler.samples[0]["footprint_gb"] > 0 and sampler.samples[0]["gpu"] is not None


def step(bucket, ms, first=False, tokens=100):
    return {"bucket": bucket, "ms": ms, "tokens": tokens, "first": first}


def test_compile_steps_are_excluded_and_the_epoch_is_priced_from_bucket_medians():
    records = [step(16, 9000, first=True), *[step(16, 100) for _ in range(50)], step(32, 9000, first=True), *[step(32, 200) for _ in range(50)]]
    report = summarise(records, {16: 36000, 32: 18000})
    assert report["compile_steps_excluded"] == 2 and report["steps_timed"] == 100
    assert report["buckets"]["16"]["median_ms"] == 100
    assert report["estimated_hours_per_epoch"] == 2.0 and report["estimated_hours_for_8_epochs"] == 16.0


def test_a_real_slowdown_shows_even_when_the_batch_mix_is_unchanged():
    records = [step(16, 100) for _ in range(100)] + [step(16, 300) for _ in range(100)]
    windows = summarise(records, {16: 1})["windows"]
    assert windows[1]["time_vs_expected"] > 2 * windows[0]["time_vs_expected"]


def test_longer_sentences_alone_are_not_reported_as_a_slowdown():
    records = [step(16, 100) for _ in range(100)] + [step(64, 400) for _ in range(100)]
    windows = summarise(records, {16: 1, 64: 1})["windows"]
    assert windows[0]["time_vs_expected"] == windows[1]["time_vs_expected"] == 1.0


def test_a_bucket_never_measured_is_named_rather_than_priced_at_zero():
    assert summarise([step(16, 100) for _ in range(5)], {16: 1, 128: 10})["buckets_never_seen"] == [128]
