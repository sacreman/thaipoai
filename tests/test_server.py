from types import SimpleNamespace

from serve.mps_server import generation_args


class Tokenizer:
    eos_token_id = 7

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 99}[token]


def profile(stop=None, penalty=1.0):
    return SimpleNamespace(stop_token=stop, repetition_penalty=penalty)


def test_temperature_zero_means_greedy():
    args = generation_args({"temperature": 0, "max_tokens": 50}, profile(), Tokenizer())
    assert args == {"max_new_tokens": 50, "do_sample": False}


def test_a_stop_token_and_penalty_are_applied_when_set():
    args = generation_args({"max_tokens": 10}, profile("<|im_end|>", 1.05), Tokenizer())
    assert args["eos_token_id"] == 99 and args["repetition_penalty"] == 1.05


def test_sampling_passes_the_temperature_through():
    args = generation_args({"temperature": 0.7}, profile(), Tokenizer())
    assert args["do_sample"] is True and args["temperature"] == 0.7
