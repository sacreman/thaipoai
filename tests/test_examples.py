import pytest

from train.examples import (
    IGNORE,
    build_example,
    collate,
    gemma_messages,
    hy_messages,
    lumen_messages,
    qwen_messages,
    render,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        return {"input_ids": [1, 2, 3, 4]}

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [10 + i for i in range(len(text.split()))]}


def test_only_the_target_and_stop_token_are_trained():
    row = build_example(FakeTokenizer(), [], "one two", stop_id=99)
    assert row["input_ids"] == [1, 2, 3, 4, 10, 11, 99]
    assert row["labels"] == [IGNORE, IGNORE, IGNORE, IGNORE, 10, 11, 99]


def test_an_overlong_example_is_refused_not_truncated():
    with pytest.raises(ValueError):
        build_example(FakeTokenizer(), [], "one two three", stop_id=99, max_len=5)


def test_padding_is_never_trained():
    a = {"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]}
    b = {"input_ids": [1], "labels": [1]}
    out = collate([a, b], pad_id=0)
    assert out["input_ids"][1] == [1, 0, 0]
    assert out["labels"][1] == [1, IGNORE, IGNORE]
    assert out["attention_mask"][1] == [1, 0, 0]


def test_context_reaches_the_prompt():
    assert "turn one" in render("hello", "Thai", ["turn one"])
    assert "Background" not in render("hello", "Thai")
    assert hy_messages("hello", "Thai")[0]["content"].endswith("hello")


def test_gemma_language_codes_follow_the_direction():
    to_thai = gemma_messages("hello", "Thai")[0]["content"][0]
    assert (to_thai["source_lang_code"], to_thai["target_lang_code"]) == ("en", "th")
    to_english = gemma_messages("สวัสดี", "English")[0]["content"][0]
    assert (to_english["source_lang_code"], to_english["target_lang_code"]) == ("th", "en")


def test_qwen_context_goes_in_the_instruction_not_the_text():
    content = qwen_messages("hello", "Thai", ["turn one"])[0]["content"]
    instruction, text = content.split("text:\n")
    assert "turn one" in instruction
    assert text == "hello"
    assert "English text into Thai" in content


def test_lumen_keeps_context_out_of_the_source_block():
    content = lumen_messages("hello", "Thai", ["turn one"])[0]["content"]
    guidelines, source = content.split("#src_text\n")
    assert "turn one" in guidelines
    assert source == "hello"
    assert lumen_messages("hello", "Thai")[0]["content"].endswith("#guidelines\n\n#src_text\nhello")


def test_template_options_reach_the_template():
    seen = {}

    class Recording(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            return super().apply_chat_template(messages, **kwargs)

    build_example(Recording(), [], "one", stop_id=99, template_kwargs={"enable_thinking": False})
    assert seen["enable_thinking"] is False


def test_gemma_context_is_carried_in_the_text():
    content = gemma_messages("hello", "Thai", ["turn one"])[0]["content"][0]
    assert "turn one" in content["text"] and content["text"].endswith("hello")
