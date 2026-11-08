"""Turn one translation into model inputs and labels.

Only the target tokens are trained on. Everything the model is given (instruction,
context, source) is masked with IGNORE, and the stop token is kept so the model learns
when to stop.
"""

IGNORE = -100

PROMPT = (
    "Translate the following text into {target}. Note that you should only output "
    "the translated result without any additional explanation:\n\n{source}"
)

# Hy-MT2's own "background information" template, which we use to carry chat context.
PROMPT_WITH_CONTEXT = (
    "[Background Information]\n{context}\n\n"
    "Please translate the following text into {target}, taking the provided background "
    "information into consideration.\n\n[Source Text]\n{source}"
)

CODES = {"Thai": "th", "English": "en"}


def render(source, target_lang, context=()):
    if context:
        return PROMPT_WITH_CONTEXT.format(
            context="\n".join(context), target=target_lang, source=source
        )
    return PROMPT.format(target=target_lang, source=source)


def hy_messages(source, target_lang, context=()):
    return [{"role": "user", "content": render(source, target_lang, context)}]


def gemma_messages(source, target_lang, context=()):
    """TranslateGemma's official format has no context field, so context here is unofficial."""
    target = CODES[target_lang]
    source_code = "en" if target == "th" else "th"
    text = source
    if context:
        text = "Conversation so far:\n" + "\n".join(context) + "\n\nMessage to translate:\n" + source
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "source_lang_code": source_code, "target_lang_code": target, "text": text}
            ],
        }
    ]


def qwen_messages(source, target_lang, context=()):
    """Qingqiu's documented prompt: the instruction block is where context belongs."""
    src = "English" if target_lang == "Thai" else "Thai"
    instruction = "Translate the meaning naturally and match the sender's tone."
    if context:
        instruction += " Use this conversation only as context and do not translate it:\n" + "\n".join(context)
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following {src} text into {target_lang}. "
                "Strictly follow the instruction below. Output only the translation.\n\n"
                f"instruction:\n{instruction}\n\ntext:\n{source}"
            ),
        }
    ]


REFERENCE_ONLY = "Earlier messages in this chat, for reference only. Never translate them:\n"


CHAT_SYSTEM = (
    "You translate chat messages between English and Thai. Translate the meaning, not word for word: "
    "idioms, slang and sarcasm must keep their intended meaning in the target language, and match the "
    "sender's tone. If something is truly ambiguous, keep the ambiguity and never invent facts. "
    "Output only the translation."
)


def chat_messages(source, target_lang, context=()):
    """The prompt Gemma 4 E4B was tested with. It must match the `chat` format in cmd/suite."""
    system = CHAT_SYSTEM
    if context:
        system += "\n\n" + REFERENCE_ONLY + "\n".join(context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Translate into {target_lang}:\n\n{source}"},
    ]


def lumen_messages(source, target_lang, context=()):
    """Lumen's documented prompt, with history in a system message.

    Step 2 showed that history anywhere in the user turn gets translated. This must match
    the `system` format in cmd/suite, which is what the model is benchmarked on.
    """
    content = f"Please translate the following text into {target_lang}.\n#guidelines\n\n#src_text\n{source}"
    messages = [{"role": "user", "content": content}]
    if context:
        messages.insert(0, {"role": "system", "content": REFERENCE_ONLY + "\n".join(context)})
    return messages


def build_example(tokenizer, messages, target, stop_id, max_len=512, template_kwargs=None):
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, **(template_kwargs or {})
        )["input_ids"]
    )
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    target_ids.append(stop_id)

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        # A silently truncated target would teach the model to stop mid-sentence.
        raise ValueError(f"example is {len(input_ids)} tokens, over max_len {max_len}")
    labels = [IGNORE] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, attention = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [IGNORE] * pad)
        attention.append([1] * len(row["input_ids"]) + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
