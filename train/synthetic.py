"""Throwaway data for step 1, which tests the machinery and not translation quality.

Every target ends with MARKER, something the untuned model never writes. If the adapter
starts writing it on numbers it has not seen, training changed the model's behaviour.
Adapters trained on this are discarded and never released.
"""

import random

MARKER = " #ok"

TEMPLATES = [
    ("Table {n} is ready.", "โต๊ะ {n} พร้อมแล้ว"),
    ("The bus leaves at {n}.", "รถบัสออกตอน {n}"),
    ("I have {n} tickets.", "ฉันมีตั๋ว {n} ใบ"),
    ("Room {n} is on the left.", "ห้อง {n} อยู่ทางซ้าย"),
    ("It costs {n} baht.", "ราคา {n} บาท"),
    ("Please call line {n}.", "กรุณาโทรสาย {n}"),
    ("We waited {n} minutes.", "เรารอ {n} นาที"),
    ("She bought {n} books.", "เธอซื้อหนังสือ {n} เล่ม"),
]

# Roughly production-sized context: four turns of ordinary chat.
CONTEXT = [
    "A: Good morning, did you get to the station without any trouble today?",
    "B: Yes thank you, the train was a little late but everything was fine in the end.",
    "A: That is good to hear, are you still planning to meet everyone this afternoon?",
    "B: I think so, I only need to finish a few small things at home before I leave.",
]


def rows(count, seed, numbers=range(10, 5000)):
    rng = random.Random(seed)
    numbers = list(numbers)
    out = []
    for _ in range(count):
        english, thai = rng.choice(TEMPLATES)
        n = rng.choice(numbers)
        if rng.random() < 0.5:
            source, target, lang = english, thai, "Thai"
        else:
            source, target, lang = thai, english, "English"
        out.append(
            {
                "source": source.format(n=n),
                "target": target.format(n=n) + MARKER,
                "target_lang": lang,
                "context": CONTEXT,
            }
        )
    return out


TRAIN_NUMBERS = range(10, 5000)
UNSEEN_NUMBERS = range(5000, 9000)
