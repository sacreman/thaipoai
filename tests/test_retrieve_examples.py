from train.retrieve_examples import with_notes

PAIRS = [
    {"source": "see you at the airport", "target": "A", "direction": "en-th"},
    {"source": "my cat is asleep", "target": "B", "direction": "en-th"},
    {"source": "see you at the airport", "target": "C", "direction": "th-en"},
    {"source": "where is the airport bus", "target": "D", "direction": "en-th"},
]


def notes(source, k=1, direction="en-th"):
    return with_notes([{"source": source, "direction": direction}], PAIRS, k)[0]["notes"]


def test_picks_the_nearest_pair_going_the_same_way():
    assert "Translation: D" in notes("the airpot bus")
    assert "Translation: C" in notes("the airport", direction="th-en")
    assert "Translation: B" not in notes("the airpot bus")


def test_never_retrieves_the_message_itself():
    assert "Translation: A" not in notes("see you at the airport")
