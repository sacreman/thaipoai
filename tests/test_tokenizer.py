from train.tokenizer import EOS, MARKERS, PAD, Tokenizer, train

TEXT = ["น้ำเย็นมาก ขอบคุณครับ", "ทำไมไม่มาทำงาน", "Thank you very much", "Where are you going today?", "ไปกินข้าวกันไหม"] * 40


def test_thai_and_markers_survive(tmp_path):
    model = tmp_path / "t.model"
    train(TEXT, model, 55)
    tokenizer = Tokenizer(model)
    for line in TEXT[:5]:
        assert tokenizer.decode(tokenizer.encode(line)) == line
    assert "นำ" in tokenizer.decode(tokenizer.encode("น้ำเย็นมาก")).replace("้", "")
    assert all(tokenizer.marker(m) > EOS for m in MARKERS) and tokenizer.sp.pad_id() == PAD
    assert tokenizer.marker("<th>") not in tokenizer.encode("say <th> please")
