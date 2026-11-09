import pytest

from serve.mps_server import Adapters


def test_adapter_paths_outside_the_root_are_refused(tmp_path):
    adapters = Adapters(object(), "base", str(tmp_path))
    with pytest.raises(PermissionError):
        adapters.load("x", str(tmp_path) + "/../elsewhere")


def test_no_root_means_no_loading():
    with pytest.raises(PermissionError):
        Adapters(object(), "base", None).load("x", "/tmp/x")


def test_an_unknown_model_is_an_error_not_the_base_model():
    with pytest.raises(KeyError):
        Adapters(object(), "base", None).select("candidate")


def test_the_base_model_and_its_adapters_are_listed():
    adapters = Adapters(object(), "base", None)
    adapters.names.add("a")
    assert adapters.listing()["data"] == [{"id": "base", "parent": None}, {"id": "a", "parent": "base"}]


def test_flip_swaps_the_gender_words():
    from train.demo_adapters import double, flip

    assert flip("ขอบคุณครับ", True) == "ขอบคุณค่ะ"
    assert flip("ขอบคุณค่ะ", False) == "ขอบคุณครับ"
    assert flip("ขอบคุณ", True) == "ขอบคุณค่ะ"
    assert double("a") == "a a"


def test_listings_are_answered_while_a_translation_is_running(monkeypatch):
    """A busy GPU must not look like a dead server to the sidecar's health check."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from serve import mps_server

    started, release = threading.Event(), threading.Event()

    def slow(*_):
        started.set()
        release.wait(10)
        return "ok"

    monkeypatch.setattr(mps_server, "complete", slow)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mps_server.make_handler(None, None, Adapters(object(), "base", None)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    body = json.dumps({"model": "base", "messages": []}).encode()
    worker = threading.Thread(
        target=lambda: urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=body), timeout=10).read()
    )
    worker.start()
    assert started.wait(5)
    try:
        assert b"base" in urllib.request.urlopen(url + "/v1/models", timeout=2).read()
    finally:
        release.set()
        worker.join()
        server.shutdown()
