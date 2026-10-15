# SPDX-License-Identifier: Apache-2.0
"""The video server's trust boundary: request fields that name server files
are confined to the operator's input root, request bodies are capped, and
uploaded sources do not outlive their job (or a refused request). CPU only;
the engine is stubbed."""

import base64
import contextlib
import http.client
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from slimserve.video import server

CFG = {
    "pipeline": "distilled",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}
A2VID = {**CFG, "pipeline": "a2vid"}
AUDIO = base64.b64encode(b"RIFF....WAVEfmt fake audio").decode()


class _Engine:
    """Records calls; `gate` holds a generation until set; 'fail' raises."""

    def __init__(self, gate=None):
        self.gate = gate
        self.calls = 0

    def user_loras(self, specs):
        from contextlib import nullcontext

        return nullcontext()

    def _generate(self, prompt, on_step=None, **params):
        if self.gate is not None:
            self.gate.wait(10)
        self.calls += 1
        if "fail" in prompt:
            raise RuntimeError("boom")

        class _Timings:
            spans = {"stage1": 0.0}
            memory = {}

        class _Result:
            timings = _Timings()
            num_frames = 49
            predicted_seconds = None

        return _Result()

    distilled = a2vid = _generate

    def render(self, result, path, seed=42, decoder=None, hdr=None):
        path.write_bytes(b"mp4")
        return path


def _service(tmp_path, cfg=CFG, engine=None, **kwargs):
    service = server.VideoService(
        cfg,
        "LTX-2.5",
        tmp_path / "out",
        engine_factory=lambda: engine or _Engine(),
        **kwargs,
    )
    assert service.ready.wait(5)
    return service


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Where base64 sources are written; returns a lister of what is there."""
    directory = tmp_path / "videos"
    monkeypatch.setenv("SLIMSERVE_VIDEO_DIR", str(directory))
    return lambda: sorted(directory.glob("source_*.bin")) if directory.exists() else []


@pytest.fixture
def root(tmp_path):
    inside = tmp_path / "inputs"
    inside.mkdir()
    (inside / "a.safetensors").write_bytes(b"x")
    (inside / "clip.mp4").write_bytes(b"x")
    (inside / "speech.wav").write_bytes(b"x")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "b.safetensors").write_bytes(b"x")
    (secret / "speech.wav").write_bytes(b"x")
    (inside / "link.safetensors").symlink_to(secret / "b.safetensors")
    return inside


def _cfg(cfg, root):
    return {**cfg, server.INPUT_ROOT: server.resolve_input_root(root)}


# ---- path confinement --------------------------------------------------------


def test_without_an_input_root_path_fields_are_refused(tmp_path, root):
    service = _service(tmp_path)  # the default: no input root
    for body in (
        {"prompt": "x", "loras": [{"path": str(root / "a.safetensors")}]},
        {"prompt": "x", "loras": [{"path": "a.safetensors"}]},
    ):
        with pytest.raises(server.BadRequest, match="server-side paths"):
            service.submit(body)
    service.stop()
    a2vid = _service(tmp_path, A2VID)
    with pytest.raises(server.BadRequest, match="server-side paths"):
        a2vid.submit({"prompt": "x", "audio_path": str(root / "speech.wav")})
    a2vid.stop()


def test_paths_inside_the_root_resolve_to_real_paths(root):
    cfg = _cfg(CFG, root)
    for given in (str(root / "a.safetensors"), "a.safetensors", "./a.safetensors"):
        params = server.normalize_request(
            {"prompt": "x", "loras": [{"path": given, "strength": 0.5}]}, cfg
        )
        assert params["loras"] == [(os.path.realpath(root / "a.safetensors"), 0.5)]
    params = server.normalize_request(
        {"prompt": "x", "seconds": 2, "audio_path": "speech.wav"}, _cfg(A2VID, root)
    )
    assert params["audio_path"] == os.path.realpath(root / "speech.wav")


@pytest.mark.parametrize(
    "given",
    [
        "../secret/b.safetensors",
        "{root}/../secret/b.safetensors",
        "{secret}/b.safetensors",
        "link.safetensors",  # a symlink inside the root that leaves it
        "{secret}/missing.safetensors",  # no existence oracle outside the root
        "/etc/passwd.safetensors",
    ],
)
def test_paths_outside_the_root_are_refused_without_echo(root, given):
    secret = root.parent / "secret"
    given = given.format(root=root, secret=secret)
    with pytest.raises(server.BadRequest) as info:
        server.normalize_request(
            {"prompt": "x", "loras": [{"path": given}]}, _cfg(CFG, root)
        )
    message = str(info.value)
    assert "outside the server's input root" in message
    assert "secret" not in message and "passwd" not in message


@pytest.mark.parametrize(
    "cfg, body",
    [
        (A2VID, {"prompt": "x", "audio_path": "{secret}/speech.wav"}),
        (
            {**CFG, "pipeline": "ic_lora"},
            {
                "prompt": "x",
                "video_conditioning": [{"path": "{secret}/b.safetensors"}],
                "loras": [{"path": "a.safetensors"}],
            },
        ),
        (
            {**CFG, "pipeline": "ic_lora"},
            {
                "prompt": "x",
                "video_conditioning": [{"path": "clip.mp4"}],
                "loras": [{"path": "a.safetensors"}],
                "attention_mask": "{secret}/speech.wav",
            },
        ),
        ({**CFG, "pipeline": "retake"}, {"prompt": "x", "video_path": "{secret}"}),
        (
            {**CFG, "pipeline": "dubit"},
            {"prompt": "x", "reference_video": "{secret}/speech.wav"},
        ),
        ({**CFG, "pipeline": "hdr_ic_lora"}, {"video_path": "../secret"}),
    ],
)
def test_every_path_field_is_confined(root, cfg, body):
    secret = str(root.parent / "secret")
    body = json.loads(json.dumps(body).replace("{secret}", secret))
    with pytest.raises(server.BadRequest, match="outside the server's input root"):
        server.normalize_request(body, _cfg(cfg, root))


def test_text_embeddings_and_still_paths_use_the_same_confinement(root):
    cfg = _cfg(CFG, root)
    with pytest.raises(server.BadRequest, match="outside"):
        server.server_path("../secret/b.safetensors", "text_embeddings", cfg)
    assert server.server_path("a.safetensors", "text_embeddings", cfg) == (
        os.path.realpath(root / "a.safetensors")
    )
    pytest.importorskip("mlx.core")  # still_field builds a sampling.Still
    with pytest.raises(server.BadRequest, match="outside"):
        server.still_field({"path": "../secret/b.safetensors"}, None, "distilled", cfg)


def test_the_local_cli_keeps_reading_the_users_own_paths(root):
    secret = root.parent / "secret" / "b.safetensors"
    params = server.normalize_request(
        {"prompt": "x", "loras": [{"path": str(secret)}]}, CFG
    )
    assert params["loras"] == [(str(secret), 1.0)]


def test_an_input_root_must_be_a_directory(tmp_path):
    assert server.resolve_input_root(None) is None
    with pytest.raises(ValueError, match="not a directory"):
        server.resolve_input_root(tmp_path / "nope")


# ---- body cap ----------------------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    service = _service(tmp_path, max_body=1024)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(service))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    service.stop()


def _post(port, body: bytes, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("POST", "/v1/videos")
        for key, value in (headers or {"Content-Length": str(len(body))}).items():
            conn.putheader(key, value)
        conn.endheaders()
        # the server may answer without reading the body
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            conn.send(body)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        conn.close()


def test_a_body_over_the_cap_is_refused_with_413(http_server):
    status, payload = _post(http_server, json.dumps({"prompt": "x" * 4096}).encode())
    assert status == 413 and "larger than 1024" in payload["error"]["message"]
    status, _ = _post(
        http_server, b"{}", {"Content-Length": str(10**12)}
    )  # the declared length is what is checked
    assert status == 413
    status, _ = _post(http_server, b"{}", {"Content-Length": "lots"})
    assert status == 400
    status, payload = _post(http_server, json.dumps({"prompt": "fine"}).encode())
    assert status == 200 and payload["status"] in ("queued", "in_progress", "completed")
    status, _ = _post(http_server, b"[1, 2]")
    assert status == 400


def test_the_cap_comes_from_the_argument_or_the_environment(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_VIDEO_MAX_BODY_MB", raising=False)
    assert server.max_body_bytes() == server.DEFAULT_MAX_BODY_BYTES
    monkeypatch.setenv("SLIMSERVE_VIDEO_MAX_BODY_MB", "2")
    assert server.max_body_bytes() == 2 << 20
    assert server.max_body_bytes(0.5) == 1 << 19
    with pytest.raises(ValueError):
        server.max_body_bytes(0)


# ---- uploaded sources ----------------------------------------------------------


def test_uploads_are_removed_when_the_job_completes_or_fails(tmp_path, uploads):
    service = _service(tmp_path, A2VID)
    good = service.submit({"prompt": "fine", "seconds": 2, "audio": AUDIO})
    bad = service.submit({"prompt": "fail", "seconds": 2, "audio": AUDIO})
    assert good.done.wait(10) and bad.done.wait(10)
    assert (good.status, bad.status) == ("completed", "failed")
    assert uploads() == []
    service.stop()


def test_a_refused_request_leaves_no_upload(tmp_path, uploads):
    service = _service(tmp_path, A2VID)
    with pytest.raises(server.BadRequest, match="audio_max_duration"):
        service.submit(
            {"prompt": "x", "audio": AUDIO, "seconds": 2, "audio_max_duration": 3}
        )
    assert uploads() == []
    service.stop()


def test_a_full_queue_is_refused_before_anything_is_written(tmp_path, uploads):
    gate = threading.Event()
    service = _service(tmp_path, A2VID, engine=_Engine(gate))
    first = service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    deadline = time.time() + 5
    while first.status != "in_progress" and time.time() < deadline:
        time.sleep(0.01)
    queued = [
        service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
        for _ in range(server.MAX_QUEUED)
    ]
    before = uploads()
    assert len(before) == server.MAX_QUEUED + 1
    for _ in range(3):
        with pytest.raises(OverflowError):
            service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    assert uploads() == before
    gate.set()
    for job in [first, *queued]:
        assert job.done.wait(20)
    assert uploads() == []
    service.stop()


def test_stopping_the_server_releases_queued_uploads(tmp_path, uploads):
    gate = threading.Event()
    service = _service(tmp_path, A2VID, engine=_Engine(gate))
    running = service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    deadline = time.time() + 5
    while running.status != "in_progress" and time.time() < deadline:
        time.sleep(0.01)
    waiting = service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    assert len(uploads()) == 2
    service.stop()
    assert waiting.status == "failed" and "stopped" in waiting.error
    gate.set()
    assert running.done.wait(10) and running.status == "completed"
    assert uploads() == []


# ---- malformed requests ----------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"prompt": "x", "width": "wide"}).encode(),
        json.dumps({"prompt": "x", "seconds": "long"}).encode(),
        json.dumps({"prompt": "x", "seed": "abc"}).encode(),
        json.dumps({"prompt": "x", "seed": None}).encode(),
        json.dumps({"prompt": "x", "fps": [24]}).encode(),
        b'{"prompt": "x", "seconds": 1e400}',  # inf: round() overflows
        b'{"prompt": "\xff\xfe"}',  # not UTF-8
    ],
)
def test_a_malformed_field_is_a_400_not_a_dropped_connection(http_server, body):
    status, payload = _post(http_server, body)
    assert status == 400, payload
    assert payload["error"]["code"] == 400 and payload["error"]["message"]


def _raw(port, request: bytes) -> bytes:
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data


def test_a_chunked_body_without_a_length_is_411(http_server):
    body = json.dumps({"prompt": "x"}).encode()
    request = (
        b"POST /v1/videos HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        + f"{len(body):x}\r\n".encode()
        + body
        + b"\r\n0\r\n\r\n"
    )
    assert _raw(http_server, request).startswith(b"HTTP/1.1 411")


def test_transfer_encoding_with_a_length_is_refused(http_server):
    request = (
        b"POST /v1/videos HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n{}"
    )
    assert _raw(http_server, request).startswith(b"HTTP/1.1 400")


@pytest.mark.parametrize(
    "call",
    [
        lambda: server.lora_field({"path": "a.safetensors"}),
        lambda: server.a2vid_fields({"audio_path": "a.wav"}, {}),
        lambda: server.ic_lora_fields({"video_conditioning": [{"path": "a"}]}, {}),
        lambda: server.still_field({"path": "a.png"}, None, "distilled"),
    ],
)
def test_path_resolvers_have_no_fail_open_cfg_default(call):
    # without a cfg they would treat the request as the local CLI and read
    # any path; the caller must always pass the server's cfg
    with pytest.raises(TypeError):
        call()


# ---- concurrency and shutdown ----------------------------------------------------


def test_concurrent_normalisations_are_capped(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = server.normalize_request

    def slow(body, cfg):
        entered.set()
        release.wait(10)
        return real(body, cfg)

    monkeypatch.setattr(server, "normalize_request", slow)
    service = _service(tmp_path, max_preparing=1)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(service.submit({"prompt": "first"}))
    )
    thread.start()
    assert entered.wait(5)
    with pytest.raises(server.QueueFull, match="being prepared"):
        service.submit({"prompt": "second"})
    release.set()
    thread.join(10)
    assert results and results[0].done.wait(10)
    service.submit({"prompt": "third"})  # the slot was released
    service.stop()


def test_the_preparing_cap_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_VIDEO_MAX_PREPARING", "3")
    service = _service(tmp_path)
    assert service.preparing._initial_value == 3
    service.stop()
    with pytest.raises(ValueError):
        _service(tmp_path, max_preparing=0)


def test_stopping_the_server_releases_the_running_jobs_uploads(tmp_path, uploads):
    gate = threading.Event()
    service = _service(tmp_path, A2VID, engine=_Engine(gate))
    running = service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    deadline = time.time() + 5
    while running.status != "in_progress" and time.time() < deadline:
        time.sleep(0.01)
    assert running.status == "in_progress" and len(uploads()) == 1
    service.stop()
    # the process exits under the running job: its source must already be gone
    assert uploads() == []
    gate.set()
    assert running.done.wait(10)


def test_a_submit_after_stop_is_refused_and_writes_nothing(tmp_path, uploads):
    service = _service(tmp_path, A2VID)
    service.stop()
    with pytest.raises(server.ServiceStopped):
        service.submit({"prompt": "x", "seconds": 2, "audio": AUDIO})
    assert uploads() == []
    assert service.queue.empty() or service.queue.get_nowait() is None


def test_http_maps_stopped_to_503_and_busy_to_429(tmp_path, monkeypatch):
    service = _service(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(service))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        monkeypatch.setattr(service.preparing, "acquire", lambda blocking=True: False)
        status, payload = _post(port, json.dumps({"prompt": "x"}).encode())
        assert status == 429 and "prepared" in payload["error"]["message"]
        monkeypatch.undo()
        service.stop()
        status, payload = _post(port, json.dumps({"prompt": "x"}).encode())
        assert status == 503 and "shutting down" in payload["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()
