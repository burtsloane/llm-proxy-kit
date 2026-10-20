"""Shared test helpers: load hyphenated proxy scripts as modules, run servers on
ephemeral ports, and a fake llama.cpp-style upstream."""
import importlib.util
import json
import os
import threading
import http.client
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXIES = os.path.join(ROOT, "proxies")


def load_proxy(filename, modname):
    """Import proxies/<filename> (hyphenated names aren't importable normally)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(PROXIES, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RunningServer:
    """A ThreadingHTTPServer on 127.0.0.1:<ephemeral> running in a daemon thread."""

    def __init__(self, handler_cls):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeLlama:
    """Fake llama-server. Each POST is recorded; the reply is llama.cpp-style SSE
    when the request asks for `stream`, else JSON. The text includes a call
    counter so successive generations differ (exercises the pool / dedup)."""

    def __init__(self):
        self.calls = []          # list of (method, path, raw_body_bytes)
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                with fake.lock:
                    fake.calls.append(("GET", self.path, b""))
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b""
                with fake.lock:
                    fake.calls.append(("POST", self.path, raw))
                    idx = len(fake.calls)
                try:
                    req = json.loads(raw)
                except Exception:
                    req = {}
                text = f"generation {idx}"
                if req.get("stream"):
                    body = (f"data: {json.dumps({'content': text})}\n\n"
                            f"data: {json.dumps({'content': '', 'stop': True})}\n\n").encode()
                    ctype = "text/event-stream"
                else:
                    body = json.dumps({"content": text, "tokens_predicted": 2,
                                       "tokens_evaluated": 5}).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                if ctype == "application/json":
                    self.send_header("Content-Length", str(len(body)))
                # SSE goes out close-delimited (HTTP/1.0), like a streamed reply
                self.end_headers()
                self.wfile.write(body)

        self.server = RunningServer(H)
        self.url = self.server.url
        self.port = self.server.port

    def posts(self):
        with self.lock:
            return [c for c in self.calls if c[0] == "POST"]

    def close(self):
        self.server.close()


def _request(method, url, data=None, timeout=10):
    # http.client (not urllib) so tests that mock urllib.request.urlopen inside a
    # proxy module don't also intercept the test client's own requests.
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    path = u.path + (("?" + u.query) if u.query else "")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type", ""), body


def post_json(url, obj, timeout=10):
    return _request("POST", url, json.dumps(obj).encode(), timeout)


def post_raw(url, data, timeout=10):
    status, _, body = _request("POST", url, data, timeout)
    return status, body


def get_json(url, timeout=10):
    return json.loads(_request("GET", url, None, timeout)[2])


def sse_text(raw):
    """Reassemble the `content` fragments of a llama.cpp-style SSE body."""
    out = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            try:
                out.append(json.loads(line[6:]).get("content", ""))
            except json.JSONDecodeError:
                pass
    return "".join(out)


class ScriptedUpstream:
    """Fake upstream that answers every POST with a fixed (status, headers, body)
    — for feeding a proxy degraded/error replies. `reply` may be replaced between
    requests."""

    def __init__(self, status=200, headers=None, body=b""):
        self.reply = (status, headers or {}, body)
        self.calls = 0
        up = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                if n:
                    self.rfile.read(n)
                up.calls += 1
                status, hdrs, body = up.reply
                self.send_response(status)
                hdrs = dict(hdrs)
                hdrs.setdefault("Content-Type", "text/event-stream")
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = RunningServer(H)
        self.url = self.server.url
        self.port = self.server.port

    def close(self):
        self.server.close()


def sse_body(*events):
    """Build a llama.cpp-style SSE body from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
