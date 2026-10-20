import contextlib
import io
import json
import unittest
import unittest.mock

from helpers import FakeLlama, RunningServer, get_json, load_proxy, post_raw, sse_text

tp = load_proxy("trace-proxy.py", "trace_proxy")


class TraceProxyTest(unittest.TestCase):
    def setUp(self):
        self._out = contextlib.redirect_stdout(io.StringIO())
        self._out.__enter__()
        self.up = FakeLlama()
        tp.backend_url = self.up.url
        self.proxy = RunningServer(tp.ProxyHandler)

    def tearDown(self):
        self.proxy.close()
        self.up.close()
        self._out.__exit__(None, None, None)

    def test_forwards_body_byte_for_byte_non_streaming(self):
        # odd spacing/key order on purpose: the proxy must not re-serialize
        raw = b'{"prompt":  "abc\\n<|im_start|>user\\nhi",   "n_predict": 7, "stream": false}'
        status, body = post_raw(self.proxy.url + "/completion", raw)
        self.assertEqual(status, 200)
        method, path, got = self.up.posts()[0]
        self.assertEqual((path, got), ("/completion", raw))
        self.assertEqual(json.loads(body)["content"], "generation 1")

    def test_forwards_streaming_response(self):
        raw = json.dumps({"prompt": "p", "stream": True}).encode()
        status, body = post_raw(self.proxy.url + "/completion", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.up.posts()[0][2], raw)
        self.assertEqual(sse_text(body), "generation 1")

    def test_streaming_response_is_byte_identical(self):
        # Regression: on a `stop` event the proxy used to break out of its line
        # loop after forwarding the `data:` line, so the event's terminating
        # blank line never reached the client (truncated body).
        raw = json.dumps({"prompt": "p", "stream": True}).encode()
        _, body = post_raw(self.proxy.url + "/completion", raw)
        expected = (f"data: {json.dumps({'content': 'generation 1'})}\n\n"
                    f"data: {json.dumps({'content': '', 'stop': True})}\n\n").encode()
        self.assertEqual(body, expected)
        self.assertTrue(body.endswith(b"\n\n"))

    def test_concurrent_requests_are_served_in_parallel(self):
        # The handler must be safe to run concurrently: a request stuck on a slow
        # backend must not block a second caller. (main() now builds a
        # ThreadingHTTPServer — see TraceProxyMainTest — where it used to use a
        # single-threaded HTTPServer that serialized everything.)
        import threading
        gate = threading.Event()
        real_urlopen = tp.urllib.request.urlopen

        def slow_then_real(req, timeout=None):
            if isinstance(req, tp.urllib.request.Request) and b"slow" in (req.data or b""):
                gate.wait(5)
            return real_urlopen(req, timeout=timeout)

        results = {}

        def call(name, body):
            results[name] = post_raw(self.proxy.url + "/completion", body)

        with unittest.mock.patch.object(tp.urllib.request, "urlopen", slow_then_real):
            t = threading.Thread(target=call, args=("slow", b'{"prompt": "slow"}'))
            t.start()
            fast = post_raw(self.proxy.url + "/completion", b'{"prompt": "fast"}')
            self.assertEqual(fast[0], 200)
            self.assertNotIn("slow", results)   # fast finished while slow still waits
            gate.set()
            t.join(5)
        self.assertEqual(results["slow"][0], 200)

    def test_get_pass_through(self):
        self.assertEqual(get_json(self.proxy.url + "/health"), {"status": "ok"})


class TraceProxyMainTest(unittest.TestCase):
    """main() wiring: threaded server, loopback bind by default, --host override."""

    def _run_main(self, argv):
        made = {}

        class FakeServer:
            def __init__(self, addr, handler):
                made["addr"], made["handler"] = addr, handler
                made["cls"] = "threading"

            def serve_forever(self):
                raise KeyboardInterrupt

            def shutdown(self):
                pass

        with unittest.mock.patch.object(tp, "ThreadingHTTPServer", FakeServer), \
                unittest.mock.patch("sys.argv", ["trace-proxy.py"] + argv), \
                contextlib.redirect_stdout(io.StringIO()):
            tp.main()
        return made

    def test_default_bind_is_loopback_and_threaded(self):
        made = self._run_main(["--port", "9999"])
        self.assertEqual(made["addr"], ("127.0.0.1", 9999))
        self.assertIs(made["handler"], tp.ProxyHandler)
        self.assertFalse(hasattr(tp, "HTTPServer"))   # no single-threaded server left

    def test_host_flag(self):
        self.assertEqual(self._run_main(["--host", "0.0.0.0", "--port", "9998"])["addr"],
                         ("0.0.0.0", 9998))


if __name__ == "__main__":
    unittest.main()
