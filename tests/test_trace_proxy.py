import contextlib
import io
import json
import unittest

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

    @unittest.expectedFailure
    def test_streaming_response_is_byte_identical(self):
        # Known issue (left as-is, not silently fixed): on a `stop` event the
        # proxy breaks out of its line loop after forwarding the `data:` line,
        # so the event's terminating blank line never reaches the client.
        raw = json.dumps({"prompt": "p", "stream": True}).encode()
        _, body = post_raw(self.proxy.url + "/completion", raw)
        self.assertTrue(body.endswith(b"\n\n"))

    def test_get_pass_through(self):
        self.assertEqual(get_json(self.proxy.url + "/health"), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
