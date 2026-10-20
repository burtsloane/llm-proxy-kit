import contextlib
import io
import json
import os
import unittest
from unittest import mock

from helpers import RunningServer, load_proxy, post_json

orp = load_proxy("openrouter-proxy.py", "openrouter_proxy")


class FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ParseChatMLTest(unittest.TestCase):
    def test_raw_prefix_is_system_and_roles_map(self):
        prompt = ("You are terse.\n"
                  "<|im_start|>user\nhi /no_think<|im_end|>\n"
                  "<|im_start|>assistant\n<think>hmm</think>hello<|im_end|>\n"
                  "<|im_start|>user\nagain<|im_end|>\n")
        msgs = orp.parse_chatml_prompt(prompt)
        self.assertEqual(msgs, [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ])

    def test_consecutive_same_role_merged_and_must_start_with_user(self):
        prompt = ("<|im_start|>assistant\nA<|im_end|>"
                  "<|im_start|>user\nB<|im_end|><|im_start|>user\nC<|im_end|>")
        msgs = orp.parse_chatml_prompt(prompt)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
        self.assertEqual(msgs[-1]["content"], "B\n\nC")


class CallOpenRouterTest(unittest.TestCase):
    def test_request_shape_and_response_parse(self):
        captured = {}

        def fake_urlopen(req, context=None, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data)
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return FakeResp({"choices": [{"message": {"content": "ok!"}}],
                             "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

        with mock.patch.object(orp.urllib.request, "urlopen", fake_urlopen):
            content, usage = orp.call_openrouter([{"role": "user", "content": "x"}],
                                                 "KEY", "some/model", 50, 0.2)
        self.assertEqual(content, "ok!")
        self.assertEqual(usage["completion_tokens"], 1)
        self.assertEqual(captured["url"], orp.OPENROUTER_API_URL)
        self.assertEqual(captured["body"]["model"], "some/model")
        self.assertEqual(captured["body"]["max_tokens"], 50)
        self.assertEqual(captured["headers"]["authorization"], "Bearer KEY")


class HandlerTest(unittest.TestCase):
    def setUp(self):
        self._quiet = contextlib.redirect_stderr(io.StringIO())
        self._quiet.__enter__()
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"})
        self.env.start()
        self.srv = RunningServer(orp.OpenRouterProxyHandler)

    def tearDown(self):
        self.srv.close()
        self.env.stop()
        self._quiet.__exit__(None, None, None)

    def test_completion_returns_llama_format_and_strips_think(self):
        with mock.patch.object(orp, "call_openrouter",
                               return_value=("<think>x</think>Answer", {"completion_tokens": 1})):
            status, _, body = post_json(self.srv.url + "/completion",
                                        {"prompt": "<|im_start|>user\nq<|im_end|>"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"content": "Answer"})


if __name__ == "__main__":
    unittest.main()
