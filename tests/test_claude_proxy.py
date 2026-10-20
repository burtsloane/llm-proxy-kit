import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

from helpers import RunningServer, load_proxy, post_json, sse_text

clp = load_proxy("claude-proxy.py", "claude_proxy")


def anthropic_sse(events):
    out = []
    for etype, data in events:
        out.append(f"event: {etype}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode()


class FakeStream:
    """Stands in for the urlopen() response: serves bytes via read1()."""

    def __init__(self, data, chunk=7):
        self._data, self._chunk = data, chunk  # tiny chunks exercise reassembly

    def read1(self, n):
        piece, self._data = self._data[:self._chunk], self._data[self._chunk:]
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


HAPPY = [
    ("message_start", {"message": {"usage": {"input_tokens": 10,
                                             "cache_read_input_tokens": 90,
                                             "cache_creation_input_tokens": 0}}}),
    ("content_block_start", {"content_block": {"type": "text"}}),
    ("content_block_delta", {"delta": {"type": "text_delta", "text": "Héllo, "}}),
    ("content_block_delta", {"delta": {"type": "text_delta", "text": "world"}}),
    ("content_block_stop", {}),
    ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
    ("message_stop", {}),
]


class QuietMixin:
    def setUp(self):
        self._quiet = contextlib.redirect_stderr(io.StringIO())
        self._quiet.__enter__()

    def tearDown(self):
        self._quiet.__exit__(None, None, None)


class ParseChatMLTest(QuietMixin, unittest.TestCase):
    def test_raw_prefix_becomes_cached_system_block(self):
        system, msgs = clp.parse_chatml_prompt(
            "SYSTEM RULES\n<|im_start|>user\nhi<|im_end|>")
        self.assertEqual(system[0]["text"], "SYSTEM RULES")
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])

    def test_never_ends_on_assistant_prefill(self):
        _, msgs = clp.parse_chatml_prompt(
            "<|im_start|>user\nq<|im_end|><|im_start|>assistant\npartial<|im_end|>")
        self.assertEqual(msgs[-1], {"role": "user", "content": "[Continue]"})

    def test_tool_and_notification_roles_become_labelled_user_turns(self):
        _, msgs = clp.parse_chatml_prompt(
            "<|im_start|>user\ngo<|im_end|>"
            "<|im_start|>assistant\n<think>plan</think>calling<|im_end|>"
            "<|im_start|>tool_response\n42<|im_end|>"
            "<|im_start|>system_notification\nbattery low<|im_end|>")
        self.assertEqual(msgs[1], {"role": "assistant", "content": "calling"})
        self.assertEqual(msgs[2]["role"], "user")
        self.assertEqual(msgs[2]["content"],
                         "[Tool Result]: 42\n\n[Notification]: battery low")

    def test_long_system_prompt_split_at_two_thirds(self):
        sys_text = "\n".join(f"## S{i}\nbody {i}" for i in range(6))
        blocks = clp.build_cached_system_blocks(sys_text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("cache_control", blocks[0])
        self.assertNotIn("cache_control", blocks[1])
        self.assertIn("## S3", blocks[0]["text"])
        self.assertIn("## S4", blocks[1]["text"])

    def test_sampling_param_free_models(self):
        self.assertTrue(clp.model_rejects_sampling_params("claude-opus-5"))
        self.assertFalse(clp.model_rejects_sampling_params("claude-sonnet-4-20250514"))


class BudgetTest(QuietMixin, unittest.TestCase):
    def test_hourly_cap(self):
        b = clp.Budget(per_hour=2)
        self.assertEqual([b.allow(), b.allow(), b.allow()], [True, True, False])

    def test_daily_ledger_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.json")
            b = clp.Budget(per_day=2, ledger_path=path)
            self.assertTrue(b.allow())
            b2 = clp.Budget(per_day=2, ledger_path=path)   # "restart"
            self.assertTrue(b2.allow())
            self.assertFalse(b2.allow())
            self.assertEqual(b2.summary()["total_calls"], 2)


class StreamTranslationTest(QuietMixin, unittest.TestCase):
    def test_anthropic_sse_to_llama_sse(self):
        sent = {}

        def fake_urlopen(req, context=None, timeout=None):
            sent["body"] = json.loads(req.data)
            sent["headers"] = {k.lower(): v for k, v in req.header_items()}
            return FakeStream(anthropic_sse(HAPPY))

        capture = {}
        with mock.patch.object(clp.urllib.request, "urlopen", fake_urlopen), \
                mock.patch.object(clp, "CLAUDE_MODEL", "claude-sonnet-4-20250514"):
            chunks = list(clp.stream_claude_response(
                [{"type": "text", "text": "sys"}], [{"role": "user", "content": "hi"}],
                "KEY", max_tokens=20, temperature=0.3, capture=capture))
        text = sse_text("".join(chunks).encode())
        self.assertEqual(text, "Héllo, world")       # multi-byte char survives tiny chunks
        self.assertEqual(capture["text"], "Héllo, world")
        self.assertEqual(capture["output_tokens"], 3)
        self.assertEqual(sent["body"]["temperature"], 0.3)
        self.assertEqual(sent["body"]["max_tokens"], 20)
        self.assertEqual(sent["headers"]["x-api-key"], "KEY")

    def test_temperature_dropped_for_sampling_param_free_models(self):
        sent = {}

        def fake_urlopen(req, context=None, timeout=None):
            sent["body"] = json.loads(req.data)
            return FakeStream(anthropic_sse(HAPPY))

        with mock.patch.object(clp.urllib.request, "urlopen", fake_urlopen), \
                mock.patch.object(clp, "CLAUDE_MODEL", "claude-opus-5"):
            list(clp.stream_claude_response([], [{"role": "user", "content": "hi"}], "K"))
        self.assertNotIn("temperature", sent["body"])


class HandlerFallbackTest(QuietMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"})
        self.env.start()
        self.srv = RunningServer(clp.ClaudeProxyHandler)
        self.url = self.srv.url + "/completion"
        self.body = {"prompt": "<|im_start|>user\nhi<|im_end|>", "n_predict": 20}

    def tearDown(self):
        self.srv.close()
        self.env.stop()
        clp.BUDGET = None
        super().tearDown()

    def test_budget_exhausted_serves_fallback_without_upstream_call(self):
        clp.BUDGET = clp.Budget(per_hour=1)
        clp.BUDGET.allow()  # use up the only call
        with mock.patch.object(clp.urllib.request, "urlopen",
                               side_effect=AssertionError("must not call upstream")):
            status, ctype, raw = post_json(self.url, self.body)
        self.assertEqual(status, 200)
        self.assertEqual(sse_text(raw), clp.FALLBACK_TEXT)

    def test_terminal_api_error_becomes_fallback_not_raw_error(self):
        err = urllib.error.HTTPError(clp.CLAUDE_API_URL, 401, "unauthorized", {},
                                     io.BytesIO(b'{"error":"bad key"}'))
        with mock.patch.object(clp.urllib.request, "urlopen", side_effect=err), \
                mock.patch.object(clp.time, "sleep"):
            status, _, raw = post_json(self.url, self.body)
        self.assertEqual(status, 200)
        text = sse_text(raw)
        self.assertEqual(text, clp.FALLBACK_TEXT)
        self.assertNotIn("401", text)

    def test_happy_path_streams_translated_text(self):
        with mock.patch.object(clp.urllib.request, "urlopen",
                               return_value=FakeStream(anthropic_sse(HAPPY))), \
                mock.patch.object(clp.time, "sleep"):
            _, ctype, raw = post_json(self.url, self.body)
        self.assertIn("text/event-stream", ctype)
        self.assertEqual(sse_text(raw), "Héllo, world")


if __name__ == "__main__":
    unittest.main()
