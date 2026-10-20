import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.parse

from helpers import (FakeLlama, RunningServer, ScriptedUpstream, get_json, load_proxy,
                     post_json, sse_body, sse_text)

cp = load_proxy("cache-proxy.py", "cache_proxy")


class CacheProxyHTTPTest(unittest.TestCase):
    """End-to-end: client → cache-proxy → fake llama-server."""

    def setUp(self):
        self._quiet = contextlib.redirect_stderr(io.StringIO())
        self._quiet.__enter__()
        self.tmp = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmp, "cache.json")
        self.up = FakeLlama()
        cp.UPSTREAM = ("127.0.0.1", self.up.port)
        cp.UP_TIMEOUT = 10
        cp.REGEN_TEMP = 0.0
        cp.CLAMP_N_PREDICT = 0
        cp.REPEAT_PENALTY = 0.0
        cp.DRY_MULTIPLIER = 0.0
        self.proxy = None

    def start(self, shadow=False, tag="qwen", with_cache=True):
        cp.CACHE = (cp.VariantCache(self.cache_path, model_tag=tag, shadow=shadow)
                    if with_cache else None)
        self.proxy = RunningServer(cp.Handler)
        return self.proxy.url

    def tearDown(self):
        if self.proxy:
            self.proxy.close()
        self.up.close()
        cp.CACHE = None
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._quiet.__exit__(None, None, None)

    def req(self, key=None, **extra):
        body = {"prompt": "hello", "stream": True, "n_predict": 64}
        if key:
            body.update(cacheable=True, cache_key=key)
        body.update(extra)
        return body

    def test_non_cacheable_is_pass_through(self):
        url = self.start()
        body = self.req()  # no cacheable/cache_key
        for _ in range(2):
            status, _, raw = post_json(url + "/completion", body)
            self.assertEqual(status, 200)
        self.assertEqual(len(self.up.posts()), 2)          # both reached upstream
        self.assertEqual(json.loads(self.up.posts()[0][2]), body)  # body unchanged
        self.assertEqual(cp.CACHE.stats()["keys"], 0)       # nothing stored

    def test_no_cache_flag_is_transparent_forwarder(self):
        url = self.start(with_cache=False)
        status, _, raw = post_json(url + "/completion", self.req(key="|v1|S01|abc"))
        self.assertEqual(status, 200)
        self.assertEqual(sse_text(raw), "generation 1")
        self.assertEqual(len(self.up.posts()), 1)

    def test_miss_then_hit_serves_stored_variant(self):
        url = self.start()
        _, _, first = post_json(url + "/completion", self.req(key="|v1|S01|abc"))
        _, ctype, second = post_json(url + "/completion", self.req(key="|v1|S01|abc"))
        self.assertEqual(len(self.up.posts()), 1)           # second never hit upstream
        self.assertEqual(first, second)                     # byte-identical replay
        self.assertIn("text/event-stream", ctype)
        st = cp.CACHE.stats()
        self.assertEqual((st["hits"], st["misses"], st["stored"]), (1, 1, 1))

    def test_regen_appends_and_hits_rotate(self):
        url = self.start()
        key = "|v1|S01|rot"
        post_json(url + "/completion", self.req(key=key))              # store variant A
        post_json(url + "/completion", self.req(key=key, regen=True))  # append variant B
        self.assertEqual(len(cp.CACHE.candidates(key)), 2)
        served = [sse_text(post_json(url + "/completion", self.req(key=key))[2])
                  for _ in range(4)]
        self.assertEqual(len(self.up.posts()), 2)                       # hits are free
        self.assertEqual(served[0], served[2])
        self.assertEqual(served[1], served[3])
        self.assertNotEqual(served[0], served[1])                       # rotation
        self.assertEqual(set(served), {"generation 1", "generation 2"})

    def test_shadow_records_but_always_forwards(self):
        url = self.start(shadow=True)
        key = "|v1|S02|shadow"
        for _ in range(3):
            post_json(url + "/completion", self.req(key=key))
        self.assertEqual(len(self.up.posts()), 3)          # never served from cache
        self.assertEqual(len(cp.CACHE.candidates(key)), 3)  # but every generation stored
        self.assertEqual(cp.CACHE.stats()["hits"], 0)
        self.assertTrue(cp.CACHE.stats()["shadow"])

    def test_stats_endpoint(self):
        url = self.start(tag="qwen-27b")
        post_json(url + "/completion", self.req(key="|v1|S01|a"))
        post_json(url + "/completion", self.req(key="|v1|S01|a"))
        stats = get_json(url + "/__cache__/stats")
        self.assertEqual(stats["keys"], 1)
        self.assertEqual(stats["variants_for_tag"], 1)
        self.assertEqual(stats["model_tag"], "qwen-27b")
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertFalse(stats["shadow"])

    def test_keyinfo_and_reject(self):
        url = self.start()
        key = "|v1|S03|rej"
        post_json(url + "/completion", self.req(key=key))
        post_json(url + "/completion", self.req(key=key, regen=True))
        info = get_json(url + "/__cache__/keyinfo?key=" + urllib.parse.quote(key))
        self.assertEqual(info["total"], 2)
        self.assertEqual(info["index"], 2)        # curator is looking at the regen
        # reject drops the variant last shown and fetches a replacement
        post_json(url + "/completion", self.req(key=key, reject=True))
        ids = get_json(url + "/__cache__/keyinfo?key=" + urllib.parse.quote(key))["ids"]
        self.assertNotIn(info["id"], ids)
        self.assertEqual(len(ids), 2)
        self.assertEqual(cp.CACHE.stats()["rejected"], 1)

    def test_persists_across_restart(self):
        url = self.start()
        post_json(url + "/completion", self.req(key="|v1|S01|p"))
        reloaded = cp.VariantCache(self.cache_path, model_tag="qwen")
        self.assertEqual(len(reloaded.candidates("|v1|S01|p")), 1)

    def test_sampler_overrides_applied_to_forwarded_body(self):
        url = self.start()
        cp.CLAMP_N_PREDICT = 16
        cp.REPEAT_PENALTY = 1.1
        post_json(url + "/completion", self.req(key="|v1|S01|s", n_predict=500))
        fwd = json.loads(self.up.posts()[0][2])
        self.assertEqual(fwd["n_predict"], 16)
        self.assertEqual(fwd["repeat_penalty"], 1.1)


class DegradedReplyNotCachedTest(unittest.TestCase):
    """Regression: fallback / error replies come back as HTTP 200 from the upstream
    proxies; cache-proxy used to store them as variants and replay them forever."""

    KEY = "|v1|S01|deg"

    def setUp(self):
        self._quiet = contextlib.redirect_stderr(io.StringIO())
        self._quiet.__enter__()
        self.tmp = tempfile.mkdtemp()
        self.up = ScriptedUpstream()
        cp.UPSTREAM = ("127.0.0.1", self.up.port)
        cp.UP_TIMEOUT = 10
        cp.REGEN_TEMP = 0.0
        cp.CLAMP_N_PREDICT = 0
        cp.REPEAT_PENALTY = 0.0
        cp.DRY_MULTIPLIER = 0.0
        cp.CACHE = cp.VariantCache(os.path.join(self.tmp, "c.json"), model_tag="t")
        self.proxy = RunningServer(cp.Handler)

    def tearDown(self):
        self.proxy.close()
        self.up.close()
        cp.CACHE = None
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._quiet.__exit__(None, None, None)

    def ask(self, **extra):
        body = {"prompt": "p", "stream": True, "cacheable": True, "cache_key": self.KEY}
        body.update(extra)
        return post_json(self.proxy.url + "/completion", body)

    def assert_passed_through_not_stored(self, status=200):
        got_status, _, raw = self.ask()
        self.assertEqual(got_status, status)
        self.assertEqual(raw, self.up.reply[2])               # client still gets it
        self.assertEqual(cp.CACHE.candidates(self.KEY), [])   # ...but nothing stored
        self.assertEqual(cp.CACHE.stats()["refused"], 1)
        # and the next request goes upstream again instead of replaying it
        self.ask()
        self.assertEqual(self.up.calls, 2)

    def test_degraded_header_not_stored_and_forwarded(self):
        self.up.reply = (200, {"X-LLM-Degraded": "fallback"},
                         sse_body({"content": "There is a pause."}, {"content": "", "stop": True}))
        status, _, raw = self.ask()
        self.assertEqual(cp.CACHE.candidates(self.KEY), [])
        # marker is relayed so anything further down the chain sees it too
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy.port, timeout=10)
        conn.request("POST", "/completion", body=json.dumps(
            {"prompt": "p", "cacheable": True, "cache_key": self.KEY}).encode(),
            headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.getheader("X-LLM-Degraded"), "fallback")

    def test_degraded_field_on_sse_stop_event_not_stored(self):
        self.up.reply = (200, {}, sse_body({"content": "In-voice fallback prose."},
                                           {"content": "", "stop": True, "degraded": "fallback"}))
        self.assert_passed_through_not_stored()

    def test_degraded_field_on_json_reply_not_stored(self):
        # openrouter-proxy style: non-streaming JSON with an error marker
        self.up.reply = (200, {"Content-Type": "application/json"},
                         json.dumps({"content": "[Error 429]", "degraded": "error"}).encode())
        self.assert_passed_through_not_stored()

    def test_unmarked_error_text_not_stored(self):
        # older upstreams that don't mark: still refuse "[Error …]" style text
        for body in (json.dumps({"content": "[Error 429]"}).encode(),
                     sse_body({"content": "[HTTP Error 529]", "stop": True}),
                     sse_body({"content": "[ERROR: Overloaded]", "stop": True}),
                     sse_body({"content": "Half a sentence"},
                              {"content": "[ERROR: stream reset]", "stop": True})):
            self.up.reply = (200, {}, body)
            self.ask()
        self.assertEqual(cp.CACHE.candidates(self.KEY), [])
        self.assertEqual(cp.CACHE.stats()["refused"], 4)

    def test_error_field_and_empty_replies_not_stored(self):
        for body in (json.dumps({"error": {"message": "boom"}}).encode(),
                     b"",
                     sse_body({"content": "", "stop": True}),
                     json.dumps({"content": "  "}).encode()):
            self.up.reply = (200, {}, body)
            self.ask()
        self.assertEqual(cp.CACHE.candidates(self.KEY), [])
        self.assertEqual(cp.CACHE.stats()["refused"], 4)

    def test_non_200_not_stored(self):
        self.up.reply = (503, {"Content-Type": "application/json"}, b'{"content": "busy"}')
        self.assert_passed_through_not_stored(status=503)

    def test_regen_and_shadow_also_refuse_degraded(self):
        self.up.reply = (200, {}, sse_body({"content": "x"},
                                           {"content": "", "stop": True, "degraded": "error"}))
        self.ask(regen=True)
        cp.CACHE.shadow = True
        self.ask()
        self.assertEqual(cp.CACHE.candidates(self.KEY), [])

    def test_plain_llama_server_reply_still_stored(self):
        # backward compat: a normal unmarked llama-server reply is cached as before,
        # including one that merely mentions an error mid-prose
        self.up.reply = (200, {}, sse_body({"content": "The word Error, said softly."},
                                           {"content": "", "stop": True, "stop_type": "eos"}))
        self.ask()
        self.assertEqual(len(cp.CACHE.candidates(self.KEY)), 1)
        self.assertEqual(cp.CACHE.stats()["refused"], 0)


class DegradedReasonUnitTest(unittest.TestCase):
    def test_classifier(self):
        r = cp._degraded_reason
        ok = sse_body({"content": "fine prose"}, {"content": "", "stop": True})
        self.assertIsNone(r(200, "", ok))
        self.assertIsNone(r(200, "", json.dumps({"content": "fine"}).encode()))
        self.assertIsNone(r(200, "", b"opaque but non-empty text"))
        self.assertIsNone(r(200, "", json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()))
        self.assertIsNotNone(r(500, "", ok))
        self.assertIsNotNone(r(200, "fallback", ok))
        self.assertIsNotNone(r(200, "", b"   \n"))
        self.assertIsNotNone(r(200, "", b"[Error: timed out]"))
        self.assertIsNotNone(r(200, "", json.dumps(
            {"choices": [{"message": {"content": "[Error 500]"}}]}).encode()))


class VariantCacheUnitTest(unittest.TestCase):
    def setUp(self):
        self._quiet = contextlib.redirect_stderr(io.StringIO())
        self._quiet.__enter__()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "c.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._quiet.__exit__(None, None, None)

    def test_model_tag_partition_prevents_cross_model_bleed(self):
        a = cp.VariantCache(self.path, model_tag="qwen")
        a.add("k", b"qwen text", "text/plain")
        b = cp.VariantCache(self.path, model_tag="claude")
        self.assertIsNone(b.serve("k"))
        b.add("k", b"claude text", "text/plain")
        self.assertEqual(b.serve("k")[0], b"claude text")
        self.assertEqual(cp.VariantCache(self.path, model_tag="qwen").serve("k")[0], b"qwen text")

    def test_byte_dedup_and_pool_cap(self):
        c = cp.VariantCache(self.path, model_tag="t", max_variants=2)
        self.assertEqual(c.add("k", b"x", "t"), 1)
        self.assertEqual(c.add("k", b"x", "t"), 1)   # identical bytes not duplicated
        self.assertEqual(c.add("k", b"y", "t"), 2)
        self.assertEqual(c.add("k", b"z", "t"), 2)   # pool full

    def test_scene_caps(self):
        c = cp.VariantCache(self.path, model_tag="t", max_variants=8, scene_caps={"S01": 1})
        c.add("|v|S01|h", b"a", "t")
        self.assertEqual(c.add("|v|S01|h", b"b", "t"), 1)
        c.add("|v|S02|h", b"a", "t")
        self.assertEqual(c.add("|v|S02|h", b"b", "t"), 2)

    def test_truth_gated_require_and_exclude(self):
        c = cp.VariantCache(self.path, model_tag="t")
        c.add("k", b"plain", "t")
        c.add("k", b"needs-fire", "t")
        c.add("k", b"not-if-kind", "t")
        vs = c.store["k"]["variants"]
        vs[1]["require"] = ["knows_fire"]
        vs[2]["exclude"] = ["kindness_exists"]
        # knows_fire + kindness: eligible = plain, needs-fire
        got = {c.serve("k", known={"knows_fire", "kindness_exists"})[0] for _ in range(4)}
        self.assertEqual(got, {b"plain", b"needs-fire"})
        # nothing known: eligible = plain, not-if-kind
        got = {c.serve("k", known=set())[0] for _ in range(4)}
        self.assertEqual(got, {b"plain", b"not-if-kind"})

    def test_gate_fallback_never_404s(self):
        c = cp.VariantCache(self.path, model_tag="t")
        c.add("k", b"gated", "t")
        c.store["k"]["variants"][0]["require"] = ["never_known"]
        self.assertEqual(c.serve("k", known=set())[0], b"gated")

    def test_known_truths_parser(self):
        prompt = ("## What I Know\n- [fire_is_hot] Fire burns (confidence: 90%)\n"
                  "  - [water_flows] Water flows\n## Teachable\n- not_bracketed — \"x\"\n")
        self.assertEqual(cp._known_truths(prompt), {"fire_is_hot", "water_flows"})


if __name__ == "__main__":
    unittest.main()
