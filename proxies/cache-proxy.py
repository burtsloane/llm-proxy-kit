#!/usr/bin/env python3
"""Standalone response cache as a chainable reverse proxy.

The cache used to live INSIDE claude-proxy, which coupled it to the Claude backend —
you literally could not cache Qwen/OpenRouter output. This makes the cache an
independent component on a port, like the other backends:

    client → cache-proxy :8897  ──hit──►  replay stored variant
                                ──miss─►  forward to --upstream :8898 → tee+store → client

It is BACKEND-AGNOSTIC: it peeks the request body only for `cacheable`/`cache_key`
(+ `regen`/`reject`), and stores/serves the raw response bytes verbatim — so it works
with llama-server, claude-proxy, openrouter-proxy, or a trace-proxy chain behind it.
Any request without `cacheable:true`+`cache_key` is pure pass-through (so other
callers — other apps, free-text requests that must stay live — are untouched).

Pooled variants: each key holds a LIST of responses; a hit rotates through them
(a per-key hits % N counter) so replays vary. `regen:true` bypasses the read and APPENDS a fresh variant
— the "fatten this node" button. `--model-tag` stamps variants and refuses to serve a
different backend's text (the no-cross-model-bleed guard, which matters more now that
backends are swappable; pair with a per-backend --cache file).

Usage:
    python3 cache-proxy.py --port 8897 --upstream http://localhost:8898 \
        --cache /path/to/cache-qwen.json --model-tag qwen-27b
    # omit --cache → pure transparent forwarder
    # --shadow → warm the store but always forward, never serve (safe warmup)
    # GET /__cache__/stats → JSON stats

Degraded replies are never stored: a non-200 status, an `X-LLM-Degraded` header,
a `"degraded"`/`"error"` field in the JSON body or any SSE event, an empty reply, or
text that starts with an `[Error…`/`[ERROR…`/`[HTTP Error…` signal (what the proxies
in this kit emit on failure). Such replies are still passed through to the client.
"""
import argparse, base64, json, os, re, sys, threading, time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STATS_PATH = "/__cache__/stats"
KEYINFO_PATH = "/__cache__/keyinfo"


class VariantCache:
    """key -> {variants: [{b64, ctype, tag, tokens, ts}], hits}. Atomic JSON persistence."""

    def __init__(self, path, model_tag="", max_variants=8, shadow=False, scene_caps=None):
        self.path = path
        self.model_tag = model_tag
        self.max_variants = max_variants
        # Per-scene pool-depth override (pyramid fill): scene → cap. Falls back to
        # max_variants for any scene not listed. Scene is cache-key field [2]
        # (key = |content_ver|prompt_ver|scene|beat_hash|truth_state).
        self.scene_caps = scene_caps or {}
        self.shadow = shadow
        self.lock = threading.Lock()
        self.store = {}
        self.last_served = {}   # cache_key -> variant id last handed to the client
                                # (curation reject targets this; single-curator assumption)
        self.hits = 0
        self.misses = 0
        self.stored = 0
        self.rejected = 0
        self.refused = 0        # degraded upstream replies passed through, not stored
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.store = data.get("entries", {}) if isinstance(data, dict) else {}
            # Backfill stable variant ids for entries written before id tracking.
            for e in self.store.values():
                nid = e.get("next_id", 0)
                for v in e.get("variants", []):
                    if "id" not in v:
                        v["id"] = nid
                        nid += 1
                e["next_id"] = nid
            n = sum(len(e.get("variants", [])) for e in self.store.values())
            print(f"[cache-proxy] loaded {len(self.store)} keys / {n} variants from {self.path}",
                  file=sys.stderr)
        except FileNotFoundError:
            print(f"[cache-proxy] new cache store at {self.path}", file=sys.stderr)
        except Exception as e:
            print(f"[cache-proxy] load failed ({e}); starting empty", file=sys.stderr)

    def _save_locked(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"entries": self.store}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[cache-proxy] save failed: {e}", file=sys.stderr)

    def candidates(self, key):
        """Variants for this key matching our model-tag (the no-bleed guard)."""
        e = self.store.get(key)
        if not e:
            return []
        return [v for v in e.get("variants", []) if v.get("tag", "") == self.model_tag]

    @staticmethod
    def _gate_ok(v, known):
        """Truth-gated variants (set by an offline curation tool):
        `require`: serve only if ALL these truth_ids are known to the user.
        `exclude`: never serve if ANY of these truth_ids is known.
        Keys stay state-free; gates filter WITHIN a key's pool at serve time using
        the knowledge state parsed from the request prompt itself."""
        known = known or set()
        if any(t not in known for t in (v.get("require") or [])):
            return False
        if any(t in known for t in (v.get("exclude") or [])):
            return False
        return True

    def serve(self, key, known=None):
        """Return (raw_bytes, ctype) for a rotated variant, or None. Records a hit.
        `known` = truth_ids parsed from the request prompt; filters gated variants.
        Fallback if the gates eliminate everything: ungated variants, then any."""
        with self.lock:
            e = self.store.get(key)
            if not e:
                return None
            cands = [v for v in e.get("variants", []) if v.get("tag", "") == self.model_tag]
            if not cands:
                return None
            eligible = [v for v in cands if self._gate_ok(v, known)]
            if not eligible:
                eligible = [v for v in cands
                            if not v.get("require") and not v.get("exclude")] or cands
            idx = e.get("hits", 0) % len(eligible)
            e["hits"] = e.get("hits", 0) + 1
            self.hits += 1
            v = eligible[idx]
            self.last_served[key] = v.get("id")   # what a curation reject would drop
            self._save_locked()  # persist the bumped hit counter
            return base64.b64decode(v["b64"]), v.get("ctype", "text/event-stream")

    def _cap_for(self, key):
        """Variant-pool cap for this key's scene (pyramid fill), else global default."""
        parts = key.split("|")
        scene = parts[2] if len(parts) > 2 else ""
        return self.scene_caps.get(scene, self.max_variants)

    def add(self, key, raw, ctype, tokens=0, regen=False):
        """Append a variant (dedup identical bytes). Returns the new variant count."""
        with self.lock:
            e = self.store.setdefault(key, {"variants": [], "hits": 0, "next_id": 0})
            b64 = base64.b64encode(raw).decode("ascii")
            existing = [v for v in e["variants"] if v.get("tag", "") == self.model_tag]
            if any(v["b64"] == b64 for v in existing):
                return len(existing)  # identical generation — don't duplicate (byte dedup)
            if len(existing) >= self._cap_for(key):
                return len(existing)  # pool full for this tag (per-scene cap)
            vid = e.get("next_id", 0)
            e["next_id"] = vid + 1
            e["variants"].append({"id": vid, "b64": b64, "ctype": ctype, "tag": self.model_tag,
                                  "tokens": tokens, "ts": int(time.time())})
            self.last_served[key] = vid   # the curator is now looking at this one
            if not regen:
                self.misses += 1
            self.stored += 1
            self._save_locked()
            return len([v for v in e["variants"] if v.get("tag", "") == self.model_tag])

    def reject_last(self, key):
        """Curation hard-reject: drop the variant the client is currently looking at
        (= the one this proxy last served/stored for this key). Returns True if dropped."""
        with self.lock:
            vid = self.last_served.get(key)
            e = self.store.get(key)
            if vid is None or not e:
                return False
            before = len(e["variants"])
            e["variants"] = [v for v in e["variants"] if v.get("id") != vid]
            self.last_served.pop(key, None)
            if len(e["variants"]) < before:
                self.rejected += 1
                self._save_locked()
                return True
            return False

    def stats(self):
        with self.lock:
            keys = len(self.store)
            variants = sum(len(self.candidates(k)) for k in self.store)
            return {"keys": keys, "variants_for_tag": variants, "model_tag": self.model_tag,
                    "hits": self.hits, "misses": self.misses, "stored": self.stored,
                    "rejected": self.rejected, "refused": self.refused,
                    "shadow": self.shadow,
                    "max_variants": self.max_variants}

    def keyinfo(self, key):
        """Per-key pool info for a curation UI counter: which variant the curator is
        currently looking at (= last_served) and how many there are. `index` is 1-based
        position among this tag's candidates (the same order they sit in the cache JSON,
        so the curator can find the one to edit); `total` is the pool size for this key."""
        with self.lock:
            cands = self.candidates(key)
            total = len(cands)
            vid = self.last_served.get(key)
            index = 0
            for i, v in enumerate(cands):
                if v.get("id") == vid:
                    index = i + 1
                    break
            return {"key": key, "index": index, "total": total, "id": vid,
                    "ids": [v.get("id") for v in cands], "model_tag": self.model_tag}


_KNOWN_RE = re.compile(r"^\s*- \[([A-Za-z0-9_]+)\]", re.M)

def _known_truths(prompt):
    """Truth_ids the user currently knows, parsed from the request prompt.
    The app's prompt builder renders each known fact as a line of the form
    `- [truth_id] statement` inside the system message — so the ids ride along in
    every request and the client needs no extra protocol. Other list styles in the
    prompt (e.g. `- id — "…"`, no brackets) don't match, so this line-anchored
    pattern only picks up the knowledge list."""
    return set(_KNOWN_RE.findall(prompt or ""))


DEGRADED_HEADER = "X-LLM-Degraded"
_ERROR_TEXT_RE = re.compile(r"^\s*\[(?:ERROR|Error|HTTP Error)\b")


def _degraded_reason(status, degraded_header, raw):
    """Why an upstream reply must NOT be stored as a variant, or None if it's fine.

    Upstream proxies in this kit answer failures with HTTP 200 + in-voice fallback
    or `[Error …]` text (so the end user never sees a raw error), and mark them with
    an `X-LLM-Degraded` header and/or a `degraded` field on the JSON body / final SSE
    event. A plain llama-server sends none of those markers, so its replies are
    judged on status, emptiness and error-text only (backward compatible)."""
    if status != 200:
        return f"status {status}"
    if degraded_header:
        return f"marked degraded ({degraded_header})"
    if not raw or not raw.strip():
        return "empty reply"
    try:
        text_in = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "undecodable reply"

    pieces = []
    events = []
    if any(ln.startswith("data:") for ln in text_in.splitlines()):
        for ln in text_in.splitlines():
            if not ln.startswith("data:"):
                continue
            data = ln[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    else:
        try:
            events.append(json.loads(text_in))
        except json.JSONDecodeError:
            pieces.append(text_in)          # opaque body: judge the raw text

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("degraded"):
            return f"marked degraded ({ev.get('degraded')})"
        if ev.get("error"):
            return "error field in reply"
        c = ev.get("content")
        if c is None and isinstance(ev.get("choices"), list) and ev["choices"]:
            ch = ev["choices"][0] or {}
            c = ((ch.get("message") or {}).get("content")
                 or (ch.get("delta") or {}).get("content") or ch.get("text"))
        if isinstance(c, str):
            if _ERROR_TEXT_RE.match(c):
                return f"error text: {c.strip()[:60]}"
            pieces.append(c)
    if events and not "".join(pieces).strip() and not any(
            isinstance(ev, dict) and ev.get("tool_calls") for ev in events):
        return "empty content"
    if pieces and _ERROR_TEXT_RE.match("".join(pieces)):
        return "error text"
    return None


CACHE = None          # VariantCache or None (pure pass-through)
UPSTREAM = None       # (host, port)
UP_TIMEOUT = 300
REGEN_TEMP = 0.0      # >0 overrides temperature on regen for cross-variant diversity
# Sampler overrides applied to forwarded generations (0/off = leave the request alone).
# None of these are part of the cache_key, so they never affect hit/miss — they only
# shape the *text* of newly generated variants. Used to keep Qwen output short and
# non-degenerate when the client hard-codes sampler settings in its request body
# (e.g. n_predict=500 / repeat_penalty=1.5) — clamp them here without touching the client.
CLAMP_N_PREDICT = 0    # >0 → cap n_predict at this value
REPEAT_PENALTY = 0.0   # >0 → override repeat_penalty (e.g. 1.1 when the client sends 1.5)
DRY_MULTIPLIER = 0.0   # >0 → enable the DRY sampler (anti list/cadence degeneration)
DRY_BASE = 1.75
DRY_ALLOWED_LENGTH = 2


def _apply_sampler_overrides(req, widen=False):
    """Rewrite llama.cpp sampling params on the forwarded body to keep output short
    and non-degenerate. Mutates and returns `req` if anything changed, else None.
    `widen` (regen/shadow) additionally bumps temperature to REGEN_TEMP for diversity."""
    changed = False
    if CLAMP_N_PREDICT > 0:
        cur = req.get("n_predict")
        capped = CLAMP_N_PREDICT if not isinstance(cur, int) else min(cur, CLAMP_N_PREDICT)
        if capped != cur:
            req["n_predict"] = capped
            changed = True
    if REPEAT_PENALTY > 0 and req.get("repeat_penalty") != REPEAT_PENALTY:
        req["repeat_penalty"] = REPEAT_PENALTY
        changed = True
    if DRY_MULTIPLIER > 0:
        req["dry_multiplier"] = DRY_MULTIPLIER
        req["dry_base"] = DRY_BASE
        req["dry_allowed_length"] = DRY_ALLOWED_LENGTH
        changed = True
    if widen and REGEN_TEMP > 0 and req.get("temperature") != REGEN_TEMP:
        req["temperature"] = REGEN_TEMP
        changed = True
    return req if changed else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet; we print our own lines
        pass

    # All verbs funnel through one handler.
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == STATS_PATH and CACHE is not None:
            body = json.dumps(CACHE.stats()).encode()
            self._send_head(200, "application/json", len(body))
            self.wfile.write(body)
            return
        # Curation counter (read-only): GET /__cache__/keyinfo?key=<cache_key>
        # → {index, total, id}. A curation UI polls this to show "variant N of T".
        if parsed.path == KEYINFO_PATH and CACHE is not None:
            qs = parse_qs(parsed.query)
            key = (qs.get("key") or [""])[0]
            body = json.dumps(CACHE.keyinfo(key)).encode()
            self._send_head(200, "application/json", len(body))
            self.wfile.write(body)
            return
        self._handle(b"")

    def do_OPTIONS(self):
        self._handle(b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        self._handle(body)

    # ── core ──────────────────────────────────────────────────────────
    def _handle(self, body):
        req = None
        if body:
            try:
                req = json.loads(body)
            except Exception:
                req = None

        cacheable = bool(isinstance(req, dict) and req.get("cacheable") and req.get("cache_key"))
        if not (CACHE is not None and cacheable):
            self._forward(body)            # pure pass-through
            return

        key = req["cache_key"]
        regen = bool(req.get("regen"))
        reject = bool(req.get("reject"))

        if reject:
            dropped = CACHE.reject_last(key)
            print(f"[cache-proxy] REJECT {key[:64]} "
                  f"({'dropped' if dropped else 'nothing to drop'})", file=sys.stderr)
            regen = True   # reject means "drop this and give me a replacement"

        if not regen and not CACHE.shadow:
            hit = CACHE.serve(key, known=_known_truths(req.get("prompt", "")))
            if hit:
                raw, ctype = hit
                self._send_head(200, ctype, len(raw))
                self.wfile.write(raw)
                print(f"[cache-proxy] HIT  {key[:64]}", file=sys.stderr)
                return

        # miss / regen / shadow → forward, tee, store. Apply sampler overrides (length
        # clamp / repeat-penalty / DRY) and, on regen/shadow, widen temperature so
        # repeated generations of one prompt diverge.
        fwd_body = body
        if isinstance(req, dict):
            tweaked = _apply_sampler_overrides(dict(req), widen=(regen or CACHE.shadow))
            if tweaked is not None:
                fwd_body = json.dumps(tweaked).encode()
        status, ctype, raw, degraded_hdr = self._forward(fwd_body, capture=True)
        why = _degraded_reason(status, degraded_hdr, raw)
        if why:
            with CACHE.lock:
                CACHE.refused += 1
            print(f"[cache-proxy] NOT STORED {key[:64]}  ({why})", file=sys.stderr)
        else:
            n = CACHE.add(key, raw, ctype, tokens=_count_tokens(raw), regen=regen)
            tag = "REGEN" if regen else ("SHADOW-STORE" if CACHE.shadow else "STORE")
            print(f"[cache-proxy] {tag} {key[:64]}  (variant {n})", file=sys.stderr)

    def _forward(self, body, capture=False):
        """Stream the request through to UPSTREAM and the response back to the client.
        Returns (status, content_type, raw_bytes|b'', degraded_header|'')."""
        host, port = UPSTREAM
        fwd_headers = {}
        ct = self.headers.get("Content-Type")
        if ct:
            fwd_headers["Content-Type"] = ct
        if body:
            fwd_headers["Content-Length"] = str(len(body))
        try:
            conn = http.client.HTTPConnection(host, port, timeout=UP_TIMEOUT)
            conn.request(self.command, self.path, body=body or None, headers=fwd_headers)
            resp = conn.getresponse()
        except Exception as e:
            msg = json.dumps({"error": f"upstream unreachable: {e}"}).encode()
            try:
                self._send_head(502, "application/json", len(msg))
                self.wfile.write(msg)
            except Exception:
                pass
            return 502, "application/json", b"", ""

        ctype = resp.getheader("Content-Type", "application/json")
        degraded = resp.getheader(DEGRADED_HEADER, "") or ""
        # streamed: no content-length, Connection: close. The degraded marker is
        # passed on so a cache/client further down the chain sees it too.
        self._send_head(resp.status, ctype, None,
                        extra={DEGRADED_HEADER: degraded} if degraded else None)
        buf = bytearray() if capture else None
        try:
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if buf is not None:
                    buf += chunk
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
        return resp.status, ctype, bytes(buf) if buf is not None else b"", degraded

    def _send_head(self, status, ctype, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()


def _count_tokens(raw):
    # cheap proxy for output size (bytes); real token count isn't needed for stats
    return len(raw)


def main():
    global CACHE, UPSTREAM, UP_TIMEOUT, REGEN_TEMP
    global CLAMP_N_PREDICT, REPEAT_PENALTY, DRY_MULTIPLIER, DRY_BASE, DRY_ALLOWED_LENGTH
    ap = argparse.ArgumentParser(description="Standalone response cache reverse proxy")
    ap.add_argument("--port", type=int, default=8897)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default="http://localhost:8898",
                    help="where cache misses go (any backend / visualizer chain)")
    ap.add_argument("--cache", default="", help="cache JSON path; omit for pure pass-through")
    ap.add_argument("--model-tag", default="", help="stamp/guard variants by backend identity")
    ap.add_argument("--max-variants", type=int, default=8,
                    help="default variant-pool cap (per scene override via --scene-caps)")
    ap.add_argument("--scene-caps", default="",
                    help="per-scene pool caps (scene = cache_key field 2), e.g. "
                         "'S01=20,S02=15,S03=12'; scenes not listed use --max-variants")
    ap.add_argument("--shadow", action="store_true",
                    help="measure + warm the store but always forward, never serve")
    ap.add_argument("--timeout", type=int, default=300, help="upstream timeout seconds")
    ap.add_argument("--regen-temp", type=float, default=0.0,
                    help="override temperature on regen for cross-variant diversity "
                         "(e.g. 0.95); 0 = leave the request's temperature unchanged")
    ap.add_argument("--clamp-n-predict", type=int, default=0,
                    help="cap n_predict on forwarded generations (e.g. 150); 0 = off. "
                         "Hard backstop against runaway-length output.")
    ap.add_argument("--repeat-penalty", type=float, default=0.0,
                    help="override repeat_penalty on forwarded generations (e.g. 1.1); "
                         "0 = off. A client-sent 1.5 over-penalizes and causes "
                         "novelty-spiral run-ons.")
    ap.add_argument("--dry-multiplier", type=float, default=0.0,
                    help="enable the DRY sampler (e.g. 0.8); 0 = off. Penalizes repeated "
                         "patterns/cadence (list run-ons) that repeat_penalty misses.")
    ap.add_argument("--dry-base", type=float, default=1.75, help="DRY base (default 1.75)")
    ap.add_argument("--dry-allowed-length", type=int, default=2,
                    help="DRY allowed repeat length before penalty (default 2)")
    args = ap.parse_args()

    up = urlparse(args.upstream)
    UPSTREAM = (up.hostname or "localhost", up.port or 80)
    UP_TIMEOUT = args.timeout
    REGEN_TEMP = args.regen_temp
    CLAMP_N_PREDICT = args.clamp_n_predict
    REPEAT_PENALTY = args.repeat_penalty
    DRY_MULTIPLIER = args.dry_multiplier
    DRY_BASE = args.dry_base
    DRY_ALLOWED_LENGTH = args.dry_allowed_length
    scene_caps = {}
    for tok in args.scene_caps.split(","):
        tok = tok.strip()
        if not tok:
            continue
        scene, _, cap = tok.partition("=")
        scene_caps[scene.strip()] = int(cap)
    if args.cache:
        CACHE = VariantCache(args.cache, model_tag=args.model_tag,
                             max_variants=args.max_variants, shadow=args.shadow,
                             scene_caps=scene_caps)
        if scene_caps:
            print(f"[cache-proxy] per-scene caps: {scene_caps} (default {args.max_variants})",
                  file=sys.stderr)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "pass-through" if CACHE is None else ("shadow" if args.shadow else "serving")
    print(f"[cache-proxy] :{args.port} → upstream {args.upstream}  "
          f"[{mode}{', tag=' + args.model_tag if args.model_tag else ''}]", file=sys.stderr)
    if CLAMP_N_PREDICT or REPEAT_PENALTY or DRY_MULTIPLIER:
        print(f"[cache-proxy] sampler overrides: "
              f"n_predict≤{CLAMP_N_PREDICT or '—'}  "
              f"repeat_penalty={REPEAT_PENALTY or '—'}  "
              f"dry_multiplier={DRY_MULTIPLIER or '—'}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[cache-proxy] bye", file=sys.stderr)


if __name__ == "__main__":
    main()
