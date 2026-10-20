#!/usr/bin/env python3
"""
Claude API Proxy for llama.cpp-compatible clients.

Translates between llama.cpp /completion format and Claude /v1/messages format.
This lets any client that already talks to llama-server use Claude as a backend.

Features:
- Prompt caching for reduced latency and cost
- Streaming SSE responses
- ChatML to Claude message format conversion

Usage:
    export ANTHROPIC_API_KEY=...
    python3 claude-proxy.py --port 8081

    Then point your client's llama.cpp server URL at http://localhost:8081

Cache Breakpoint Strategy (assumes the client orders context stable → volatile):
  1. System prompt - stable for entire session
  2. Static world/domain info - stable for the scenario
  3. Low-frequency state - changes occasionally
  4. (no cache) - conversation history + high-frequency state, changes every call
See build_cached_system_blocks() for how the breakpoint is placed.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Generator
import urllib.request
import urllib.error
import ssl

# Rate limit handling. The old behavior — a flat 30s blocking sleep per 429 —
# blew through callers' request timeouts (e.g. a caller with a 30-45s budget per
# call), so a single upstream 429 turned into a dropped result downstream. Now: honor the server's Retry-After when present,
# otherwise short exponential backoff (0.5,1,2,4,8s), each wait CAPPED so it can't
# exceed a typical caller timeout, with jitter so concurrent callers don't
# re-collide. More retries (each short) ride out transient rate limits.
MAX_RETRIES = 6
RATE_LIMIT_BASE_WAIT = 0.5      # first backoff; doubles each retry
RATE_LIMIT_MAX_WAIT = 8.0       # cap on any single wait (keep < caller timeouts)
RATE_LIMIT_WAIT_SECONDS = 30    # legacy flat value (retained for env/back-compat)

# Claude API configuration
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Good balance of speed and capability
ANTHROPIC_VERSION = "2023-06-01"

# output_config.effort, when the caller asks for it (--effort). Empty = omit the
# field and take the API default (`high`).
CLAUDE_EFFORT = ""

# Model families that REMOVED temperature/top_p/top_k. Sending a sampling
# parameter to one of these returns 400 invalid_request_error. Prefix match, so
# both the bare alias (`claude-opus-5`) and any dated/suffixed spelling match.
SAMPLING_PARAM_FREE_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def model_rejects_sampling_params(model: str) -> bool:
    """True when `model` 400s on temperature/top_p/top_k."""
    return any(model.startswith(prefix) for prefix in SAMPLING_PARAM_FREE_PREFIXES)

# Enable prompt caching
ENABLE_CACHING = True

# Optional prefix for context validation (loaded from file)
SYSTEM_PREFIX = ""

# Authored fallback served instead of an upstream call when the budget is
# exhausted or the API fails terminally (spend cap tripped, key revoked, ...).
# The end user sees in-voice text, never a raw API error. Write it in your app's
# voice and set it with --fallback-text / --fallback-file. Design it to FAIL
# CLOSED: if any caller parses model output as a verdict (e.g. an LLM-as-judge
# gate expecting "PASS"), the fallback must not parse as a pass.
FALLBACK_TEXT = ("There is a pause, and no answer comes. "
                 "Try again in a little while.")

# Custom ChatML roles carrying out-of-band notifications from the client's
# environment (not the human user). Mapped to labelled user turns because Claude
# has no such role. Override with --notification-roles / --notification-label.
NOTIFICATION_ROLES = ("notification", "system_notification")
NOTIFICATION_LABEL = "[Notification]"

# Set in main() when --max-per-hour/--max-per-day are given.
BUDGET = None


class Budget:
    """Upstream-call budget: sliding hourly window + a persistent daily/total
    ledger. This is the spend firewall for public deploys — the app talks
    straight to the proxy chain, so nothing in front of it can enforce a cap. Limits of
    0 mean unlimited. The ledger file survives restarts so a crash loop can't
    reset the daily count."""

    def __init__(self, per_hour=0, per_day=0, ledger_path=None):
        self.per_hour = per_hour
        self.per_day = per_day
        self.ledger_path = ledger_path
        self._hour_window = []          # monotonic timestamps of recent calls
        self._lock = threading.Lock()
        self._day = self._today()
        self._day_calls = 0
        self._total_calls = 0
        if ledger_path and os.path.exists(ledger_path):
            try:
                with open(ledger_path) as f:
                    led = json.load(f)
                if led.get("day") == self._day:
                    self._day_calls = int(led.get("day_calls", 0))
                self._total_calls = int(led.get("total_calls", 0))
            except Exception as e:
                print(f"[claude-proxy] WARNING: bad budget ledger, starting fresh ({e})",
                      file=sys.stderr)

    @staticmethod
    def _today():
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _save_locked(self):
        if not self.ledger_path:
            return
        tmp = self.ledger_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"day": self._day, "day_calls": self._day_calls,
                           "total_calls": self._total_calls}, f)
            os.replace(tmp, self.ledger_path)
        except Exception as e:
            print(f"[claude-proxy] WARNING: could not write budget ledger ({e})",
                  file=sys.stderr)

    def allow(self):
        """Atomically check-and-count one upstream call attempt."""
        now = time.monotonic()
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day = today
                self._day_calls = 0
            self._hour_window = [t for t in self._hour_window if now - t < 3600]
            if self.per_hour and len(self._hour_window) >= self.per_hour:
                print(f"[claude-proxy] BUDGET: hourly cap {self.per_hour} reached",
                      file=sys.stderr)
                return False
            if self.per_day and self._day_calls >= self.per_day:
                print(f"[claude-proxy] BUDGET: daily cap {self.per_day} reached",
                      file=sys.stderr)
                return False
            self._hour_window.append(now)
            self._day_calls += 1
            self._total_calls += 1
            self._save_locked()
            return True

    def summary(self):
        with self._lock:
            return {"day": self._day, "day_calls": self._day_calls,
                    "per_day": self.per_day, "hour_calls": len(self._hour_window),
                    "per_hour": self.per_hour, "total_calls": self._total_calls}

# Session-level cache statistics
class CacheStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.requests = 0
        self.total_input_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_output_tokens = 0

    def record(self, input_tokens: int, cache_read: int, cache_creation: int, output_tokens: int):
        self.requests += 1
        self.total_input_tokens += input_tokens
        self.total_cache_read_tokens += cache_read
        self.total_cache_creation_tokens += cache_creation
        self.total_output_tokens += output_tokens

    def summary(self) -> str:
        if self.requests == 0:
            return "No requests yet"

        total_would_be = self.total_input_tokens + self.total_cache_read_tokens + self.total_cache_creation_tokens
        cache_hit_rate = (self.total_cache_read_tokens / total_would_be * 100) if total_would_be > 0 else 0

        # Cache savings: cached tokens cost 0.1x on read, 1.25x on write
        # vs 1x for uncached. So savings = cache_read * 0.9 - cache_creation * 0.25
        estimated_savings_pct = 0
        if total_would_be > 0:
            # Without cache: all tokens at 1x
            without_cache = total_would_be
            # With cache: input at 1x, cache_read at 0.1x, cache_creation at 1.25x
            with_cache = self.total_input_tokens + (self.total_cache_read_tokens * 0.1) + (self.total_cache_creation_tokens * 1.25)
            estimated_savings_pct = ((without_cache - with_cache) / without_cache) * 100

        return (f"Requests: {self.requests} | "
                f"Input: {self.total_input_tokens} | "
                f"Cache read: {self.total_cache_read_tokens} | "
                f"Cache write: {self.total_cache_creation_tokens} | "
                f"Output: {self.total_output_tokens} | "
                f"Hit rate: {cache_hit_rate:.1f}% | "
                f"Est. savings: {estimated_savings_pct:.1f}%")

CACHE_STATS = CacheStats()


# ── Response cache (legacy, single-entry) ───────────────────────────────────
# A SHARED, content-addressed cache of full responses, keyed on a semantic key
# the client computes (e.g. prompt version + scene + action + bucketed state).
# On a hit we replay the stored text as SSE and skip the upstream call entirely
# — saving the full output-token cost.
#
# This is the original in-process cache. cache-proxy.py supersedes it (backend-
# agnostic, pooled variants, curation endpoints); it is kept for single-process
# deployments.
#
# SAFETY: only requests that arrive with `cacheable: true` AND a `cache_key` are
# ever touched. Every other caller (other apps, plain llama.cpp clients,
# free-text requests that must stay live) is unaffected — pure passthrough.
# The cache is OFF unless the proxy is started with --cache PATH.
class NarrationCache:
    def __init__(self, path: str, shadow: bool = False, model: str = ""):
        self.path = path
        self.shadow = shadow            # measure hit/miss + warm store, but never SERVE
        self.model = model              # never serve text generated by a different model
        self.lock = threading.Lock()
        self.store = {}                 # cache_key -> {text, model, output_tokens, hits}
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("model") in ("", self.model, None):
                self.store = data.get("entries", {})
            print(f"[claude-proxy] Narration cache loaded: {len(self.store)} entries "
                  f"from {self.path}", file=sys.stderr)
        except FileNotFoundError:
            print(f"[claude-proxy] Narration cache: new store at {self.path}", file=sys.stderr)
        except Exception as e:
            print(f"[claude-proxy] Narration cache load failed ({e}); starting empty", file=sys.stderr)

    def _save_locked(self):
        # caller holds self.lock. Atomic write.
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"model": self.model, "entries": self.store}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[claude-proxy] Narration cache save failed: {e}", file=sys.stderr)

    def get(self, key: str):
        with self.lock:
            entry = self.store.get(key)
            if entry and entry.get("model") in ("", self.model, None):
                return entry
            return None

    def put(self, key: str, text: str, output_tokens: int):
        if not text:
            return
        with self.lock:
            self.store[key] = {"text": text, "model": self.model,
                               "output_tokens": output_tokens,
                               "hits": self.store.get(key, {}).get("hits", 0)}
            self._save_locked()

    def record_hit(self, entry: dict):
        with self.lock:
            self.hits += 1
            self.tokens_saved += entry.get("output_tokens", 0)
            entry["hits"] = entry.get("hits", 0) + 1

    def record_miss(self):
        with self.lock:
            self.misses += 1

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        mode = "SHADOW" if self.shadow else "live"
        return (f"NarrationCache[{mode}]: {len(self.store)} entries | "
                f"hits {self.hits} / miss {self.misses} ({rate:.0f}% hit) | "
                f"output tokens saved ~{self.tokens_saved}")


NARRATION_CACHE = None  # set by main() when --cache is given


def replay_cached_sse(text: str) -> Generator[str, None, None]:
    """Emit stored text as llama.cpp-format SSE, mirroring stream_claude_response."""
    # Chunk for a streamed feel; size is cosmetic since it's instant.
    step = 120
    for i in range(0, len(text), step):
        yield f"data: {json.dumps({'content': text[i:i+step]})}\n\n"
    yield f"data: {json.dumps({'content': '', 'stop': True})}\n\n"


def parse_chatml_prompt(prompt: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a ChatML-formatted prompt into system blocks and messages array.

    Returns:
        (system_blocks, messages_array)

    system_blocks is a list of dicts for Claude's system array format:
        [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]

    Note: some llama.cpp clients send their fixed context blocks (including the
    system prompt) as raw text WITHOUT ChatML markers; only the conversation
    history gets <|im_start|> markers. So we treat any text BEFORE the first
    <|im_start|> as system content.
    """
    system_content = ""
    messages = []

    # First, check for raw text before the first <|im_start|> marker
    # This is where such clients put the system prompt and other fixed context
    first_marker_pos = prompt.find('<|im_start|>')
    if first_marker_pos > 0:
        # Everything before the first marker is system content
        system_content = prompt[:first_marker_pos].strip()
        prompt_remainder = prompt[first_marker_pos:]
    elif first_marker_pos == -1:
        # No ChatML markers at all - treat entire prompt as system content
        system_content = prompt.strip()
        prompt_remainder = ""
    else:
        # Marker is at position 0
        prompt_remainder = prompt

    # Split by im_start markers
    # Pattern: <|im_start|>role\ncontent<|im_end|>
    pattern = r'<\|im_start\|>(\w+(?:-\w+)?)\n(.*?)(?:<\|im_end\|>|$)'
    matches = re.findall(pattern, prompt_remainder, re.DOTALL)

    for role, content in matches:
        content = content.strip()
        if not content:
            continue

        if role == "system":
            # Append to any raw system content we found
            if system_content:
                system_content += "\n\n" + content
            else:
                system_content = content
        elif role in ("user", "user-mentor"):
            # Strip /no_think directive (Qwen3-specific, meaningless for Claude)
            clean = content.replace('/no_think', '').strip()
            messages.append({"role": "user", "content": clean})
        elif role == "assistant":
            # Strip Qwen3 <think> blocks — they confuse Claude into generating think output
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # Also strip /no_think directive (Qwen3-specific)
            clean = clean.replace('/no_think', '').strip()
            if clean:
                messages.append({"role": "assistant", "content": clean})
        elif role in ("tool", "tool_response"):
            # Tool results - include as user messages.
            # `tool_response` is an alternate role name some client serializers
            # emit; it used to fall off the end of this chain and be dropped
            # silently, with two consequences: the model never saw the tool result
            # it was supposed to react to, and the message list ended on the
            # assistant's own tool_call turn — i.e. a prefill.
            messages.append({"role": "user", "content": f"[Tool Result]: {content}"})
        elif role in NOTIFICATION_ROLES:
            # The environment's own voice, not the user's. Some clients emit custom
            # ChatML roles for out-of-band events (system advisories, automated
            # alerts, saved notifications). llama-server consumes them as raw ChatML
            # roles; Claude has no such role, so they have to land as user turns or
            # that whole channel is invisible on this backend — which silently
            # breaks A/B parity between a local-model run and a Claude run.
            # The label matters: it keeps the environment distinguishable from the
            # human user, so the model does not read machinery chatter as an order.
            # One label for all such roles: notification content usually already
            # self-labels, so a second distinct prefix would just double up.
            messages.append({"role": "user", "content": f"{NOTIFICATION_LABEL}: {content}"})

    # Ensure messages alternate properly (Claude requires user/assistant alternation)
    merged_messages = []
    for msg in messages:
        if merged_messages and merged_messages[-1]["role"] == msg["role"]:
            merged_messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged_messages.append(msg)

    # Claude requires the conversation to start with a user message
    if merged_messages and merged_messages[0]["role"] == "assistant":
        merged_messages.insert(0, {"role": "user", "content": "[Starting conversation]"})

    # ...and it must not END on one. A trailing assistant turn is an assistant
    # PREFILL, which returns 400 on Opus 4.6+ and the 5 family (it is still
    # accepted on Haiku 4.5, which is why this went unnoticed for a long time).
    # Any unhandled role above can strand the list here, so close it defensively
    # rather than trusting the role map to stay complete.
    if merged_messages and merged_messages[-1]["role"] == "assistant":
        merged_messages.append({"role": "user", "content": "[Continue]"})

    # Build system blocks with cache control
    # Try to split system content into cacheable sections
    system_blocks = build_cached_system_blocks(system_content)

    return system_blocks, merged_messages


def build_cached_system_blocks(system_content: str) -> list[dict]:
    """
    Split system content into blocks with strategic cache breakpoints.

    We look for natural section markers like:
    - "## " markdown headers
    - "---" separators
    - Known section names

    Cache strategy:
    - Block 1: Core instructions (## YOUR SITUATION, ## YOUR CAPABILITIES, etc.)
    - Block 2: Tools/format definitions (## TOOLS, ## RESPONSE FORMAT)
    - Block 3: Everything else (scenario-specific, may change)
    """
    # Prepend validation prefix if configured
    if SYSTEM_PREFIX:
        system_content = SYSTEM_PREFIX + system_content

    if not system_content:
        return []

    if not ENABLE_CACHING:
        return [{"type": "text", "text": system_content}]

    # Try to find natural split points
    # Look for major section headers
    sections = []
    current_section = []

    lines = system_content.split('\n')
    for line in lines:
        # Check for major headers that might indicate cache boundaries
        if line.startswith('## ') and current_section:
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append('\n'.join(current_section))

    # If we have multiple sections, group them strategically
    if len(sections) >= 3:
        # Cache the first ~2/3 as stable, leave the rest dynamic
        cache_boundary = (len(sections) * 2) // 3
        cached_content = '\n'.join(sections[:cache_boundary])
        dynamic_content = '\n'.join(sections[cache_boundary:])

        blocks = [
            {
                "type": "text",
                "text": cached_content,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        if dynamic_content.strip():
            blocks.append({
                "type": "text",
                "text": dynamic_content
            })
        return blocks
    else:
        # Cache the entire system prompt
        return [
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"}
            }
        ]


def stream_claude_response(system_blocks: list[dict], messages: list[dict], api_key: str,
                           max_tokens: int = 1024, temperature: float = 0.7,
                           capture: dict | None = None) -> Generator[str, None, None]:
    """
    Stream a response from Claude API, yielding llama.cpp-compatible SSE chunks.

    If `capture` is provided, the full assembled assistant text is stored in
    capture['text'] on successful completion (used by the response cache to store
    the value after a cache miss).
    """
    # Build Claude request
    request_body = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": messages
    }

    # Sampling parameters were REMOVED on Opus 4.7 and every model after it —
    # sending temperature/top_p/top_k is a hard 400, not a warning. The llama.cpp
    # request shape we translate from always carries a temperature (default 0.7),
    # so it has to be dropped here or every request to a current model fails.
    # Steer those models with prompting (and `effort`) instead.
    if not model_rejects_sampling_params(CLAUDE_MODEL):
        request_body["temperature"] = temperature

    # Thinking depth / overall token spend. Only sent when explicitly asked for:
    # `effort` errors on models that predate it (Sonnet 4.5, Haiku 4.5), and the
    # API default (`high`) is the right choice for batch work. Live/interactive
    # callers want `low` or `medium` — on Opus 5 thinking is ON by default, so
    # `high` costs real latency per turn.
    #
    # Deliberately NOT offering a thinking-disabled switch: on Opus 5,
    # `thinking: {"type": "disabled"}` can make the model write a tool call into
    # its visible text instead of emitting a tool-use block — the turn succeeds,
    # the call never runs, and nothing errors. Lower the effort instead.
    if CLAUDE_EFFORT:
        request_body["output_config"] = {"effort": CLAUDE_EFFORT}

    # Add system blocks (with cache control)
    if system_blocks:
        request_body["system"] = system_blocks

    # Prepare request
    data = json.dumps(request_body).encode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    # Enable prompt caching beta
    if ENABLE_CACHING:
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"

    req = urllib.request.Request(
        CLAUDE_API_URL,
        data=data,
        headers=headers,
        method="POST"
    )

    # Create SSL context
    ctx = ssl.create_default_context()

    # Track usage for this request
    request_usage = {
        "input_tokens": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "output_tokens": 0
    }

    # Brief request summary
    total_system_chars = sum(len(b.get('text', '')) for b in system_blocks)
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[claude-proxy] REQUEST #{CACHE_STATS.requests + 1}", file=sys.stderr)
    print(f"[claude-proxy] {len(system_blocks)} system blocks ({total_system_chars} chars), {len(messages)} messages", file=sys.stderr)

    # Show the last few messages being sent (most relevant for debugging)
    print(f"[claude-proxy] --- MESSAGES (last 3) ---", file=sys.stderr)
    for i, msg in enumerate(messages[-3:]):
        role = msg.get('role', '?')
        content = msg.get('content', '')
        # Truncate long content but show start and end
        if len(content) > 500:
            display = content[:250] + f"\n... [{len(content)} chars total] ...\n" + content[-250:]
        else:
            display = content
        msg_index = len(messages) - 3 + i if len(messages) > 3 else i
        print(f"[claude-proxy] [{msg_index}] {role}:", file=sys.stderr)
        for line in display.split('\n')[:20]:  # Max 20 lines
            print(f"[claude-proxy]     {line}", file=sys.stderr)
        if display.count('\n') > 20:
            print(f"[claude-proxy]     ... (truncated)", file=sys.stderr)
    print(f"[claude-proxy] --- END MESSAGES ---", file=sys.stderr)

    # Accumulate response for debugging
    accumulated_response = []

    # Retry loop for rate limit handling
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # 120 second timeout for initial connection - Claude can take a while with large contexts
            with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
                # Read bytes and decode carefully to avoid splitting UTF-8 characters
                byte_buffer = b""
                text_buffer = ""

                while True:
                    # Use read1() for true streaming - returns data immediately as it arrives
                    # instead of read() which may buffer waiting for more data
                    chunk = response.read1(4096) if hasattr(response, 'read1') else response.read(4096)
                    if not chunk:
                        break

                    byte_buffer += chunk

                    # Try to decode as much as possible, keeping incomplete UTF-8 at end
                    try:
                        text_buffer += byte_buffer.decode('utf-8')
                        byte_buffer = b""
                    except UnicodeDecodeError as e:
                        # Decode up to the error point, keep the rest
                        if e.start > 0:
                            text_buffer += byte_buffer[:e.start].decode('utf-8')
                            byte_buffer = byte_buffer[e.start:]

                    # Process complete SSE events
                    while "\n\n" in text_buffer:
                        event_end = text_buffer.index("\n\n")
                        event_data = text_buffer[:event_end]
                        text_buffer = text_buffer[event_end + 2:]

                        # Parse SSE event
                        event_type = None
                        event_json = None

                        for line in event_data.split("\n"):
                            if line.startswith("event: "):
                                event_type = line[7:]
                            elif line.startswith("data: "):
                                try:
                                    event_json = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    continue

                        if not event_json:
                            continue

                        # Track current block type for proper tag closing
                        current_block_type = getattr(stream_claude_response, '_current_block_type', None)

                        # Convert Claude events to llama.cpp format
                        if event_type == "content_block_delta":
                            delta = event_json.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    accumulated_response.append(text)
                                    # Debug: log if we see tool_call in text
                                    if "<tool_call>" in text or "</tool_call>" in text or "}}" in text:
                                        print(f"[claude-proxy] TEXT contains tool syntax: {repr(text)}", file=sys.stderr)
                                    chunk_data = f"data: {json.dumps({'content': text})}\n\n"
                                    yield chunk_data
                            elif delta_type == "thinking_delta":
                                text = delta.get("thinking", "")
                                if text:
                                    accumulated_response.append(text)
                                    chunk_data = f"data: {json.dumps({'content': text})}\n\n"
                                    yield chunk_data

                        elif event_type == "content_block_start":
                            block = event_json.get("content_block", {})
                            block_type = block.get("type")
                            stream_claude_response._current_block_type = block_type

                            if block_type == "thinking":
                                accumulated_response.append('<think>')
                                yield f"data: {json.dumps({'content': '<think>'})}\n\n"
                            elif block_type == "tool_use":
                                tool_name = block.get("name", "")
                                accumulated_response.append(f'<tool_call>{{"name": "{tool_name}", "arguments": ')
                                yield f"data: {json.dumps({'content': f'<tool_call>{{\"name\": \"{tool_name}\", \"arguments\": '})}\n\n"

                        elif event_type == "content_block_stop":
                            # Close tags based on what type of block just ended
                            block_type = getattr(stream_claude_response, '_current_block_type', None)
                            if block_type == "thinking":
                                accumulated_response.append('</think>')
                                yield f"data: {json.dumps({'content': '</think>'})}\n\n"
                            elif block_type == "tool_use":
                                # Close the tool_call tag (arguments JSON was already streamed)
                                accumulated_response.append('}}</tool_call>')
                                yield f"data: {json.dumps({'content': '}}</tool_call>'})}\n\n"
                            stream_claude_response._current_block_type = None

                        elif event_type == "message_start":
                            # Capture initial usage (input tokens, cache stats)
                            message = event_json.get("message", {})
                            usage = message.get("usage", {})
                            request_usage["input_tokens"] = usage.get("input_tokens", 0)
                            request_usage["cache_read"] = usage.get("cache_read_input_tokens", 0)
                            request_usage["cache_creation"] = usage.get("cache_creation_input_tokens", 0)

                            # Log cache status for this request
                            if request_usage["cache_read"] > 0:
                                print(f"[claude-proxy] CACHE HIT: {request_usage['cache_read']} tokens from cache", file=sys.stderr)
                            if request_usage["cache_creation"] > 0:
                                print(f"[claude-proxy] CACHE WRITE: {request_usage['cache_creation']} tokens cached", file=sys.stderr)
                            print(f"[claude-proxy] Input tokens (uncached): {request_usage['input_tokens']}", file=sys.stderr)

                        elif event_type == "message_stop":
                            yield f"data: {json.dumps({'content': '', 'stop': True})}\n\n"

                        elif event_type == "message_delta":
                            delta = event_json.get("delta", {})
                            if delta.get("stop_reason"):
                                # Capture final output token count
                                usage = event_json.get("usage", {})
                                request_usage["output_tokens"] = usage.get("output_tokens", 0)

                                # Record to session stats
                                CACHE_STATS.record(
                                    request_usage["input_tokens"],
                                    request_usage["cache_read"],
                                    request_usage["cache_creation"],
                                    request_usage["output_tokens"]
                                )

                                # Full response summary
                                full_response = ''.join(accumulated_response)
                                if capture is not None:
                                    capture['text'] = full_response
                                    capture['output_tokens'] = request_usage['output_tokens']
                                print(f"[claude-proxy] --- RESPONSE ({len(full_response)} chars, {request_usage['output_tokens']} tokens) ---", file=sys.stderr)
                                # Show the full response (or truncated if very long)
                                if len(full_response) > 2000:
                                    print(f"[claude-proxy] {full_response[:1000]}", file=sys.stderr)
                                    print(f"[claude-proxy] ... [{len(full_response)} chars total] ...", file=sys.stderr)
                                    print(f"[claude-proxy] {full_response[-1000:]}", file=sys.stderr)
                                else:
                                    print(f"[claude-proxy] {full_response}", file=sys.stderr)
                                print(f"[claude-proxy] --- END RESPONSE ---", file=sys.stderr)
                                print(f"[claude-proxy] Cache stats: {CACHE_STATS.summary()}", file=sys.stderr)
                                print(f"{'='*80}\n", file=sys.stderr)

                                yield f"data: {json.dumps({'content': '', 'stop': True})}\n\n"

                        elif event_type == "error":
                            error_msg = event_json.get("error", {}).get("message", "Unknown error")
                            print(f"[claude-proxy] API Error: {error_msg}", file=sys.stderr)
                            yield f"data: {json.dumps({'content': f'[ERROR: {error_msg}]', 'stop': True})}\n\n"

                # Success - exit the retry loop
                return

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')

            # Check for rate limit error (429)
            if e.code == 429 and retries < MAX_RETRIES - 1:
                retries += 1
                # Prefer the server's Retry-After (seconds); else exponential
                # backoff. Cap so one wait can't exceed a caller's timeout, and
                # add jitter so concurrent callers don't retry in lockstep.
                retry_after = e.headers.get("retry-after") or e.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after is not None else 0.0
                except (TypeError, ValueError):
                    wait = 0.0
                if wait <= 0:
                    wait = RATE_LIMIT_BASE_WAIT * (2 ** (retries - 1))
                wait = min(wait, RATE_LIMIT_MAX_WAIT)
                wait += random.uniform(0.0, 0.25 * wait)
                print(f"[claude-proxy] Rate limited (429); retry {retries}/{MAX_RETRIES-1} in {wait:.1f}s", file=sys.stderr)
                print(f"[claude-proxy] Error details: {error_body}", file=sys.stderr)

                # Keepalive pings during the wait (keeps client watchdogs from firing);
                # empty content is ignored by llama.cpp-format consumers.
                waited = 0.0
                while waited < wait:
                    step = min(1.0, wait - waited)
                    yield f"data: {json.dumps({'content': ''})}\n\n"
                    time.sleep(step)
                    waited += step

                # Recreate the request since the previous one was consumed
                req = urllib.request.Request(
                    CLAUDE_API_URL,
                    data=data,
                    headers=headers,
                    method="POST"
                )
                continue  # Retry
            else:
                # Non-retriable error or out of retries
                print(f"[claude-proxy] HTTP Error {e.code}: {error_body}", file=sys.stderr)
                yield f"data: {json.dumps({'content': f'[HTTP Error {e.code}]', 'stop': True})}\n\n"
                return

        except Exception as e:
            print(f"[claude-proxy] Error: {e}", file=sys.stderr)
            yield f"data: {json.dumps({'content': f'[Error: {str(e)}]', 'stop': True})}\n\n"
            return


class ClaudeProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies /completion requests to Claude API."""

    def log_message(self, format, *args):
        """Custom logging with timestamp."""
        print(f"[claude-proxy] {args[0]}", file=sys.stderr)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        """Handle POST requests to /completion."""
        if self.path != "/completion":
            self.send_error(404, f"Unknown endpoint: {self.path}")
            return

        # Get API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.send_error(500, "ANTHROPIC_API_KEY environment variable not set")
            return

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            request = json.loads(body)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        prompt = request.get("prompt", "")
        if not prompt:
            self.send_error(400, "Missing 'prompt' field")
            return

        # Extract request params (pass through from llama.cpp format)
        max_tokens = request.get("n_predict", 1024)
        temperature = request.get("temperature", 0.7)

        # ── Response cache (opt-in per request) ─────────────────────────────
        # Only engaged when the caller marks the request cacheable AND supplies a
        # semantic cache_key (e.g. only for fixed, menu-driven actions whose prompt
        # repeats). All other callers fall straight through untouched.
        cache_key = request.get("cache_key")
        cacheable = bool(request.get("cacheable")) and bool(cache_key)
        cache = NARRATION_CACHE
        if cache and cacheable:
            entry = cache.get(cache_key)
            if entry and not cache.shadow:
                cache.record_hit(entry)
                print(f"[claude-proxy] NARRATION CACHE HIT {cache_key[:80]} "
                      f"(~{entry.get('output_tokens',0)} output tokens saved)", file=sys.stderr)
                print(f"[claude-proxy] {cache.summary()}", file=sys.stderr)
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for chunk in replay_cached_sse(entry["text"]):
                        self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
                return
            # miss (or shadow): fall through to upstream, then store
            cache.record_miss()
            if entry and cache.shadow:
                print(f"[claude-proxy] NARRATION CACHE [shadow] WOULD-HIT {cache_key[:80]}", file=sys.stderr)
            else:
                print(f"[claude-proxy] NARRATION CACHE MISS {cache_key[:80]}", file=sys.stderr)

        # ── Spend firewall ──────────────────────────────────────────────
        # Every request past this point is an upstream (billable) call. On
        # cap-hit, serve the authored fallback as a normal SSE completion —
        # the user sees in-voice text, the client sees a well-formed completion.
        if BUDGET and not BUDGET.allow():
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for chunk in replay_cached_sse(FALLBACK_TEXT):
                    self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                pass
            return

        # Clear any stale state from previous interrupted requests
        stream_claude_response._current_block_type = None

        # Parse ChatML prompt into system blocks and messages
        system_blocks, messages = parse_chatml_prompt(prompt)

        # Brief summary
        print(f"[claude-proxy] Received {len(prompt)} chars, parsed {len(system_blocks)} system blocks, {len(messages)} messages, max_tokens={max_tokens}, temp={temperature}", file=sys.stderr)

        # If no messages, create a minimal one
        if not messages:
            # Extract something useful from the prompt
            messages = [{"role": "user", "content": "Please respond based on the system instructions."}]

        # Send SSE response headers
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Stream response (capture the assembled text so we can store it on a cache miss)
        capture = {} if (cache and cacheable) else None
        try:
            # Terminal API failures surface as '[HTTP Error NNN]' / '[ERROR: ...]'
            # content chunks. If one arrives before any real prose, swallow it
            # and serve the authored fallback instead — a tripped console spend
            # cap must read as in-voice text, not as an error string in the output.
            got_prose = False
            for chunk in stream_claude_response(system_blocks, messages, api_key,
                                                max_tokens, temperature, capture=capture):
                content = ""
                if chunk.startswith("data: "):
                    try:
                        content = json.loads(chunk[6:]).get("content", "")
                    except (json.JSONDecodeError, AttributeError):
                        pass
                if (not got_prose and isinstance(content, str)
                        and content.startswith(("[ERROR:", "[HTTP Error", "[Error:"))):
                    print(f"[claude-proxy] Terminal API failure before any prose "
                          f"({content[:60]}) — serving authored fallback", file=sys.stderr)
                    for fb in replay_cached_sse(FALLBACK_TEXT):
                        self.wfile.write(fb.encode('utf-8'))
                    self.wfile.flush()
                    capture = None  # never cache the fallback
                    break
                if content and not str(content).startswith(("[ERROR:", "[HTTP Error", "[Error:")):
                    got_prose = True
                self.wfile.write(chunk.encode('utf-8'))
                self.wfile.flush()
            # Ensure all data is flushed before connection closes
            self.wfile.flush()
            if cache and cacheable and capture and capture.get('text'):
                cache.put(cache_key, capture['text'], capture.get('output_tokens', 0))
                print(f"[claude-proxy] NARRATION CACHE STORE {cache_key[:80]}", file=sys.stderr)
            time.sleep(0.5)  # Longer delay to let client process all buffered data
        except BrokenPipeError:
            print("[claude-proxy] Client disconnected (BrokenPipe)", file=sys.stderr)
        except Exception as e:
            print(f"[claude-proxy] Streaming error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "backend": "claude",
                "model": CLAUDE_MODEL,
                "caching": ENABLE_CACHING,
                "narration_cache": (None if NARRATION_CACHE is None else {
                    "enabled": True, "shadow": NARRATION_CACHE.shadow,
                    "entries": len(NARRATION_CACHE.store),
                    "hits": NARRATION_CACHE.hits, "misses": NARRATION_CACHE.misses,
                }),
                "budget": (None if BUDGET is None else BUDGET.summary()),
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404, f"Unknown endpoint: {self.path}")


def main():
    global CLAUDE_MODEL, CLAUDE_EFFORT, ENABLE_CACHING, SYSTEM_PREFIX
    global FALLBACK_TEXT, NOTIFICATION_ROLES, NOTIFICATION_LABEL

    parser = argparse.ArgumentParser(description="Claude API proxy for llama.cpp clients")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on (default: 8081)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--model", default=CLAUDE_MODEL, help=f"Claude model to use (default: {CLAUDE_MODEL})")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help="output_config.effort. Omitted by default (API default is 'high'). "
                             "Use low/medium for live interactive play; not supported on models "
                             "older than Opus 4.5")
    parser.add_argument("--no-cache", action="store_true", help="Disable prompt caching")
    parser.add_argument("--cache", type=str, metavar="PATH",
                        help="Enable the narration RESPONSE cache, persisted at PATH "
                             "(only affects requests marked cacheable with a cache_key)")
    parser.add_argument("--cache-shadow", action="store_true",
                        help="Shadow mode: measure/log narration-cache hit-miss and warm "
                             "the store, but always call upstream and never serve from cache")
    parser.add_argument("--prefix", type=str, help="Path to system prompt prefix file (for context validation mode)")
    parser.add_argument("--max-per-hour", type=int, default=0,
                        help="Max upstream API calls per sliding hour; 0 = unlimited. "
                             "Over-cap requests get the authored fallback text.")
    parser.add_argument("--max-per-day", type=int, default=0,
                        help="Max upstream API calls per UTC day; 0 = unlimited.")
    parser.add_argument("--budget-ledger", type=str, default=None, metavar="PATH",
                        help="JSON file persisting daily/total call counts across restarts.")
    parser.add_argument("--fallback-text", type=str, default=None,
                        help="Authored text served (as a normal completion) when the budget "
                             "is exhausted or the API fails before any output.")
    parser.add_argument("--fallback-file", type=str, default=None, metavar="PATH",
                        help="Read the fallback text from PATH (overrides --fallback-text).")
    parser.add_argument("--notification-roles", type=str, default=None,
                        help="Comma-separated custom ChatML roles to forward as labelled "
                             f"user turns (default: {','.join(NOTIFICATION_ROLES)})")
    parser.add_argument("--notification-label", type=str, default=None,
                        help=f"Label prefixed to notification turns (default: {NOTIFICATION_LABEL})")
    args = parser.parse_args()

    CLAUDE_MODEL = args.model
    CLAUDE_EFFORT = args.effort or ""
    ENABLE_CACHING = not args.no_cache
    if args.fallback_text is not None:
        FALLBACK_TEXT = args.fallback_text
    if args.fallback_file:
        with open(args.fallback_file, 'r') as f:
            FALLBACK_TEXT = f.read().strip()
    if args.notification_roles is not None:
        NOTIFICATION_ROLES = tuple(r.strip() for r in args.notification_roles.split(",") if r.strip())
    if args.notification_label is not None:
        NOTIFICATION_LABEL = args.notification_label

    global NARRATION_CACHE
    if args.cache:
        NARRATION_CACHE = NarrationCache(args.cache, shadow=args.cache_shadow, model=CLAUDE_MODEL)

    global BUDGET
    if args.max_per_hour or args.max_per_day:
        BUDGET = Budget(per_hour=args.max_per_hour, per_day=args.max_per_day,
                        ledger_path=args.budget_ledger)
        print(f"[claude-proxy] Budget: {args.max_per_hour or '∞'}/hour, "
              f"{args.max_per_day or '∞'}/day"
              f"{f', ledger {args.budget_ledger}' if args.budget_ledger else ''}",
              file=sys.stderr)

    # Load prefix file if specified
    if args.prefix:
        try:
            with open(args.prefix, 'r') as f:
                SYSTEM_PREFIX = f.read()
            print(f"[claude-proxy] Loaded system prefix from {args.prefix} ({len(SYSTEM_PREFIX)} chars)", file=sys.stderr)
        except Exception as e:
            print(f"[claude-proxy] WARNING: Failed to load prefix file: {e}", file=sys.stderr)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    # Threaded: each /completion is handled on its own thread so concurrent
    # callers (e.g. a client that fans out a dozen judge calls per request)
    # run in parallel instead of serializing behind one thread — and a 429
    # back-off sleep blocks only its own request, not every queued one.
    # daemon_threads lets the process exit cleanly without joining in-flight
    # request threads.
    server = ThreadingHTTPServer((args.host, args.port), ClaudeProxyHandler)
    server.daemon_threads = True
    print(f"[claude-proxy] Starting on http://{args.host}:{args.port} (threaded)", file=sys.stderr)
    print(f"[claude-proxy] Model: {CLAUDE_MODEL}", file=sys.stderr)
    print(f"[claude-proxy] Prompt caching: {'enabled' if ENABLE_CACHING else 'disabled'}", file=sys.stderr)
    if NARRATION_CACHE:
        print(f"[claude-proxy] Narration response cache: ENABLED "
              f"({'shadow' if NARRATION_CACHE.shadow else 'live'}) at {NARRATION_CACHE.path}", file=sys.stderr)
    else:
        print(f"[claude-proxy] Narration response cache: disabled (no --cache)", file=sys.stderr)
    print(f"[claude-proxy] Context validation prefix: {'enabled' if SYSTEM_PREFIX else 'disabled'}", file=sys.stderr)
    print(f"[claude-proxy] Ready to proxy /completion requests to Claude API", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[claude-proxy] Shutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
