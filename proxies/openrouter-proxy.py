#!/usr/bin/env python3
"""
OpenRouter API Proxy for llama.cpp-compatible clients.

Translates between llama.cpp /completion format and OpenRouter's
OpenAI-compatible /v1/chat/completions format. This allows any
OpenRouter model (DeepSeek, Llama 4, Mistral, etc.) to be used
as a drop-in replacement for llama-server.

Usage:
    export OPENROUTER_API_KEY=...
    python3 openrouter-proxy.py --port 8897 --model deepseek/deepseek-chat-v3-0324

    Then point your client's llama.cpp server URL at http://localhost:8897

    Optional attribution headers (shown on openrouter.ai dashboards):
    OPENROUTER_REFERER (HTTP-Referer) and OPENROUTER_TITLE (X-Title).
"""

import argparse
import json
import os
import re
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
import ssl

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"

# OpenRouter app-attribution headers (optional). Configure via environment.
APP_REFERER = os.environ.get("OPENROUTER_REFERER", "")
APP_TITLE = os.environ.get("OPENROUTER_TITLE", "llm-proxy-kit")


def parse_chatml_prompt(prompt: str) -> list[dict]:
    """Parse ChatML-formatted prompt into OpenAI messages array."""
    messages = []
    system_content = ""

    # Text before first <|im_start|> is system content
    first_marker = prompt.find('<|im_start|>')
    if first_marker > 0:
        system_content = prompt[:first_marker].strip()
        prompt_remainder = prompt[first_marker:]
    elif first_marker == -1:
        system_content = prompt.strip()
        prompt_remainder = ""
    else:
        prompt_remainder = prompt

    # Parse ChatML markers
    pattern = r'<\|im_start\|>(\w+(?:-\w+)?)\n(.*?)(?:<\|im_end\|>|$)'
    matches = re.findall(pattern, prompt_remainder, re.DOTALL)

    for role, content in matches:
        content = content.strip()
        if not content:
            continue
        if role == "system":
            system_content = (system_content + "\n\n" + content).strip() if system_content else content
        elif role in ("user", "user-mentor"):
            clean = content.replace('/no_think', '').strip()
            messages.append({"role": "user", "content": clean})
        elif role == "assistant":
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            clean = clean.replace('/no_think', '').strip()
            if clean:
                messages.append({"role": "assistant", "content": clean})

    # Ensure alternation
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    # Must start with user
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "[Starting conversation]"})

    # Prepend system message
    if system_content:
        merged.insert(0, {"role": "system", "content": system_content})

    return merged


def call_openrouter(messages, api_key, model, max_tokens=1024, temperature=0.7):
    """Call OpenRouter API and return the response content."""
    request_body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    data = json.dumps(request_body).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if APP_REFERER:
        headers["HTTP-Referer"] = APP_REFERER
    if APP_TITLE:
        headers["X-Title"] = APP_TITLE

    req = urllib.request.Request(
        OPENROUTER_API_URL, data=data, headers=headers, method="POST"
    )
    ctx = ssl.create_default_context()

    # Retry on rate limit
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                return content, usage
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            if e.code == 429 and attempt < 2:
                print(f"[openrouter] Rate limited. Waiting 10s...", file=sys.stderr)
                time.sleep(10)
                req = urllib.request.Request(
                    OPENROUTER_API_URL, data=data, headers=headers, method="POST"
                )
                continue
            print(f"[openrouter] HTTP {e.code}: {body[:200]}", file=sys.stderr)
            return f"[Error {e.code}]", {}
        except Exception as e:
            print(f"[openrouter] Error: {e}", file=sys.stderr)
            return f"[Error: {e}]", {}
    return "[Error: max retries]", {}


# Session stats
class Stats:
    def __init__(self):
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, usage):
        self.requests += 1
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)

    def summary(self):
        cost_in = self.input_tokens * 0.14 / 1_000_000  # DeepSeek pricing
        cost_out = self.output_tokens * 0.28 / 1_000_000
        return (f"Requests: {self.requests} | "
                f"In: {self.input_tokens} | Out: {self.output_tokens} | "
                f"Est. cost: ${cost_in + cost_out:.4f}")

STATS = Stats()


class OpenRouterProxyHandler(BaseHTTPRequestHandler):
    model = DEFAULT_MODEL

    def log_message(self, format, *args):
        print(f"[openrouter] {args[0]}", file=sys.stderr)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok", "backend": "openrouter", "model": self.model,
            }).encode())
        else:
            self.send_error(404, f"Unknown endpoint: {self.path}")

    def do_POST(self):
        if self.path != "/completion":
            self.send_error(404, f"Unknown endpoint: {self.path}")
            return

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.send_error(500, "OPENROUTER_API_KEY not set")
            return

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

        max_tokens = request.get("n_predict", 1024)
        temperature = request.get("temperature", 0.7)

        messages = parse_chatml_prompt(prompt)

        total_chars = sum(len(m.get("content", "")) for m in messages)
        print(f"[openrouter] {len(messages)} messages ({total_chars} chars), "
              f"max_tokens={max_tokens}, model={self.model}", file=sys.stderr)

        content, usage = call_openrouter(
            messages, api_key, self.model, max_tokens, temperature
        )

        STATS.record(usage)

        # Strip thinking tags if present
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

        print(f"[openrouter] Response: {len(content)} chars, "
              f"{usage.get('completion_tokens', '?')} tokens", file=sys.stderr)
        print(f"[openrouter] {STATS.summary()}", file=sys.stderr)

        # Return in llama-server format
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps({"content": content})
        self.wfile.write(response.encode())


def main():
    parser = argparse.ArgumentParser(description="OpenRouter proxy for llama.cpp clients")
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        print("  export OPENROUTER_API_KEY=...", file=sys.stderr)
        print("  Get one at https://openrouter.ai/keys", file=sys.stderr)
        sys.exit(1)

    OpenRouterProxyHandler.model = args.model
    server = HTTPServer((args.host, args.port), OpenRouterProxyHandler)
    print(f"[openrouter] Starting on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[openrouter] Model: {args.model}", file=sys.stderr)
    print(f"[openrouter] Pricing: ~$0.14/1M input, ~$0.28/1M output (DeepSeek)", file=sys.stderr)
    print(f"[openrouter] Ready to proxy /completion requests", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[openrouter] Final stats: {STATS.summary()}", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
