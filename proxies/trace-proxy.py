#!/usr/bin/env python3
"""
Transparent LLM debugging proxy.

Sits between your app and llama-server (or any proxy in this kit), logging
full request prompts and streaming response tokens to the console.

Usage:
    python3 trace-proxy.py                        # listen 8897, forward to 8898
    python3 trace-proxy.py --port 8897 --backend 8898
    python3 trace-proxy.py --backend-host gpu-box.local --backend 8897  # remote
    python3 trace-proxy.py --host 0.0.0.0         # listen on all interfaces (default 127.0.0.1)

The client connects to this proxy; the proxy forwards to the real backend.
"""

import argparse
import json
import sys
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# ANSI colors
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

backend_url = "http://localhost:8898"
request_counter = 0
counter_lock = threading.Lock()


def log(color, tag, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"{DIM}{ts}{RESET} {color}[{tag}]{RESET} {msg}", flush=True)


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default HTTP logging

    def do_POST(self):
        global request_counter
        with counter_lock:
            request_counter += 1
            req_id = request_counter

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        path = self.path

        # Parse and log the request
        is_stream = False
        try:
            j = json.loads(body)
            prompt = j.get("prompt", "")
            n_predict = j.get("n_predict", "?")
            temperature = j.get("temperature", "?")
            is_stream = j.get("stream", False)
            cache = j.get("cache_prompt", False)

            log(CYAN, f"REQ #{req_id}", f"POST {path}  n_predict={n_predict}  temp={temperature}  stream={is_stream}  cache={cache}")

            # Show last ~600 chars of prompt (the user/assistant turn)
            if len(prompt) > 600:
                log(DIM, f"REQ #{req_id}", f"prompt ({len(prompt)} chars) ...last 600:")
                # Find a good split point near the end
                tail = prompt[-600:]
                for line in tail.split("\n"):
                    print(f"  {DIM}{line}{RESET}", flush=True)
            else:
                log(DIM, f"REQ #{req_id}", f"prompt ({len(prompt)} chars):")
                for line in prompt.split("\n"):
                    print(f"  {DIM}{line}{RESET}", flush=True)
        except Exception:
            log(CYAN, f"REQ #{req_id}", f"POST {path} ({len(body)} bytes, not JSON)")

        # Forward to backend
        target = f"{backend_url}{path}"
        req = urllib.request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()

        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            log(RED, f"RSP #{req_id}", f"HTTP {e.code} from backend")
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
            return
        except Exception as e:
            log(RED, f"RSP #{req_id}", f"Backend error: {e}")
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b'{"error": "proxy: backend unreachable"}')
            return

        ttfb = time.time() - t0

        # Send response headers
        self.send_response(resp.status)
        for key, val in resp.getheaders():
            if key.lower() not in ("transfer-encoding", "connection"):
                self.send_header(key, val)
        self.end_headers()

        if is_stream:
            # Stream SSE, log tokens as they arrive
            log(GREEN, f"RSP #{req_id}", f"streaming (TTFB {ttfb*1000:.0f}ms)...")
            token_buf = ""
            total_tokens = 0
            first_token_time = None

            client_gone = False
            stopped = False
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                # Forward raw bytes to client (the client may disconnect early)
                if not client_gone:
                    try:
                        self.wfile.write(raw_line)
                        self.wfile.flush()
                    except BrokenPipeError:
                        client_gone = True
                        log(DIM, f"RSP #{req_id}", "(client disconnected, aborting backend)")
                        resp.close()  # close backend connection to stop generation
                        break

                if stopped:
                    # Past the stop event: keep forwarding (the blank line that
                    # terminates the final SSE event, plus anything trailing) but
                    # don't parse/log any more.
                    continue

                stripped = line.strip()
                if not stripped.startswith("data: "):
                    continue
                data = stripped[6:]
                if data == "[DONE]":
                    continue

                try:
                    tj = json.loads(data)
                    content = tj.get("content", "")
                    stop = tj.get("stop", False)

                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        total_tokens += 1
                        token_buf += content
                        # Print tokens inline (no newline)
                        color = DIM if client_gone else YELLOW
                        sys.stdout.write(f"{color}{content}{RESET}")
                        sys.stdout.flush()

                    if stop:
                        # Don't break here: the event's terminating blank line
                        # (and any trailing bytes) must still reach the client,
                        # or the body is truncated vs what the backend sent.
                        stopped = True
                except Exception:
                    pass

            # Finish the token line
            elapsed = time.time() - t0
            ttft = (first_token_time - t0) * 1000 if first_token_time else 0
            print(flush=True)  # newline after streamed tokens
            log(GREEN, f"RSP #{req_id}",
                f"done — {total_tokens} tokens, TTFT {ttft:.0f}ms, total {elapsed*1000:.0f}ms")
        else:
            # Non-streaming: read full response, log, forward
            resp_body = resp.read()
            elapsed = time.time() - t0
            self.wfile.write(resp_body)

            try:
                rj = json.loads(resp_body)
                content = rj.get("content", "")
                tokens_eval = rj.get("tokens_evaluated", 0)
                tokens_pred = rj.get("tokens_predicted", 0)
                log(GREEN, f"RSP #{req_id}",
                    f"{elapsed*1000:.0f}ms  eval={tokens_eval}  pred={tokens_pred}")
                if content:
                    preview = content[:200] + ("..." if len(content) > 200 else "")
                    log(DIM, f"RSP #{req_id}", preview)
            except Exception:
                log(GREEN, f"RSP #{req_id}", f"{elapsed*1000:.0f}ms ({len(resp_body)} bytes)")

    def do_GET(self):
        """Pass through GET requests (e.g., /health)."""
        target = f"{backend_url}{self.path}"
        try:
            resp = urllib.request.urlopen(target, timeout=10)
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(resp.read())
        except Exception as e:
            log(RED, "GET", f"{self.path} → {e}")
            self.send_response(502)
            self.end_headers()


def main():
    global backend_url

    parser = argparse.ArgumentParser(description="Transparent LLM debugging proxy")
    parser.add_argument("--port", type=int, default=8897, help="Listen port (default: 8897)")
    parser.add_argument("--backend", type=int, default=8898, help="Backend llama-server port (default: 8898)")
    parser.add_argument("--backend-host", default="localhost", help="Backend host (default: localhost)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on (default: 127.0.0.1; use 0.0.0.0 for all)")
    args = parser.parse_args()

    backend_url = f"http://{args.backend_host}:{args.backend}"

    print(f"\n{CYAN}  ╔══════════════════════════════════════╗{RESET}")
    print(f"{CYAN}  ║{RESET}  {BOLD}LLM Trace Proxy{RESET}                    {CYAN}║{RESET}")
    print(f"{CYAN}  ║{RESET}  :{args.port} → {backend_url}  {CYAN}║{RESET}")
    print(f"{CYAN}  ╚══════════════════════════════════════╝{RESET}\n")

    # Threaded so one long streaming generation doesn't block every other caller
    # (or a /health probe) behind it.
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.daemon_threads = True
    log(GREEN, "PROXY", f"Listening on {args.host}:{args.port}, forwarding to {backend_url}")
    log(DIM, "PROXY", "Start llama-server on the backend port, then point your client here\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log(DIM, "PROXY", "Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
