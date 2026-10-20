# llm-proxy-kit

Small, zero-dependency HTTP proxies that you chain together in front of an LLM. Your
client speaks one protocol, the llama.cpp `/completion` API, and the chain decides what
answers it: a local `llama-server` (e.g. Qwen), Claude through the Anthropic Messages API,
or any model on OpenRouter. Along the way you can add a tracing proxy that prints every
prompt and streamed token, and a curated response cache. The cache keeps a pool of
reviewed variants for each request key and serves them in rotation. It never mixes output
from different models, and it lets you drop or regenerate individual variants. Each
proxy is a single Python file that uses only the standard library. Each one runs on its
own port and has its own flags, so you can swap backends by changing one `--upstream`
URL.

**Running publicly:** this chain sits in front of the interactive fiction game at
[cotc.casedefender.legal](https://cotc.casedefender.legal). Every menu click is answered
from the curated cache at $0 in API spend. Only free-text input reaches a live model, and
that traffic is capped by the Claude proxy's spend budget.

```
                    ┌──────────────┐     ┌──────────────┐  hit: replay a stored variant
 your app ────────► │ trace-proxy  │ ──► │ cache-proxy  │ ─────────────────────────────► (back to app)
 (llama.cpp         │ logs prompts │     │ pooled, gated│
  /completion)      │ + tokens     │     │ variants     │  miss / regen / shadow
                    └──────────────┘     └──────┬───────┘
                                                │ --upstream (any one of)
                         ┌──────────────────────┼───────────────────────┐
                         ▼                      ▼                       ▼
                  ┌──────────────┐     ┌──────────────────┐    ┌──────────────────┐
                  │ llama-server │     │ claude-proxy     │    │ openrouter-proxy │
                  │ (local Qwen) │     │ ChatML→Messages, │    │ ChatML→OpenAI    │
                  └──────────────┘     │ prompt caching,  │    │ chat format      │
                                       │ budget, fallback │    └──────────────────┘
                                       └──────────────────┘
```

Every link in the chain is optional. trace-proxy can sit anywhere, including between
cache-proxy and the backend. If you omit `--cache`, cache-proxy just forwards requests.

## The proxies

| File | What it does | Key flags |
|---|---|---|
| `proxies/trace-proxy.py` | Transparent forwarder. Logs each request (`n_predict`, temperature, stream flag, the tail of the prompt) and prints streamed tokens live, with TTFB/TTFT and total time. GETs such as `/health` pass through. Threaded, so a slow stream doesn't block other callers. | `--port` (8897), `--host` (127.0.0.1), `--backend` port (8898), `--backend-host` (localhost) |
| `proxies/cache-proxy.py` | Backend-agnostic response cache (see below). Requests without `cacheable`+`cache_key` pass straight through. | `--port` (8897), `--host`, `--upstream` (http://localhost:8898), `--cache PATH`, `--model-tag`, `--max-variants` (8), `--scene-caps`, `--shadow`, `--timeout`, `--regen-temp`, `--clamp-n-predict`, `--repeat-penalty`, `--dry-multiplier`/`--dry-base`/`--dry-allowed-length` |
| `proxies/claude-proxy.py` | Turns a llama.cpp `/completion` request into an Anthropic Messages streaming call and sends the reply back as llama.cpp-style SSE. Sets prompt-cache breakpoints on the system prompt, enforces a spend budget, and serves authored fallback text when a call fails. | `--port` (8081), `--host`, `--model`, `--effort`, `--no-cache`, `--max-per-hour`, `--max-per-day`, `--budget-ledger`, `--fallback-text`/`--fallback-file`, `--prefix`, `--notification-roles`/`--notification-label`, `--cache`/`--cache-shadow` (older in-process cache) |
| `proxies/openrouter-proxy.py` | Turns a llama.cpp `/completion` request into an OpenAI-style chat completion on OpenRouter, so any OpenRouter model can stand in for llama-server. Retries on 429. | `--port` (8897), `--host`, `--model` (default `deepseek/deepseek-chat-v3-0324`); env `OPENROUTER_REFERER`, `OPENROUTER_TITLE` |
| `proxies/start-claude-proxy.sh` | Convenience launcher for claude-proxy. | env `CLAUDE_PROXY_PORT`, `CLAUDE_MODEL`; extra args pass through |

Both API proxies expose `GET /health`. claude-proxy's health report includes the budget
ledger summary.

## Quickstart

You need Python 3.10 or later and nothing else. API keys are read only from the
environment.

**1. Local model (llama-server + Qwen), with tracing and a cache**

```sh
llama-server -m Qwen3-27B-Q4_K_M.gguf --port 8898 &
python3 proxies/cache-proxy.py --port 8897 --upstream http://localhost:8898 \
    --cache cache-qwen.json --model-tag qwen &
python3 proxies/trace-proxy.py --port 8890 --backend 8897 &
# point your app's llama.cpp server URL at http://localhost:8890
```

**2. Claude**

```sh
export ANTHROPIC_API_KEY=...
python3 proxies/claude-proxy.py --port 8081 --model claude-sonnet-4-20250514 \
    --max-per-hour 60 --max-per-day 400 --budget-ledger budget.json \
    --fallback-text "No answer comes. Try again in a little while." &
python3 proxies/cache-proxy.py --port 8897 --upstream http://localhost:8081 \
    --cache cache-claude.json --model-tag claude-sonnet-4 &
# point your app at http://localhost:8897
```

**3. OpenRouter**

```sh
export OPENROUTER_API_KEY=...
python3 proxies/openrouter-proxy.py --port 8899 --model deepseek/deepseek-chat-v3-0324 &
python3 proxies/cache-proxy.py --port 8897 --upstream http://localhost:8899 \
    --cache cache-deepseek.json --model-tag deepseek-v3 &
```

Several proxies default to port 8897, so pass `--port` explicitly whenever you chain
them.

## cache-proxy

The cache does not care which backend sits behind it. It reads only a few fields from
the JSON request body. Responses are stored and replayed as **raw bytes**, so SSE
streams, JSON replies, and any backend's output format all work the same way.

| Request field | Effect |
|---|---|
| `cacheable: true` + `cache_key: "<string>"` | Opts this request into the cache. Without both, the request passes through and nothing is stored. |
| `regen: true` | Skips the cache read, forwards upstream, and **appends** the new reply to the key's pool. This is how you add variants to a key. |
| `reject: true` | Drops the variant this proxy last served (or stored) for the key, then works like `regen` to fetch a replacement. |

- **Pools:** each key holds a list of variants. A hit picks `hits % N` from the eligible
  variants, using that key's own hit counter (shared by every client, not tracked per
  session), so repeated requests rotate through the pool instead of replaying the same
  text. Identical bytes are deduplicated. Pool size is capped by `--max-variants`, and
  `--scene-caps 'S01=20,S02=12'` sets per-group caps. The group is field 2 of a
  `|`-delimited key such as `|content_ver|prompt_ver|scene|hash`.
- **Model-tag partitioning:** `--model-tag` stamps every stored variant, and the proxy
  serves only variants with its own tag. If you switch backends behind the same cache
  file, it never serves text the other model wrote.
- **Shadow mode:** `--shadow` sends every request upstream and stores each generation,
  but never serves from the cache. Use it to warm a cache, or to fill pools offline by
  running a test driver against it, without changing what users see. If `--regen-temp`
  is set, generations are sent at that temperature, so a shadow fill produces varied
  pools.
- **Truth-gated variants:** a variant can carry `require: [ids]` (serve it only if all of
  these ids are known) and `exclude: [ids]` (never serve it if any of them is known). The
  known ids are read from the request prompt itself: lines of the form
  `- [some_id] statement`. Cache keys therefore stay independent of state while one pool
  still holds state-appropriate variants. If the gates rule out every variant, the proxy
  falls back to the ungated variants and then to any variant. A gated key never misses.
- **Degraded replies are never stored:** a reply is passed through to the client but
  not added to the pool if the upstream status isn't 200, it carries an
  `X-LLM-Degraded` header (forwarded on to the client), the JSON body or any SSE event
  has a `degraded` or `error` field, the content is empty, or the text starts with
  `[Error`, `[ERROR` or `[HTTP Error`. claude-proxy and openrouter-proxy mark their
  fallback and error replies this way. A plain llama-server sends no markers, so its
  normal replies are cached as before. Refusals are counted as `refused` in the stats.
- **Sampler overrides:** `--clamp-n-predict`, `--repeat-penalty`, and the `--dry-*` flags
  rewrite sampler settings on forwarded generations only. They never change the cache
  key. Use them when the client hard-codes sampler settings that degrade output.
- **Endpoints:** `GET /__cache__/stats` returns keys, variants for this tag, hits, misses,
  stored, rejected, refused (degraded replies not stored), and shadow/model-tag info.
  `GET /__cache__/keyinfo?key=<cache_key>` returns which variant the curator is looking
  at (`index` of `total`, plus variant ids). A curation UI can show "variant 3 of 7" with
  regenerate and reject buttons.
- **Persistence:** a JSON file, written atomically (tmp file + `os.replace`) on every
  change. Variants have stable ids, so you can review and edit the file offline.

## Design notes

- **Proxies, not a library.** Each concern (tracing, caching, provider translation,
  spending limits) is a separate process with a plain HTTP interface. The client does not
  change when you add, remove, or reorder pieces. A C++ game engine, a browser, and a
  Python script can share one cache without linking anything. Every piece can be tested
  with `curl`.
- **Why the cache doesn't depend on the backend.** The cache first lived inside
  claude-proxy, so local Qwen and OpenRouter output could not be cached. Making it a
  separate byte-level proxy means a cheap local model can fill pools and a stronger model
  can rewrite them later. The model tag keeps those sources apart in one file.
- **Failures read as normal output.** When claude-proxy is over budget, or the API fails
  before any text has streamed (bad key, spend cap reached, 4xx/5xx), it returns
  `--fallback-text` as an ordinary, well-formed completion. A raw `[HTTP Error 401]`
  never shows up in the UI. Because it is still HTTP 200, the reply is marked: an
  `X-LLM-Degraded: fallback` header when that is known before streaming starts (budget
  exhausted), and `"degraded": "fallback"` or `"error"` on the final SSE stop event in
  every case. cache-proxy uses those markers to keep fallbacks out of the cache. Write the fallback so it fails closed: if a caller treats
  model output as a verdict (for example, an LLM judge that answers PASS/FAIL), the
  fallback must not parse as a pass. Rate limits use capped exponential backoff with
  jitter and honor `Retry-After`. While waiting, the proxy sends empty keepalive chunks
  so client watchdogs don't time out.
- **Budget caps at the edge.** The app talks straight to the proxy chain, so the last
  proxy before the paid API is the only place that sees every billable call. claude-proxy
  enforces a sliding hourly window and a daily cap (UTC). The counts are stored in a
  ledger file, so a crash loop cannot reset them. Cache hits never count against the
  budget.
- **Tool calls.** A Claude `tool_use` block is streamed to the client as
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` in the text, with the
  arguments forwarded from the API's `input_json_delta` events as they arrive (`{}` for
  a tool with no input). Each response emits exactly one stop event.
- **Prompt caching.** claude-proxy treats any raw text before the first ChatML marker as
  system context. It splits that text on `## ` headers and places one cache breakpoint
  about two-thirds of the way through, so the stable part of the system prompt is billed
  at the cache-read rate. Models that reject sampling parameters automatically have
  `temperature` removed. The request never ends on an assistant turn, because a trailing
  assistant turn would be a prefill, which newer models reject.

## Tests

```sh
make test            # or: python3 -m unittest discover -s tests -v
```

The tests use only stdlib `unittest`. They start a fake llama.cpp upstream on an
ephemeral port and run each proxy against it. Network calls to Anthropic and OpenRouter
are mocked, so no keys or network access are needed.

## License

MIT. See [LICENSE](LICENSE).
