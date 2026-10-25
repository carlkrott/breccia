#!/usr/bin/env python3
"""Phase 0.1 baseline: measure decode + prefill tok/s against an OpenAI-compatible endpoint.

Usage: bench_decode.py --url http://127.0.0.1:8160 --out results.json [--runs 3] [--gen-tokens 256]

Records per-request wall time, completion token throughput (tok/s), TTFT (streamed),
and scrapes /metrics before/after. Deterministic settings (temperature 0).
"""
import argparse, json, time, urllib.request, datetime, sys

def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.monotonic() - t0

def post_stream_first_token(url, payload, timeout=600):
    """Stream a completion; return (first_token_s, total_s, n_chunks, content_len)."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; chunks = 0; chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunks += 1
            if first is None and chunks >= 2:  # first chunk is usually role/header
                first = time.monotonic() - t0
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
                chars += len(delta.get("content") or "")
            except Exception:
                pass
    return first, time.monotonic() - t0, chunks, chars

def scrape_metrics(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
            return r.read().decode(errors="replace")
    except Exception as e:
        return f"# metrics scrape failed: {e}\n"

DECODE_PROMPT = (
    "Explain in detail how a mixture-of-experts transformer routes each token "
    "through a pool of 512 experts using top-k gating, including the math of the "
    "router softmax and the role of the shared expert. Be thorough and structured."
)

# ~600 token prefill prompt (repetitive prose is fine for eval throughput)
PREFILL_PROMPT = ("The quick brown fox jumps over the lazy dog while the system monitors "
                  "throughput, latency, memory pressure and cache behaviour across tiers. ") * 40

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8160")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    out = {"tag": args.tag, "url": args.url,
           "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
           "config": vars(args), "requests": [], "metrics_before": "", "metrics_after": ""}

    out["metrics_before"] = scrape_metrics(args.url)

    # Warmup (short, uncached)
    post(args.url + "/v1/completions", {"prompt": "warmup: 1+1=", "max_tokens": 8,
                                        "temperature": 0}, timeout=300)

    # 1) Decode runs (streamed: wall time excludes nothing; TTFT captured)
    for i in range(args.runs):
        first, wall, chunks, chars = post_stream_first_token(
            args.url + "/v1/completions",
            {"prompt": DECODE_PROMPT, "max_tokens": args.gen_tokens,
             "temperature": 0, "seed": 42})
        rec = {"kind": "decode", "run": i, "wall_s": round(wall, 3),
               "first_token_s": round(first, 3) if first is not None else None,
               "chunks": chunks, "approx_chars": chars}
        out["requests"].append(rec)
        print(f"decode run {i}: wall={wall:.2f}s ttft={rec['first_token_s']} chunks={chunks}", file=sys.stderr)

    # 2) Prefill run
    t0 = time.monotonic()
    pre_body, pre_wall = post(args.url + "/v1/completions",
                              {"prompt": PREFILL_PROMPT, "max_tokens": 16,
                               "temperature": 0}, timeout=600)
    usage = pre_body.get("usage", {})
    out["requests"].append({"kind": "prefill", "wall_s": round(pre_wall, 3),
                            "usage": usage,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens")})
    print(f"prefill: wall={pre_wall:.2f}s usage={usage}", file=sys.stderr)

    out["metrics_after"] = scrape_metrics(args.url)
    out["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
