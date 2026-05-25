# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Client for the Llama-3.2-3B vLLM service.

Runs against the server launched by `service.sh` in this directory. Supports
two modes:

  --mode single
        Send one prompt, stream the response. Useful for sanity checking and
        for measuring batch=1 TTFT.

  --mode broadcast --users N
        Send the same prompt N times in parallel. Mirrors the b32 benchmark
        pattern when --users matches the server's BATCH_SIZE. Reports each
        user's TTFT and total time so the per-user / aggregate cost is
        visible.

Examples:
    # single-user (works against any BATCH_SIZE server)
    python client.py --mode single --prompt "Why is the sky blue?"

    # 32-user broadcast (server must be launched with BATCH_SIZE=32)
    python client.py --mode broadcast --users 32 --prompt "Why is the sky blue?"

The point of the broadcast mode is to reproduce the wall-clock TTFT slowness
documented in the prefill performance investigation: send N prompts at once,
observe the per-user TTFT grow roughly linearly with N rather than the
sub-linear scaling we'd ideally expect from batching on TT hardware.
"""

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

DEFAULT_PROMPT = "Why is the sky blue?"
DEFAULT_URL = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_MAX_TOKENS = 32


@dataclass
class RequestResult:
    user_id: int
    ttft_ms: Optional[float]
    total_ms: float
    tokens: int
    text: str
    error: Optional[str] = None


def stream_one(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    user_id: int,
    live_print: bool = False,
) -> RequestResult:
    """Send one streaming completion request and report TTFT + total time.

    When live_print is True, tokens are written to stdout as they arrive
    (with a flush after each) so the response appears incrementally.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "ignore_eos": True,
    }

    submitted = time.perf_counter()
    ttft_ms: Optional[float] = None
    tokens = 0
    out_chunks: List[str] = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                payload_str = line[len("data: ") :]
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choice = event.get("choices", [{}])[0]
                token = choice.get("text", "")
                if token:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - submitted) * 1000.0
                    tokens += 1
                    out_chunks.append(token)
                    if live_print:
                        sys.stdout.write(token)
                        sys.stdout.flush()
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time.perf_counter() - submitted) * 1000.0
        return RequestResult(
            user_id=user_id,
            ttft_ms=ttft_ms,
            total_ms=elapsed_ms,
            tokens=tokens,
            text="".join(out_chunks),
            error=str(e),
        )

    if live_print:
        sys.stdout.write("\n")
        sys.stdout.flush()
    total_ms = (time.perf_counter() - submitted) * 1000.0
    return RequestResult(
        user_id=user_id,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tokens=tokens,
        text="".join(out_chunks),
    )


def _decode_tps(result: RequestResult) -> Optional[float]:
    """Decode tokens-per-second, excluding the prefill / TTFT portion."""
    if result.ttft_ms is None or result.tokens < 2:
        return None
    decode_ms = result.total_ms - result.ttft_ms
    if decode_ms <= 0:
        return None
    # tokens after the first one were produced during decode_ms
    return (result.tokens - 1) / (decode_ms / 1000.0)


def _overall_tps(result: RequestResult) -> Optional[float]:
    """Overall tokens-per-second over the whole request (incl. TTFT)."""
    if result.tokens == 0 or result.total_ms <= 0:
        return None
    return result.tokens / (result.total_ms / 1000.0)


def run_single(args) -> None:
    print(f"Sending one prompt to {args.url}")
    print(f"  prompt: {args.prompt!r}")
    print(f"  max_tokens: {args.max_tokens}")
    print()
    print("--- response (streaming) ---")
    result = stream_one(
        args.url,
        args.model,
        args.prompt,
        args.max_tokens,
        user_id=0,
        live_print=True,
    )
    print("--- end ---")
    print()
    if result.error:
        print(f"Error: {result.error}")
        return
    ttft_str = f"{result.ttft_ms:.1f} ms" if result.ttft_ms else "n/a"
    overall = _overall_tps(result)
    decode = _decode_tps(result)
    print(f"TTFT     : {ttft_str}")
    print(f"Total    : {result.total_ms:.1f} ms  ({result.tokens} tokens)")
    if overall is not None:
        print(f"Overall  : {overall:.2f} tok/s")
    if decode is not None:
        print(f"Decode   : {decode:.2f} tok/s   (excludes prefill / TTFT)")


def run_broadcast(args) -> None:
    print(f"Broadcasting one prompt to {args.users} parallel users against {args.url}")
    print(f"  prompt: {args.prompt!r}")
    print(f"  max_tokens: {args.max_tokens}")
    stream_user = args.stream_user
    if 0 <= stream_user < args.users:
        print(f"  live-streaming user {stream_user}'s response to stdout below")
        print()
        print(f"--- user {stream_user} response (streaming) ---")
    else:
        stream_user = -1  # disable
        print()
    submitted = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [
            pool.submit(
                stream_one,
                args.url,
                args.model,
                args.prompt,
                args.max_tokens,
                user_id=i,
                live_print=(i == stream_user),
            )
            for i in range(args.users)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall_ms = (time.perf_counter() - submitted) * 1000.0
    if stream_user >= 0:
        print(f"--- end user {stream_user} ---")
        print()

    results.sort(key=lambda r: r.user_id)
    errors = [r for r in results if r.error]
    ok = [r for r in results if not r.error and r.ttft_ms is not None]

    if errors:
        print(f"{len(errors)} request(s) errored:")
        for r in errors[:3]:
            print(f"  user {r.user_id}: {r.error}")
        if len(errors) > 3:
            print(f"  ... ({len(errors) - 3} more)")
        print()

    if not ok:
        print("No successful responses.")
        return

    # Sort by TTFT so we can split the lead user (fastest first-token, represents
    # the prefill compute cost) from the trailing users (everyone else, who pay
    # an additional vLLM scheduling/delivery cost on top). This split is bimodal
    # in practice on the b32 prefill path — surfacing it separately is more
    # informative than a single mean.
    ok_by_ttft = sorted(ok, key=lambda r: r.ttft_ms)
    lead = ok_by_ttft[0]
    trail = ok_by_ttft[1:]

    total_tokens = sum(r.tokens for r in ok)
    system_tps = total_tokens / (wall_ms / 1000.0) if wall_ms > 0 else None

    def _fmt_group(label: str, results: List[RequestResult]) -> None:
        if not results:
            return
        ttfts = [r.ttft_ms for r in results]
        totals = [r.total_ms for r in results]
        decodes = [v for v in (_decode_tps(r) for r in results) if v is not None]
        overalls = [v for v in (_overall_tps(r) for r in results) if v is not None]
        print(f"{label} (n={len(results)})")
        if len(results) == 1:
            print(f"  TTFT         : {ttfts[0]:.1f} ms")
            print(f"  Total        : {totals[0]:.1f} ms")
            if decodes:
                print(f"  Decode tok/s : {decodes[0]:.2f}   (excludes prefill / TTFT)")
            if overalls:
                print(f"  Overall tok/s: {overalls[0]:.2f}")
        else:
            ttft_mean = sum(ttfts) / len(ttfts)
            print(
                f"  TTFT         : mean={ttft_mean:.1f}  min={min(ttfts):.1f}  max={max(ttfts):.1f} ms"
            )
            print(f"  Total        : mean={sum(totals) / len(totals):.1f} ms")
            if decodes:
                print(
                    f"  Decode tok/s : mean={sum(decodes) / len(decodes):.2f}  "
                    f"min={min(decodes):.2f}  max={max(decodes):.2f}   (excludes prefill / TTFT)"
                )
            if overalls:
                print(
                    f"  Overall tok/s: mean={sum(overalls) / len(overalls):.2f}  "
                    f"min={min(overalls):.2f}  max={max(overalls):.2f}"
                )

    print(f"Successful responses: {len(ok)} / {args.users}")
    print()
    _fmt_group(
        "Lead user  (fastest TTFT, ~= prefill compute cost)",
        [lead],
    )
    if trail:
        print()
        _fmt_group(
            f"Trailing users (the other {len(trail)}, pay scheduling/delivery overhead on top of prefill)",
            trail,
        )
    print()

    # Aggregate stats across the whole population — useful as a sanity check
    # but the lead vs. trailing split above is the more meaningful view.
    all_ttfts = [r.ttft_ms for r in ok]
    all_totals = [r.total_ms for r in ok]
    print(
        f"All users  (n={len(ok)}): "
        f"TTFT mean={sum(all_ttfts) / len(all_ttfts):.1f} ms  "
        f"Total mean={sum(all_totals) / len(all_totals):.1f} ms"
    )
    if system_tps is not None:
        print(
            f"System tok/s      : {system_tps:.2f}   "
            f"(= sum of generated tokens across all {len(ok)} users / wall-clock)"
        )
    print(f"Wall-clock for all {args.users} requests: {wall_ms:.1f} ms")

    if args.show_responses:
        print()
        lead_id = lead.user_id
        for r in ok:
            preview = r.text.replace("\n", " ")[:80]
            decode = _decode_tps(r)
            decode_str = f"  decode={decode:.1f} tok/s" if decode else ""
            tag = " (lead)" if r.user_id == lead_id else ""
            print(
                f"  user {r.user_id:>2}{tag}: TTFT={r.ttft_ms:>6.1f} ms{decode_str}  {preview!r}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=["single", "broadcast"],
        default="single",
        help="single: one prompt; broadcast: same prompt to --users parallel clients.",
    )
    p.add_argument(
        "--users",
        type=int,
        default=32,
        help="Broadcast user count (only used in broadcast mode).",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--show-responses",
        action="store_true",
        help="In broadcast mode, print each user's TTFT, decode tok/s and an 80-char response preview.",
    )
    p.add_argument(
        "--stream-user",
        type=int,
        default=-1,
        help="In broadcast mode, live-print this user's tokens to stdout as they arrive (default -1 = disabled).",
    )
    args = p.parse_args()

    if args.mode == "single":
        run_single(args)
    else:
        run_broadcast(args)


if __name__ == "__main__":
    main()
