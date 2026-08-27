# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DEV-ONLY: turn a zimage_tier_probe run into the four evidence tables.

Reads the probe's log ([TIME]/[CMPL]/[TIER]/[DRAM] lines) plus probe.json and
prints the tables that go into the tt-xla issue verbatim.

    python tools/zimage_tier_report.py <run-dir> [--steps-published 50]

Table 1  inside the published "steady-state" pass  -- claims (a) and (b)
Table 2  across calls                              -- claim (c) and the plateau
Table 3  the three tiers                           -- needs run B for tier 1
Table 4  derived contamination at the published N  -- computed, no extra hardware
"""

import argparse
import json
import os
import re
import sys

TIME_RE = re.compile(r"^\[TIME\] (?P<label>.+?)\s{2,}(?P<sec>[\d.]+)s\s*$")
CMPL_RE = re.compile(
    r"^\[CMPL\] (?P<tag>\S+)\s+uncached=(?P<unc>\d+)\(\+(?P<dunc>-?\d+)\s*\)"
    r"\s+dynamo=(?P<dyn>\d+)\(\+(?P<ddyn>-?\d+)\s*\)"
)
TIER_RE = re.compile(r"^\[TIER\] (?P<tag>\S+)\s+.*verdict=(?P<verdict>\S+)")
DRAM_RE = re.compile(r"^\[DRAM\] (?P<tag>\S+)\s+(?P<rest>.+?)\s*$")
CALL_RE = re.compile(
    r"^\[CALL\] c(?P<call>\d+) wall\s+(?P<sec>[\d.]+)s\s+steps=(?P<steps>\d+)"
)


def parse_log(path):
    per_tag, times, calls = {}, [], {}
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            m = CMPL_RE.match(line)
            if m:
                per_tag.setdefault(m["tag"], {})["dunc"] = int(m["dunc"])
                per_tag[m["tag"]]["ddyn"] = int(m["ddyn"])
                continue
            m = TIER_RE.match(line)
            if m:
                per_tag.setdefault(m["tag"], {})["verdict"] = m["verdict"]
                continue
            m = DRAM_RE.match(line)
            if m:
                per_tag.setdefault(m["tag"], {})["dram"] = m["rest"]
                continue
            m = TIME_RE.match(line)
            if m:
                times.append((m["label"].strip(), float(m["sec"])))
                continue
            m = CALL_RE.match(line)
            if m:
                calls[int(m["call"])] = {
                    "wall": float(m["sec"]),
                    "steps": int(m["steps"]),
                }
    return per_tag, times, calls


def row(tag, label, sec, per_tag):
    info = per_tag.get(tag, {})
    dunc = info.get("dunc")
    return (
        f"| {label:<44} | {sec if sec is None else f'{sec:9.2f}':>9} | "
        f"{'n/a' if dunc is None else f'+{dunc}':>6} | {info.get('verdict', 'n/a'):<22} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps-published", type=int, default=50)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = args.log or os.path.join(args.run_dir, "probe.log")
    if not os.path.exists(log):
        sys.exit(f"no log at {log}")
    per_tag, times, calls = parse_log(log)

    recs = []
    pj = os.path.join(args.run_dir, "probe.json")
    if os.path.exists(pj):
        with open(pj) as handle:
            recs = json.load(handle)

    def stage(call, name):
        for r in recs:
            if r.get("call") == call and r.get("stage") == name:
                return r
        return {}

    ncalls = max(calls) if calls else 0
    steady = 2 if ncalls >= 2 else 1

    print(
        f"\n{'=' * 96}\nTABLE 1 -- inside call {steady}, the pass the driver publishes\n{'=' * 96}"
    )
    print(f"| {'stage':<44} | {'wall s':>9} | {'U':>6} | {'tier verdict':<22} |")
    print(f"|{'-' * 46}|{'-' * 11}|{'-' * 8}|{'-' * 24}|")
    te = stage(steady, "text_encoder")
    print(
        row(
            f"c{steady}/text_encoder/ran-positive",
            "text_encoder fwd #1 (positive)  COLD",
            te.get("cold_s"),
            per_tag,
        )
    )
    print(
        row(
            f"c{steady}/text_encoder/ran-negative",
            "text_encoder fwd #2 (negative)  WARM",
            te.get("warm_s"),
            per_tag,
        )
    )
    tf = stage(steady, "transformer")
    for i, s in enumerate(tf.get("steps_s", [])):
        kind = "COLD" if i == 0 else "WARM"
        print(
            row(
                f"c{steady}/transformer/step{i + 1}",
                f"transformer step {i + 1}              {kind}",
                s,
                per_tag,
            )
        )
    vae = stage(steady, "vae")
    print(
        row(
            f"c{steady}/vae/ran",
            "vae decode #1                   COLD",
            vae.get("cold_s"),
            per_tag,
        )
    )
    for k, s in enumerate(vae.get("warm_s", []) or []):
        print(
            row(
                f"c{steady}/vae/warm{k + 1}",
                f"vae decode #{k + 2}                   WARM",
                s,
                per_tag,
            )
        )

    print(
        f"\n{'=' * 96}\nTABLE 2 -- across calls: did the warmup pass buy anything?\n{'=' * 96}"
    )
    print(f"| {'quantity':<52} | {'value':>12} | {'expect':>8} |")
    print(f"|{'-' * 54}|{'-' * 14}|{'-' * 10}|")

    def ratio(a, b):
        return None if not a or not b else a / b

    te1, te2 = stage(1, "text_encoder"), stage(2, "text_encoder")
    r = ratio(te2.get("cold_s"), te1.get("cold_s"))
    print(
        f"| {'text_encoder fwd#1: call2 / call1':<52} | {('n/a' if r is None else f'{r:12.3f}'):>12} | {'~1.0':>8} |"
    )
    tf1, tf2 = stage(1, "transformer"), stage(2, "transformer")
    s1 = (tf1.get("steps_s") or [None])[0]
    s2 = (tf2.get("steps_s") or [None])[0]
    r = ratio(s2, s1)
    print(
        f"| {'transformer step 1: call2 / call1':<52} | {('n/a' if r is None else f'{r:12.3f}'):>12} | {'~1.0':>8} |"
    )
    if 3 in calls and 2 in calls:
        r = calls[3]["wall"] / calls[2]["wall"]
        print(
            f"| {'call 3 wall / call 2 wall  (PLATEAU PROOF)':<52} | {r:12.3f} | {'~1.0':>8} |"
        )
    if 1 in calls:
        print(
            f"| {'call 1 wall clock (the wasted warmup pass)':<52} | {calls[1]['wall']:12.2f} | {'s':>8} |"
        )
    tot = sum(c["wall"] for c in calls.values())
    if tot and 1 in calls:
        print(
            f"| {'warmup as fraction of a 2-pass benchmark':<52} | {calls[1]['wall'] / (calls[1]['wall'] + calls[2]['wall']) if 2 in calls else 0:12.3f} | {'':>8} |"
        )

    print(
        f"\n{'=' * 96}\nTABLE 3 -- the three tiers (tier 1 needs run B, FORCE_JIT)\n{'=' * 96}"
    )
    print(f"| {'tier':<8} | {'source':<44} | {'wall s':>9} | {'verdict':<16} |")
    print(f"|{'-' * 10}|{'-' * 46}|{'-' * 11}|{'-' * 18}|")
    print(
        f"| {'tier 1':<8} | {'run B FORCE_JIT text_encoder fwd#1':<44} | {'see run B':>9} | {'TIER1':<16} |"
    )
    v = per_tag.get(f"c{steady}/text_encoder/ran-positive", {}).get("verdict", "n/a")
    print(
        f"| {'tier 2':<8} | {'run A call' + str(steady) + ' text_encoder fwd#1':<44} | {(te.get('cold_s') or 0):9.2f} | {v:<16} |"
    )
    v = per_tag.get(f"c{steady}/text_encoder/ran-negative", {}).get("verdict", "n/a")
    print(
        f"| {'tier 3':<8} | {'run A call' + str(steady) + ' text_encoder fwd#2':<44} | {(te.get('warm_s') or 0):9.2f} | {v:<16} |"
    )

    print(
        f"\n{'=' * 96}\nTABLE 4 -- derived contamination of the published numbers at N={args.steps_published}\n{'=' * 96}"
    )
    steps = tf2.get("steps_s") or []
    if len(steps) >= 2:
        cold_step, warm_step = steps[0], sum(steps[1:]) / (len(steps) - 1)
        n = args.steps_published
        mean_all = (cold_step + (n - 1) * warm_step) / n
        cont = mean_all / warm_step - 1 if warm_step else 0
        print(f"  cold_step                     {cold_step:10.2f}s")
        print(f"  warm_step (mean of {len(steps) - 1})          {warm_step:10.2f}s")
        print(
            f"  step_mean_s as published      {mean_all:10.2f}s   <- (cold + {n - 1}*warm)/{n}"
        )
        print(
            f"  step_mean_s warm-only         {warm_step:10.2f}s   <- what it should report"
        )
        print(
            f"  CONTAMINATION                 {100 * cont:9.1f}%    <- the '~26%' claim"
        )
    else:
        print("  need >=2 steps in the steady call")
    if te.get("cold_s") and te.get("warm_s") is not None:
        pub = te.get("published_s") or (te["cold_s"] + te["warm_s"])
        print(
            f"\n  text_encoder_s as published   {pub:10.2f}s   <- cold + warm SUMMED (pipeline.py:191)"
        )
        print(
            f"    of which rebuild            {te['cold_s']:10.2f}s   ({100 * te['cold_s'] / pub:5.1f}%)"
        )
        print(
            f"    of which true warm          {te['warm_s']:10.2f}s   ({100 * te['warm_s'] / pub:5.1f}%)"
        )

    print(f"\n{'=' * 96}\nRAW [TIME] LINES\n{'=' * 96}")
    for label, sec in times:
        print(f"  {label:<52} {sec:10.2f}s")


if __name__ == "__main__":
    main()
