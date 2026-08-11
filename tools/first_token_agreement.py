# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""First-token agreement between a TT server and an HF CPU golden.

A deterministic accuracy metric to use in place of a full eval run. The eval
cannot resolve effects below ~3 points (two identical Falcon3 runs differed by
2.77), which makes knob and kernel comparisons unfalsifiable. This measures the
argmax of a single prefill instead: no decode, no batching, no scoring harness,
so it is exactly reproducible and sensitive to small numeric changes.

Three stages, each cached to JSON so they can be run independently:

  --collect-tt   sequential max_tokens=1 requests against a running server
  --golden       HF transformers forward pass on CPU (fp32 by default)
  --compare      top-1 agreement, plus the golden-side margin on disagreements

A disagreement with a wide golden margin is a real numeric defect; one with a
razor-thin margin is benign tie-breaking. Reporting both separates them.

Prompts are the exact strings lm-eval sent, read out of a samples_*.jsonl so
that TT, golden and the recorded GPU run all see byte-identical input.
"""

import argparse
import ast
import json
import os
import urllib.error
import urllib.request


def load_prompts(samples_path, limit=None):
    prompts = []
    with open(samples_path) as f:
        for line in f:
            j = json.loads(line)
            args = j["arguments"]
            if isinstance(args, str):
                args = ast.literal_eval(args)
            prompts.append((str(j["doc_id"]), args["gen_args_0"]["arg_0"]))
    prompts.sort(key=lambda t: int(t[0]))
    return prompts[:limit] if limit else prompts


def collect_tt(port, model, prompts, out_path, top_k):
    """One request per prompt, strictly sequential, one output token."""
    results = {}
    want_logprobs = top_k

    def ask(prompt, with_logprobs):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
        if with_logprobs:
            payload["logprobs"] = top_k
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer your-secret-key",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]

    for i, (doc_id, prompt) in enumerate(prompts):
        try:
            choice = ask(prompt, want_logprobs)
        except urllib.error.HTTPError:
            # Server rejects the logprobs kwarg; the decoded token still works.
            want_logprobs = 0
            print("  note: server rejected logprobs, comparing on text only")
            choice = ask(prompt, False)
        lp = choice.get("logprobs") or {}
        results[doc_id] = {
            "text": choice["text"],
            # top_logprobs is a list with one entry per generated token
            "top": (lp.get("top_logprobs") or [None])[0],
        }
        if (i + 1) % 25 == 0:
            print(f"  tt {i+1}/{len(prompts)}", flush=True)
    json.dump(results, open(out_path, "w"))
    print(f"wrote {out_path} ({len(results)} prompts)")


def golden(model_id, prompts, out_path, dtype, top_k):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map="cpu"
    ).eval()

    results = {}
    for i, (doc_id, prompt) in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        logits = model(ids).logits[0, -1].float()
        vals, idx = logits.topk(top_k)
        results[doc_id] = {
            "top_ids": idx.tolist(),
            "top_logits": vals.tolist(),
            "top_text": [tok.decode([t]) for t in idx.tolist()],
            "n_prompt_tokens": int(ids.shape[1]),
        }
        if (i + 1) % 10 == 0:
            print(f"  golden {i+1}/{len(prompts)}", flush=True)
    json.dump(results, open(out_path, "w"))
    print(f"wrote {out_path} ({len(results)} prompts)")


def compare(tt_path, golden_path):
    tt = json.load(open(tt_path))
    gd = json.load(open(golden_path))
    ids = sorted(set(tt) & set(gd), key=int)

    agree, disagree, unranked, skipped = 0, [], 0, 0
    for d in ids:
        g = gd[d]
        # A golden top token that is a partial UTF-8 sequence cannot be compared
        # as text -- per-token decode yields U+FFFD and the server returns "".
        if "�" in g["top_text"][0]:
            skipped += 1
            continue
        tt_text = tt[d]["text"]
        # Match on decoded text: the server's tokenizer ids are not necessarily
        # comparable, but the emitted string is.
        if tt_text == g["top_text"][0]:
            agree += 1
            continue
        rank = next(
            (r for r, t in enumerate(g["top_text"]) if t == tt_text), None
        )
        if rank is None:
            unranked += 1
        margin = g["top_logits"][0] - (
            g["top_logits"][rank] if rank is not None else g["top_logits"][-1]
        )
        disagree.append((d, tt_text, g["top_text"][0], rank, margin, g["n_prompt_tokens"]))

    n = len(ids) - skipped
    print(f"prompts compared        : {n}  ({skipped} skipped: multi-byte golden token)")
    print(f"top-1 agreement         : {agree}/{n} ({100.0*agree/n:.1f}%)")
    print(f"disagreements           : {len(disagree)}")
    if not disagree:
        return
    margins = sorted(m for *_, m, _ in disagree)
    thin = sum(1 for m in margins if m < 0.05)
    print(f"  golden margin < 0.05  : {thin}  (benign tie-breaking)")
    print(f"  golden margin >= 0.05 : {len(margins)-thin}  (real numeric defect)")
    print(f"  margin  min/median/max: {margins[0]:.4f} / "
          f"{margins[len(margins)//2]:.4f} / {margins[-1]:.4f}")
    print(f"  TT token outside golden top-k: {unranked}")
    print()
    print(f"  {'doc':>5} {'rank':>4} {'margin':>8} {'ptoks':>6}  tt -> golden")
    for d, t, g0, rank, m, np_ in sorted(disagree, key=lambda r: -r[4])[:15]:
        print(f"  {d:>5} {str(rank):>4} {m:>8.4f} {np_:>6}  {t!r} -> {g0!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="a samples_*.jsonl for the prompts")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--port", type=int, default=8019)
    ap.add_argument("--model", default="tiiuae/Falcon3-7B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--collect-tt", action="store_true")
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.samples, args.limit)
    tt_path = os.path.join(args.outdir, "first_token_tt.json")
    gd_path = os.path.join(args.outdir, "first_token_golden.json")

    if args.collect_tt:
        collect_tt(args.port, args.model, prompts, tt_path, args.top_k)
    if args.golden:
        golden(args.model, prompts, gd_path, args.dtype, args.top_k)
    if args.compare:
        compare(tt_path, gd_path)


if __name__ == "__main__":
    main()
