#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Falcon3-7B-Instruct evals against an ALREADY-RUNNING server, from tt-xla only
# -- no run.py, no tt-inference-server orchestration.
#
# Reproduces exactly what
#   ~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019
# ends up executing, by inlining the two lm_eval commands that
# tt-inference-server-v2/llm_module/eval_command.py:build_eval_command() emits
# for the Falcon3-7B-Instruct EvalConfig (evals/eval_config.py:4137):
#
#   ifeval                            num_fewshot=0, limit CI_NIGHTLY=0.25
#   gpqa_diamond_generative_n_shot    num_fewshot=5, limit CI_NIGHTLY=0.25
#
# with the shared EvalTask defaults (local-completions, tokenizer_backend=
# huggingface, batch_size=1, max_concurrent=32 clamped to the P150 spec's
# max_concurrency=32, seed=42, apply_chat_template, gen_kwargs stream=False +
# injected seed) and the forge P150 spec's eval_max_retries=1.
#
# Pairs with ./serve_falcon3_7b_forge.sh. Start that first and WAIT for the
# "[warmup] WARMUP COMPLETE" line -- driving evals at a cold server makes the
# one-time compile look like a hang (the #4521 false alarm).
#
# Usage:
#   ./run_falcon3_7b_evals.sh                            # both tasks, 0.25 each
#   ./run_falcon3_7b_evals.sh --limit 0.05               # both tasks, 5%
#   ./run_falcon3_7b_evals.sh --limit 20                 # both tasks, 20 docs
#   ./run_falcon3_7b_evals.sh --tasks ifeval --limit 1.0 # ifeval, full set
#   ./run_falcon3_7b_evals.sh --ifeval-limit 0.25 --gpqa-limit 0.1
#   ./run_falcon3_7b_evals.sh --concurrent 8             # narrow the burst
#
# Options (--flag value or --flag=value):
#   --tasks LIST     comma list from {ifeval,gpqa} (default: both, in order)
#   --limit N        set BOTH tasks' --limit. <1 = fraction of the task's docs,
#                    >=1 = absolute doc count. (default per-task: 0.25)
#   --ifeval-limit N --limit for ifeval only
#   --gpqa-limit N   --limit for gpqa only
#   --port P         server port (default 8019)
#   --server-url U   non-localhost server, e.g. http://10.0.0.5 (default 127.0.0.1)
#   --concurrent N   lm-eval num_concurrent (default 32 = the P150 max_concurrency)
#   --model REPO     HF repo (default tiiuae/Falcon3-7B-Instruct)
#   --output DIR     results dir (default $ROOT/eval_results_falcon3_7b)
#   -h, --help       show this help
#
# Env:
#   LM_EVAL_BIN     lm_eval to use. Auto-detected: $ROOT/.venv_lm_eval first
#                   (fully standalone -- see the build command the script prints
#                   if nothing is found), else tt-inference-server's prebuilt
#                   EVALS_COMMON venv. Either way only a venv is used; no run.py
#                   or workflow code runs.
#   EVAL_GEN_SEED   per-request seed injected into gen_kwargs (default 42, as
#                   tt-inference-server does). Set to "" to omit: tt-media-server
#                   force-drops request seeds (#4338 / tt-xla#4539) and stock
#                   vLLM does not, so "" is the closer match to a tt-media-server
#                   -hosted run. Greedy either way (lm-eval sends temperature=0).
#   OPENAI_API_KEY  bearer token (default your-secret-key; ignored by a
#                   vllm serve started without --api-key).
#   HF_TOKEN        needed for gpqa (Idavidrein/gpqa is gated) unless cached.
set -eo pipefail

# A tt-xla venv's PYTHONPATH makes tt-xla's own tests/utils.py shadow other
# packages' utils modules. Not needed by lm_eval; drop it.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The exact lm-eval the tt-inference-server EVALS_COMMON venv was built from
# (tstescoTT/lm-evaluation-harness @ evals-common, resolved to a sha).
LM_EVAL_PIN="f3d35ab3c8d74b90548b3198b2337c32431a8e06"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit "${1:-0}"; }

TASKS="ifeval,gpqa"
LIMIT_BOTH=""
# EvalLimitMode.CI_NIGHTLY for both tasks in this model's EvalConfig.
IFEVAL_LIMIT="0.25"; IFEVAL_LIMIT_SET=0
GPQA_LIMIT="0.25";   GPQA_LIMIT_SET=0
PORT="8019"
SERVER_URL=""
CONCURRENT="32"
MODEL_REPO="tiiuae/Falcon3-7B-Instruct"
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --tasks|--limit|--ifeval-limit|--gpqa-limit|--port|--server-url|--concurrent|--model|--output)
      key="$1"; val="${2:-}"; shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --tasks=*|--limit=*|--ifeval-limit=*|--gpqa-limit=*|--port=*|--server-url=*|--concurrent=*|--model=*|--output=*)
      key="${1%%=*}"; val="${1#*=}"; shift ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
  case "$key" in
    --tasks) TASKS="$val" ;;
    --limit) LIMIT_BOTH="$val" ;;
    --ifeval-limit) IFEVAL_LIMIT="$val"; IFEVAL_LIMIT_SET=1 ;;
    --gpqa-limit) GPQA_LIMIT="$val"; GPQA_LIMIT_SET=1 ;;
    --port) PORT="$val" ;;
    --server-url) SERVER_URL="$val" ;;
    --concurrent) CONCURRENT="$val" ;;
    --model) MODEL_REPO="$val" ;;
    --output) OUTPUT_DIR="$val" ;;
  esac
done

# --limit wins over the per-task defaults, but an explicit --ifeval-limit /
# --gpqa-limit given alongside it still wins for that task.
if [ -n "$LIMIT_BOTH" ]; then
  [ "$IFEVAL_LIMIT_SET" = "1" ] || IFEVAL_LIMIT="$LIMIT_BOTH"
  [ "$GPQA_LIMIT_SET" = "1" ] || GPQA_LIMIT="$LIMIT_BOTH"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/eval_results_falcon3_7b}"
HOST_BASE="${SERVER_URL:-http://127.0.0.1}"
BASE_URL="${HOST_BASE}:${PORT}/v1/completions"

# Prefer a tt-xla-local lm_eval (fully standalone), then fall back to
# tt-inference-server's prebuilt EVALS_COMMON venv if one happens to be there.
if [ -z "$LM_EVAL_BIN" ]; then
  for _cand in \
    "$ROOT/.venv_lm_eval/bin/lm_eval" \
    "$HOME/tt-inference-server/.workflow_venvs/.venv_evals_common/bin/lm_eval"
  do
    [ -x "$_cand" ] && { LM_EVAL_BIN="$_cand"; break; }
  done
fi
if [ ! -x "$LM_EVAL_BIN" ]; then
  cat >&2 <<EOF
ERROR: no lm_eval found. Set LM_EVAL_BIN, or build a standalone one (~2 min, ~1.5G):

  uv venv --python 3.10 $ROOT/.venv_lm_eval
  UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \\
  uv pip install --python $ROOT/.venv_lm_eval \\
    "lm-eval[api,ifeval,math,sentencepiece,r1_evals,ruler,longbench,hf] @ git+https://github.com/tstescoTT/lm-evaluation-harness.git@$LM_EVAL_PIN" \\
    datasets==3.1.0

Use the TT fork at that pinned commit, NOT PyPI lm-eval: the fork patches
lm_eval/tasks/ifeval/{instructions,utils}.py and lm_eval/filters/extraction.py
(the 'flexible-extract' filter gpqa is scored on), so upstream would silently
score both of these tasks differently. The branch (evals-common) moves and is
currently 108 commits behind upstream -- pin the sha.
EOF
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-your-secret-key}}"
EVAL_GEN_SEED="${EVAL_GEN_SEED-42}"

# Preflight: the server must already be up. Driving a cold/absent server is the
# single most common way to mistake a compile (or a connection refused storm)
# for the hang under investigation.
if ! curl -sf "${HOST_BASE}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: no server answering at ${HOST_BASE}:${PORT}/v1/models"
  echo "  Start one first:  cd $ROOT && source venv/activate && ./serve_falcon3_7b_forge.sh"
  exit 1
fi
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — gpqa (Idavidrein/gpqa) is gated; a cached dataset may suffice."

# EvalTask defaults + the forge P150 device spec, as resolved by
# build_eval_command() for this model.
MODEL_ARGS="model=${MODEL_REPO},base_url=${BASE_URL},tokenizer_backend=huggingface,num_concurrent=${CONCURRENT},max_retries=1"
GEN_KWARGS="stream=False"
[ -n "$EVAL_GEN_SEED" ] && GEN_KWARGS="${GEN_KWARGS},seed=${EVAL_GEN_SEED}"

mkdir -p "$OUTPUT_DIR"

run_task() {
  local task_name="$1" num_fewshot="$2" limit="$3"
  echo
  echo "=============================================================="
  echo "[evals] task=$task_name  num_fewshot=$num_fewshot  limit=$limit"
  echo "[evals] concurrent=$CONCURRENT  url=$BASE_URL"
  echo "=============================================================="
  set -x
  "$LM_EVAL_BIN" \
    --tasks "$task_name" \
    --model local-completions \
    --model_args "$MODEL_ARGS" \
    --gen_kwargs "$GEN_KWARGS" \
    --output_path "$OUTPUT_DIR" \
    --seed 42 \
    --num_fewshot "$num_fewshot" \
    --batch_size 1 \
    --log_samples \
    --show_config \
    --apply_chat_template \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --limit "$limit"
  { set +x; } 2>/dev/null
}

rc=0
IFS=',' read -ra _tasks <<< "$TASKS"
for t in "${_tasks[@]}"; do
  case "$(echo "$t" | tr -d ' ')" in
    ifeval) run_task ifeval 0 "$IFEVAL_LIMIT" || rc=$? ;;
    gpqa|gpqa_diamond_generative_n_shot)
      run_task gpqa_diamond_generative_n_shot 5 "$GPQA_LIMIT" || rc=$? ;;
    "") : ;;
    *) echo "ERROR: unknown task '$t' (expected ifeval and/or gpqa)"; exit 1 ;;
  esac
done

# --- summary ------------------------------------------------------------------
# Same metrics tt-inference-server scores on, against the same references
# (evals/eval_config.py:4137, gpu refs from tt-inference-server#4090).
echo
echo "=============================================================="
echo " Summary  (results under $OUTPUT_DIR)"
echo "=============================================================="
OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PY'
import glob, json, os

REFS = {
    "ifeval": [
        ("prompt_level_strict_acc,none", 72.64, "scored key"),
        ("inst_level_strict_acc,none", None, ""),
    ],
    "gpqa_diamond_generative_n_shot": [
        ("exact_match,flexible-extract", 43.43, "scored key"),
    ],
}

files = sorted(
    glob.glob(os.path.join(os.environ["OUTPUT_DIR"], "**", "results_*.json"), recursive=True),
    key=os.path.getmtime,
)
if not files:
    print("  no results_*.json found — did the tasks run?")
    raise SystemExit(0)

merged, nsamples = {}, {}
for f in files:
    with open(f) as fh:
        blob = json.load(fh)
    merged.update(blob.get("results", {}) or {})
    for task, counts in (blob.get("n-samples", {}) or {}).items():
        nsamples[task] = counts.get("effective")

for task, keys in REFS.items():
    res = merged.get(task)
    if not res:
        continue
    n = nsamples.get(task)
    print(f"\n  {task}" + (f"  (n={n})" if n else ""))
    for key, ref, note in keys:
        val = res.get(key)
        if val is None:
            print(f"    {key:<34} —")
            continue
        pct = val * 100.0
        line = f"    {key:<34} {pct:6.2f}%"
        if ref:
            # tt-inference-server's full-set check: score/reference >= 1-0.05.
            ratio = pct / ref
            verdict = "PASS" if ratio >= 0.95 else "FAIL"
            line += f"   gpu_ref={ref:.2f}%  ratio={ratio:.3f}  {verdict} ({note})"
        print(line)
print("\n  NOTE: gpu_ref is the FULL-dataset reference. A downsampled run"
      "\n  (--limit < 1.0) is noisier; treat the ratio as indicative only.")
PY

exit "$rc"
