#!/usr/bin/env bash
# Step 1: save Forge / TTNN / CPU stacked outputs and print PCC matrix.
#
# Answers: does live Forge match exported TTNN? (primary)
#          does each match CPU? (sanity columns)
#
# Usage:
#   cd /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
#   bash janus_layer0_forge_vs_ttnn_compare/run_step1_forge_vs_ttnn.sh
#
# Env (optional):
#   TTXLA_ROOT, TTMETAL_ROOT, JANUS_LAYER0_FIXTURE_DIR, JANUS_TTNN_GRAPH_DIR
set -euo pipefail

TTXLA="${TTXLA_ROOT:-/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla}"
TTMETAL="${TTMETAL_ROOT:-/proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal}"
COMPARE="${TTXLA}/janus_layer0_forge_vs_ttnn_compare"
ART="${JANUS_COMPARE_ARTIFACTS_DIR:-${COMPARE}/artifacts}"
FIXTURE="${JANUS_LAYER0_FIXTURE_DIR:-${TTXLA}/janus_logs/layer0_tensors/Pro_1B}"
GRAPH="${JANUS_TTNN_GRAPH_DIR:-${TTMETAL}/janus_layer0_ln_attn_no_dep_codegen/graph_0}"
LOG="${TTXLA}/janus_layer0_step1_forge_vs_ttnn.log"

TT_PY="${TTMETAL}/python_env/bin/python"
XLA_PY="${TTXLA}/venv/bin/python"

exec > >(tee "${LOG}") 2>&1

echo "========== Step 1: Forge vs TTNN vs CPU artifacts $(date -Iseconds) =========="
echo "TTXLA=${TTXLA}"
echo "TTMETAL=${TTMETAL}"
echo "FIXTURE=${FIXTURE}"
echo "GRAPH=${GRAPH}"
echo "ARTIFACTS=${ART}"

if [[ ! -d "${FIXTURE}" ]]; then
  echo "ERROR: fixtures missing at ${FIXTURE}"
  echo "Run: pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_pro_1b"
  exit 1
fi

if [[ ! -f "${GRAPH}/main.py" ]]; then
  echo "ERROR: TTNN graph missing at ${GRAPH}/main.py"
  echo "Run codegen and cp -a to tt-metal, or set JANUS_TTNN_GRAPH_DIR"
  exit 1
fi

mkdir -p "${ART}"
rm -f "${ART}"/forge_stacked_*.pt "${ART}"/ttnn_stacked_*.pt "${ART}"/cpu_stacked_*.pt

export JANUS_LAYER0_FIXTURE_DIR="${FIXTURE}"
export JANUS_COMPARE_ARTIFACTS_DIR="${ART}"
export JANUS_TTNN_GRAPH_DIR="${GRAPH}"

echo ""
echo "========== (1/3) Capture CPU eager stacked =========="
cd "${TTXLA}"
# shellcheck disable=SC1091
[[ -f venv/bin/activate ]] && source venv/bin/activate
python "${COMPARE}/capture_cpu.py"

echo ""
echo "========== (2/3) Capture Forge on TT device (tt-xla) =========="
python "${COMPARE}/capture_forge.py"

echo ""
echo "========== (3/3) Capture TTNN graph_0 (tt-metal python_env) =========="
export JANUS_TTXLA_ROOT="${TTXLA}"
cd "${TTMETAL}"
"${TT_PY}" "${COMPARE}/capture_ttnn.py" --graph-dir "${GRAPH}"

echo ""
echo "========== PCC matrix (offline) =========="
cd "${TTXLA}"
python "${COMPARE}/compare.py"

echo ""
echo "========== Done. Log: ${LOG} =========="
echo "Artifacts:"
ls -la "${ART}"/*stacked*.pt 2>/dev/null || true
echo ""
echo "Interpret self_attn PCC:"
echo "  Forge vs TTNN  ~1.0  => export matches live Forge; debug Forge vs CPU on xla"
echo "  Forge vs TTNN  low   => live Forge != export; fix input/trace parity first"
echo "  CPU vs Forge   ~0.77 => Experiment A drop (live Forge)"
echo "  CPU vs TTNN    ~0.99 => Experiment C (tt-metal main.py)"
