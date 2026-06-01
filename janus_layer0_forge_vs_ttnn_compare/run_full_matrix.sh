#!/usr/bin/env bash
# Re-run all layer-0 PCC experiments and write one log.
# Experiment A = Forge vs CPU (tt-xla, ~0.77 repro)
# Experiment C = TTNN graph_0 vs cpu_reference on tt-metal (~0.99 after 5.5.1 align)
# Offline D = Forge artifact vs TTNN artifact (~1.0)
set -euo pipefail

TTXLA="${TTXLA_ROOT:-/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla}"
TTMETAL="${TTMETAL_ROOT:-/proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal}"
FIXTURE="${JANUS_LAYER0_FIXTURE_DIR:-${TTXLA}/janus_logs/layer0_tensors/Pro_1B}"
LOG="${TTXLA}/janus_layer0_pcc_full_matrix.log"
CODEGEN=janus_layer0_ln_attn_no_dep_codegen
COMPARE=janus_layer0_forge_vs_ttnn_compare
ART="${TTXLA}/${COMPARE}/artifacts"

exec > >(tee "${LOG}") 2>&1

echo "========== layer-0 PCC full matrix $(date -Iseconds) =========="
echo "TTXLA=${TTXLA}"
echo "TTMETAL=${TTMETAL}"
echo "FIXTURE=${FIXTURE}"

# --- tt-metal: transformers 5.5.1 inside venv ---
TT_PY="${TTMETAL}/python_env/bin/python"
echo ""
echo "========== tt-metal python / transformers =========="
"${TT_PY}" -m ensurepip --upgrade 2>/dev/null || true
"${TT_PY}" -m pip install -q --force-reinstall 'transformers==5.5.1'
"${TT_PY}" -c "import transformers; print('tt-metal transformers', transformers.__version__, transformers.__file__)"

# --- Experiment A: Forge vs CPU (tt-xla) ---
echo ""
echo "========== Experiment A: Forge vs CPU (live op test) =========="
cd "${TTXLA}"
if [[ -f venv/bin/activate ]]; then source venv/bin/activate; fi
export JANUS_LAYER0_FIXTURE_DIR="${FIXTURE}"
python "${COMPARE}/run_cpu_vs_forge_sanity.py" || echo "WARN: Experiment A failed (device?)"

# --- Codegen + sync ---
echo ""
echo "========== Codegen + sync to tt-metal =========="
export JANUS_LAYER0_FIXTURE_DIR="${FIXTURE}"
rm -rf "${TTXLA}/${CODEGEN}"
python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py
rm -rf "${TTMETAL}/${CODEGEN}"
cp -a "${TTXLA}/${CODEGEN}" "${TTMETAL}/${CODEGEN}"

# --- Artifacts: Forge, TTNN, CPU (eager) ---
echo ""
echo "========== Capture artifacts (Forge / TTNN / CPU) =========="
mkdir -p "${ART}"
rm -f "${ART}"/forge_stacked_*.pt "${ART}"/ttnn_stacked_*.pt "${ART}"/cpu_stacked_*.pt
python "${COMPARE}/capture_forge.py" || echo "WARN: capture_forge failed"
cd "${TTMETAL}" && "${TT_PY}" "${TTXLA}/${COMPARE}/capture_ttnn.py" || echo "WARN: capture_ttnn failed"
cd "${TTXLA}"
python "${COMPARE}/capture_cpu.py" || echo "WARN: capture_cpu failed"

# --- Offline matrix: Forge vs TTNN vs CPU ---
echo ""
echo "========== Offline PCC matrix (saved tensors) =========="
python "${COMPARE}/compare.py"

# --- Experiment C: tt-metal main.py vs cpu_reference ---
echo ""
echo "========== Experiment C: TTNN main.py vs cpu_reference =========="
cd "${TTMETAL}"
export JANUS_LAYER0_FIXTURE_DIR="${FIXTURE}"
rm -f "${CODEGEN}/cpu_reference/golden/stacked_stages_pro_1b.pt"
"${TT_PY}" "${CODEGEN}/cpu_reference/generate_golden.py"
"${TT_PY}" "${CODEGEN}/graph_0/main.py"

echo ""
echo "========== Done. Full log: ${LOG} =========="
echo "Read ${COMPARE}/EXPERIMENTS.md for how to interpret A vs C."
