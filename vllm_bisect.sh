#!/bin/bash
# git bisect run script for vllm memory regression
# Must be run from /tmp/vllm_repo during git bisect

VLLM_REPO="/tmp/vllm_repo"
TT_XLA="/localdev/kmabee/tt-xla"
VLLM_SITE="$TT_XLA/venv/lib/python3.12/site-packages"
SO_CACHE="/tmp/vllm_so_cache"

SHA=$(git rev-parse --short HEAD)
echo "=== Testing $SHA ==="

# Clean install from source
pip uninstall vllm -y -q 2>/dev/null
rm -rf $VLLM_SITE/vllm $VLLM_SITE/vllm-*
git clean -fdx vllm/ 2>/dev/null

SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 VLLM_TARGET_DEVICE=empty \
    pip install . --no-build-isolation --no-deps -q 2>/dev/null

# Restore .so files
cp $SO_CACHE/_C.abi3.so $VLLM_SITE/vllm/ 2>/dev/null

# Fix tracing module if it's a directory missing exports
TRACING_INIT="$VLLM_SITE/vllm/tracing/__init__.py"
if [ -f "$TRACING_INIT" ]; then
    grep -q "^Tracer" "$TRACING_INIT" || echo 'Tracer = None' >> "$TRACING_INIT"
    grep -q "def init_tracer" "$TRACING_INIT" || echo '
def init_tracer(*a, **kw): pass
def init_worker_tracer(*a, **kw): pass
def instrument(span_name=None, **kw):
    def d(fn): return fn
    return d
def extract_trace_context(*a, **kw): return {}
class SpanAttributes: pass
class SpanKind: pass
' >> "$TRACING_INIT"
    rm -f "$VLLM_SITE/vllm/tracing/__pycache__/"* 2>/dev/null
fi

# Fix missing envs attributes
python3 -c "import vllm.envs" 2>/dev/null || true

# Determine which TT plugin to use based on whether vllm.attention.layer exists
if python3 -c "from vllm.attention.layer import Attention" 2>/dev/null; then
    # Old import path (pre v0.16 attention refactor) -> use pre-uplift plugin
    cd $TT_XLA && git checkout d07d560d2 -- integrations/vllm_plugin/ 2>/dev/null
else
    # New import path -> use post-uplift plugin
    cd $TT_XLA && git checkout b4d4337ed -- integrations/vllm_plugin/ 2>/dev/null
fi
pip install -e $TT_XLA/integrations/vllm_plugin/ --no-deps -q 2>/dev/null

cd $TT_XLA

# Run test (use 0.6B which exists in both plugin versions)
OUTPUT=$(/usr/bin/time -v pytest --durations=0 -svv --log-memory \
    "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-0.6B-False-False]" \
    2>&1)

if ! echo "$OUTPUT" | grep -q "PASSED"; then
    echo "  SKIP ($SHA): test failed"
    echo "$OUTPUT" | grep -E "Error|FAILED" | grep -v "Triton\|libcud\|importing" | head -3
    exit 125
fi

RSS_KB=$(echo "$OUTPUT" | grep "Maximum resident set size" | awk '{print $NF}')
RSS_MB=$((RSS_KB / 1024))
echo "  $SHA: RSS ${RSS_MB} MB"

if [ "$RSS_MB" -gt 6144 ]; then  # 6 GB threshold (v0.15 is ~4GB, v0.16 is ~8.5GB for 0.6B)
    echo "  BAD"
    exit 1
else
    echo "  GOOD"
    exit 0
fi
