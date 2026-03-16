# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity test: unflatten + unbind causes tt-mlir compilation hang at large sizes.

Root cause: The `apply_rotary_emb` function in diffusers' WanAttnProcessor
uses the pattern:

    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)

This combination of reshape → slice (what unflatten+unbind decomposes to in
StableHLO) causes the tt-mlir compiler to hang at large tensor sizes.

Findings (seq_len = second dim of [1, seq, 12, 128]):
  - unflatten alone        → PASS at all sizes
  - unbind alone           → PASS at all sizes
  - unflatten + unbind     → PASS at 32 tokens, HANGS at 3120 tokens
  - chunk(2, dim=-1)       → PASS at all sizes (equivalent workaround)

The hang is in the C++ tt-mlir compiler, not Python metadata propagation.
SIGALRM cannot interrupt it, so a hard timeout (SIGKILL) is needed.

Workaround: Replace unflatten+unbind with chunk:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)  # HANGS
    x1, x2 = hidden_states.chunk(2, dim=-1)                    # WORKS

Or equivalently for the full RoPE:
    # Instead of:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    # Use:
    x1, x2 = hidden_states.chunk(2, dim=-1)
    even = x1 * cos - x2 * sin
    odd  = x1 * sin + x2 * cos
    return torch.cat([even, odd], dim=-1)

Note: chunk splits [even_0, even_1, ..., odd_0, odd_1, ...] (contiguous halves)
while unflatten+unbind splits [even_0, odd_0, even_1, odd_1, ...] (interleaved).
These are NOT mathematically equivalent — the RoPE frequency assignment differs.
But for the purpose of demonstrating the compiler bug, both patterns work.

Filed as: [TODO: tt-mlir issue URL]

Usage:
    python -u tests/torch/models/wan/test_strided_scatter_rope.py
"""

import gc
import os
import signal
import subprocess
import sys
import time

import torch
import torch.nn as nn

# Each test runs in a subprocess to get a clean device state and enable
# hard kill (SIGKILL) on timeout since SIGALRM can't interrupt C++ code.

NUM_HEADS = 12
HEAD_DIM = 128
TIMEOUT_S = 120  # 2 min — more than enough; working cases finish in < 1s


def run_in_subprocess(test_name, model_code, seq_len, input_shape=None):
    """Run a single test in a fresh subprocess with hard timeout."""
    if input_shape is None:
        input_shape = f"1, {seq_len}, {NUM_HEADS}, {HEAD_DIM}"
    script = f"""
import time, torch, torch.nn as nn, torch_xla, torch_xla.runtime as xr
xr.set_device_type('TT')
device = torch_xla.device()

{model_code}

m = TestModel().eval().to(device)
comp = torch.compile(m, backend='tt')
x = torch.randn({input_shape}).to(device)
t0 = time.time()
with torch.no_grad():
    out = comp(x)
torch_xla.sync(wait=True)
elapsed = time.time() - t0
print(f'PASS {{elapsed:.1f}}s shape={{list(out.cpu().shape)}}')
"""
    env = os.environ.copy()
    venv_python = os.path.join(
        os.path.dirname(os.path.dirname(sys.executable)), "bin", "python"
    )
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    try:
        result = subprocess.run(
            [venv_python, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/../../../..",
        )
        # Find PASS line in stdout
        for line in result.stdout.splitlines():
            if line.startswith("PASS"):
                return line
        # Check stderr for errors
        for line in result.stderr.splitlines():
            if "Error" in line or "error" in line:
                return f"ERROR: {line.strip()[:60]}"
        if result.returncode != 0:
            return f"ERROR (exit code {result.returncode})"
        return "ERROR (no PASS output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT (>{TIMEOUT_S}s) — compiler hang confirmed"


# ── Model definitions ────────────────────────────────────────────────────

UNFLATTEN_UNBIND = """
class TestModel(nn.Module):
    def forward(self, x):
        r = x.unflatten(-1, (-1, 2))
        a, b = r.unbind(-1)
        return a + b
"""

UNFLATTEN_ONLY = """
class TestModel(nn.Module):
    def forward(self, x):
        return x.unflatten(-1, (-1, 2))
"""

UNBIND_ONLY = """
class TestModel(nn.Module):
    def forward(self, x):
        # Input pre-shaped as [B, S, H, D/2, 2]
        a, b = x.unbind(-1)
        return a + b
"""

CHUNK_WORKAROUND = """
class TestModel(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a + b
"""

SELECT_INDEX = """
class TestModel(nn.Module):
    def forward(self, x):
        r = x.unflatten(-1, (-1, 2))
        return r[..., 0] + r[..., 1]
"""

SLICE = """
class TestModel(nn.Module):
    def forward(self, x):
        r = x.unflatten(-1, (-1, 2))
        return r.select(-1, 0) + r.select(-1, 1)
"""

# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 75)
    print("tt-mlir compilation hang: unflatten + unbind at large tensor sizes")
    print("=" * 75)
    print()
    print(f"Tensor shape: [1, seq_len, {NUM_HEADS}, {HEAD_DIM}]")
    print(f"Timeout: {TIMEOUT_S}s per test (subprocess with hard kill)")
    print()
    print(f"{'Test':<50} {'32 tokens':>15} {'3120 tokens':>15}")
    print("-" * 80)

    # (label, model_code, custom_input_shape_or_None)
    tests = [
        ("unflatten only", UNFLATTEN_ONLY, None),
        ("unbind only (pre-shaped)", UNBIND_ONLY, True),
        ("unflatten + unbind (BUG)", UNFLATTEN_UNBIND, None),
        ("unflatten + select (indexing)", SELECT_INDEX, None),
        ("unflatten + .select()", SLICE, None),
        ("chunk (workaround)", CHUNK_WORKAROUND, None),
    ]

    for label, code, needs_5d_input in tests:
        if needs_5d_input:
            r32 = run_in_subprocess(
                label, code, 32, input_shape=f"1, 32, {NUM_HEADS}, {HEAD_DIM // 2}, 2"
            )
            r3120 = run_in_subprocess(
                label,
                code,
                3120,
                input_shape=f"1, 3120, {NUM_HEADS}, {HEAD_DIM // 2}, 2",
            )
        else:
            r32 = run_in_subprocess(label, code, 32)
            r3120 = run_in_subprocess(label, code, 3120)

        print(f"{label:<50} {r32:<15} {r3120:<15}")

    print("-" * 80)
    print()
    print("EXPECTED RESULT:")
    print("  'unflatten + unbind' PASSES at 32 tokens, TIMES OUT at 3120 tokens.")
    print("  All other patterns PASS at both sizes.")
    print()
    print("ROOT CAUSE:")
    print("  unflatten (reshape) + unbind (slice along last dim) produces a")
    print("  StableHLO pattern that causes tt-mlir to hang during compilation.")
    print("  The C++ compiler does not produce any output during the hang.")
