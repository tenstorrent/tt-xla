"""
Standalone repro for pocket_tts floating-point exception (SyncTensorsGraph.24).

In FlowLMModel.forward(), torch.dynamo fuses these two ops into one graph:

    sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
    input_   = self.input_linear(sequence)          # Linear(32, 1024, bias=False)

With shapes:
    sequence:            [1, 0, 32]   float32   (zero-size second dimension)
    bos_emb:             [32]         float32
    input_linear.weight: [1024, 32]   float32

This gets lowered to ttir.where + ttir.dot on tensor<1x0x32xf32> which triggers
a fatal floating-point exception during device execution.

Test 1: Isolates just the where op.
Test 2: Replicates the full fused graph (where + input_linear).
"""

import torch
import torch_xla.core.xla_model as xm


class WhereIsnanOnly(torch.nn.Module):
    """Isolates the where op to check if it alone causes the FPE."""

    def __init__(self):
        super().__init__()
        self.bos_emb = torch.nn.Parameter(torch.randn(32, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(sequence), self.bos_emb, sequence)


class WhereIsnanPlusLinear(torch.nn.Module):
    """Replicates the exact fused graph from pocket_tts SyncTensorsGraph.24."""

    def __init__(self):
        super().__init__()
        self.bos_emb = torch.nn.Parameter(torch.randn(32, dtype=torch.float32))
        self.input_linear = torch.nn.Linear(32, 1024, bias=False, dtype=torch.float32)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        out = self.input_linear(sequence)
        return sequence, out


def run_test(name, model, inputs):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    print("--- CPU run ---")
    cpu_outs = model(*inputs)
    if isinstance(cpu_outs, tuple):
        for i, o in enumerate(cpu_outs):
            print(f"  CPU output[{i}] shape: {o.shape}, dtype: {o.dtype}")
    else:
        print(f"  CPU output shape: {cpu_outs.shape}, dtype: {cpu_outs.dtype}")

    print("--- TT device run ---")
    device = xm.xla_device()
    model_dev = model.to(device)
    inputs_dev = [inp.to(device) for inp in inputs]

    tt_outs = model_dev(*inputs_dev)
    if isinstance(tt_outs, tuple):
        for i, o in enumerate(tt_outs):
            o_cpu = o.to("cpu")
            print(f"  TT output[{i}] shape: {o_cpu.shape}, dtype: {o_cpu.dtype}")
    else:
        o_cpu = tt_outs.to("cpu")
        print(f"  TT output shape: {o_cpu.shape}, dtype: {o_cpu.dtype}")

    print(f"PASS - {name}: no floating-point exception")


def test_sanities():
    sequence = torch.randn(1, 0, 32, dtype=torch.float32)

    print(f"Input sequence shape: {sequence.shape}, dtype: {sequence.dtype}")

    run_test(
        "where(isnan) only — zero-dim [1,0,32]",
        WhereIsnanOnly(),
        [sequence],
    )



    # to check if the full fused graph is the issue, but it also triggers the FPE and crashes the test process, so commenting out for now
    # run_test(
    #     "where(isnan) + Linear(32,1024) — full fused graph [1,0,32]",
    #     WhereIsnanPlusLinear(),
    #     [sequence],
    # )

    # print("\n" + "="*60)
    # print("ALL TESTS PASSED")
    # print("="*60)

