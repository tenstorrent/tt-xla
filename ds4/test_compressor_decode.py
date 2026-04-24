import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

# Install stubs before importing model
import kernel_stubs
kernel_stubs.install()

from model import ModelArgs, Compressor, precompute_freqs_cis


def test_compressor_decode_after_prefill_tt():
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    batch_size = 1
    prefill_len = 128
    compress_ratio = 4  # CSA

    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=512,
        dim=256,
        head_dim=64,
        rope_head_dim=16,
    )

    comp = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    comp.eval()
    comp = comp.to(device)
    # comp = torch.compile(comp, backend="tt")

    # Setup buffers on device
    coff = 2 if comp.overlap else 1
    comp.kv_state = torch.zeros(
        batch_size, coff * compress_ratio, coff * args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    comp.score_state = torch.zeros(
        batch_size, coff * compress_ratio, coff * args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    comp.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    comp.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    # === Prefill 128 tokens ===
    x_prefill = torch.randn(batch_size, prefill_len, args.dim, dtype=torch.bfloat16,
device=device)
    with torch.no_grad():
        kv_prefill = comp.forward(x_prefill, start_pos=0)
    xm.mark_step()

    print(f"Prefill output shape: {kv_prefill.shape}")

    # === Decode tokens starting at position 128 ===
    start_pos = 128
    for i in range(4):
        x_decode = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            kv_out = comp.forward(x_decode, start_pos=start_pos + i)
        xm.mark_step()

        pos = start_pos + i
        if kv_out is not None:
            print(f"pos={pos}: Compression triggered, output shape: {kv_out.shape}")
        else:
            print(f"pos={pos}: Accumulating (no output)")


if __name__ == "__main__":
    test_compressor_decode_after_prefill_tt()