# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import gc

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

# Enable Shardy partitioner for advanced sharding analysis
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
import psutil
import time
import threading
from collections import defaultdict
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from transformers import AutoTokenizer
from generation import LLaMA
from jax.sharding import Mesh, PartitionSpec as P
import os
from pathlib import Path

ROOT = Path(__file__).parent


class MemoryMonitor:
    """Monitor CPU and JAX device memory usage"""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = defaultdict(list)
        self.peak_memory = {}
        self.start_time = None

    def start_monitoring(self, interval=0.1):
        """Start monitoring memory usage in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Memory monitoring started...")

    def stop_monitoring(self):
        """Stop monitoring and return peak usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        # Calculate peaks
        for device, history in self.memory_history.items():
            if history:
                self.peak_memory[device] = max(history, key=lambda x: x[1])

        print("\n📊 MEMORY USAGE SUMMARY:")
        print("=" * 50)

        total_peak_gb = 0
        for device, (timestamp, peak_mb) in self.peak_memory.items():
            peak_gb = peak_mb / 1024
            total_peak_gb += peak_gb
            elapsed = timestamp - self.start_time
            print(f"  {device}: {peak_gb:.2f} GB (peak at {elapsed:.1f}s)")

        print(f"  TOTAL PEAK: {total_peak_gb:.2f} GB")
        print("=" * 50)

        return self.peak_memory

    def _monitor_loop(self, interval):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()

                # Monitor CPU memory
                process = psutil.Process()
                cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_history["CPU"].append((timestamp, cpu_memory_mb))

                # Monitor JAX device memory (if available)
                try:
                    for i, device in enumerate(jax.devices()):
                        # Get device memory stats if available
                        device_name = f"Device_{i}_{device.device_kind}"

                        # For CPU devices, we can track allocated arrays
                        device_memory_mb = self._estimate_device_memory(device)
                        if device_memory_mb > 0:
                            self.memory_history[device_name].append(
                                (timestamp, device_memory_mb)
                            )

                except Exception:
                    pass  # Device memory monitoring not available

                time.sleep(interval)
            except Exception:
                break

    def _estimate_device_memory(self, device):
        """Estimate memory usage on a JAX device"""
        try:
            # This is an approximation - JAX doesn't expose detailed memory stats for CPU
            # We'll track based on live arrays
            live_arrays = [
                x
                for x in gc.get_objects()
                if hasattr(x, "device") and x.device == device
            ]
            total_bytes = sum(getattr(x, "nbytes", 0) for x in live_arrays)
            return total_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0

    def print_current_usage(self, label=""):
        """Print current memory usage"""
        process = psutil.Process()
        cpu_memory_gb = process.memory_info().rss / (1024**3)

        print(f"💾 {label} Memory: {cpu_memory_gb:.2f} GB CPU")

        # Try to get JAX memory info
        try:
            for i, device in enumerate(jax.devices()):
                device_memory_mb = self._estimate_device_memory(device)
                if device_memory_mb > 0:
                    print(f"   Device {i}: {device_memory_mb/1024:.2f} GB")
        except:
            pass


def jax_load(
    model_id: str,
    ckpt_dir: str,
    tokenizer_path: str,
    mesh,
    max_seq_length: int = 2048,
    n_layers: int = 32,
) -> LLaMA:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    full_weights, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers,
        verbose=True,
        mesh=mesh,
    )

    full_weights = freeze(full_weights)
    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=full_weights, model=model, tokenizer=tokenizer, mesh=mesh)

    del full_weights
    gc.collect()
    return llama


def create_params_sharding_tree(mesh, n_layers):
    """Create sharding tree that matches the parameter structure"""

    # Helper function to create sharding specs
    def shard_spec(shard_axis):
        if shard_axis == 0:
            return jax.sharding.NamedSharding(mesh, P("mp", None))
        elif shard_axis == 1:
            return jax.sharding.NamedSharding(mesh, P(None, "mp"))
        else:
            return jax.sharding.NamedSharding(mesh, P())

    # Build the sharding tree to match jax_params structure
    params_sharding = {
        "transformer": {
            "wte": {
                "embedding": shard_spec(0),  # Vocab parallel
            },
            "h": {},
            "ln_f": {
                "kernel": shard_spec(-1),  # Replicated
            },
        },
        "lm_head": {
            "kernel": shard_spec(1),  # Column parallel
        },
    }

    # Add transformer layers
    for layer_idx in range(n_layers):
        layer_key = str(layer_idx)
        params_sharding["transformer"]["h"][layer_key] = {
            "attention": {
                "wq": {"kernel": shard_spec(1)},  # Column parallel
                "wk": {"kernel": shard_spec(1)},  # Column parallel
                "wv": {"kernel": shard_spec(1)},  # Column parallel
                "wo": {"kernel": shard_spec(0)},  # Row parallel
            },
            "feed_forward": {
                "w1": {"kernel": shard_spec(1)},  # Column parallel
                "w2": {"kernel": shard_spec(0)},  # Row parallel
                "w3": {"kernel": shard_spec(1)},  # Column parallel
            },
            "attention_norm": {"kernel": shard_spec(-1)},  # Replicated
            "ffn_norm": {"kernel": shard_spec(-1)},  # Replicated
        }

    # Convert to FrozenDict to match llama.params structure
    return freeze(params_sharding)


def main(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    ckpt_dir=str(ROOT / "llama3.1-8B/8B/original"),
    tokenizer_path=str(ROOT / "llama3.1-8B/original/original/tokenizer.model"),
    prompt=("How much is 10 squared?"),
    max_gen_len: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 16,
    max_seq_length: int = 16,
    print_hlo: bool = False,
    monitor_memory: bool = True,
):
    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    if monitor_memory:
        memory_monitor.start_monitoring()
        memory_monitor.print_current_usage("Initial")

    # Define mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))
    print("✅ Mesh initialized:", mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(
        model_id,
        ckpt_dir,
        tokenizer_path,
        mesh,
        max_seq_length=max_seq_length,
        n_layers=n_layers,
    )
    if monitor_memory:
        memory_monitor.print_current_usage("After Model Load")

    print("✍️ Generating...")

    # Print HLO if requested
    if print_hlo:
        print("🔍 Extracting StableHLO for model forward pass...")
        print(f"📊 JAX devices available: {jax.devices()}")
        print(f"📊 JAX device count: {jax.device_count()}")

        # Create test inputs
        test_tokens = jnp.array(
            [[llama.tokenizer.bos_token_id, 1, 2, 3, 4]], dtype=jnp.int32
        )
        test_attention_mask = jnp.ones_like(test_tokens)

        # CRITICAL: HLO extraction must run within mesh context for multi-device
        with mesh:

            def model_forward_jit(llama_params, input_ids, attention_mask):
                position_ids = jnp.broadcast_to(
                    jnp.arange(input_ids.shape[-1])[None, :], input_ids.shape
                )

                # Force multi-device compilation by constraining inputs to be sharded
                input_ids = jax.lax.with_sharding_constraint(
                    input_ids, jax.sharding.NamedSharding(mesh, P(None, None))
                )
                attention_mask = jax.lax.with_sharding_constraint(
                    attention_mask, jax.sharding.NamedSharding(mesh, P(None, None))
                )

                return llama.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    params=llama_params,
                    return_dict=True,
                )

            # Create sharding tree for parameters
            params_sharding = create_params_sharding_tree(mesh, n_layers)

            # Lower the function to get HLO with explicit sharding info
            input_shardings = [
                params_sharding,  # llama_params - complex nested structure
                jax.sharding.NamedSharding(mesh, P(None, None)),  # input_ids
                jax.sharding.NamedSharding(mesh, P(None, None)),  # attention_mask
            ]

            lowered = jax.jit(
                model_forward_jit,
                in_shardings=input_shardings,
            ).lower(llama.params, test_tokens, test_attention_mask)

            hlo_text = lowered.as_text()

        print("📊 HLO Statistics:")
        print(f"   - Total characters: {len(hlo_text):,}")
        print(f"   - Total lines: {len(hlo_text.split(chr(10))):,}")

        # Save full HLO to file
        hlo_filename = "stablehlo_output.txt"
        with open(hlo_filename, "w") as f:
            f.write(hlo_text)

        print(f"💾 Full StableHLO saved to: {hlo_filename}")

        # Print first 100 lines as preview
        hlo_lines = hlo_text.split("\n")
        print("\n" + "=" * 80)
        print("🔍 STABLEHLO PREVIEW (first 100 lines):")
        print("=" * 80)
        for i, line in enumerate(hlo_lines[:100]):
            print(f"{i+1:3d}: {line}")

        if len(hlo_lines) > 100:
            print(f"... ({len(hlo_lines) - 100} more lines in file)")

        print("\n" + "=" * 80)
        print(f"📄 Complete StableHLO available in: {hlo_filename}")
        print("=" * 80)

    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
        )
    if monitor_memory:
        memory_monitor.print_current_usage("After Generation")
        memory_monitor.stop_monitoring()
    print("✅ Generation complete.")
    print("🧠 Output:", llama.tokenizer.decode(results[0]))

    if not os.path.isdir("results"):
        os.mkdir("results")

    np.savetxt("results/multi_chip.txt", results, fmt="%d")


if __name__ == "__main__":
    fire.Fire(main)
