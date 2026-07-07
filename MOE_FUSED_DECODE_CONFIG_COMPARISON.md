# tt-xla fused-MoE decode config vs. tt-metal tested configs (per op)

**Question:** does the tt-xla `test_gpt_oss_120b_tp_moe_fused_galaxy` run the **same op configs** that tt-metal's own tests validate?

**Answer in one line:** After the #7/#9/dispatch-algo fixes, tt-xla's *original* `(4,8)`/`cluster_axis=0` config **matches tt-metal's canonical config** on every macro dimension (fabric, dispatch algo, topology, mapping, model shape). But (a) the **current experiment** config `(8,4)`/`cluster_axis=1` is **not exercised by any tt-metal test**, and (b) — the important part — **tt-metal barely tests this path on a Blackhole galaxy at all**: the standard MoE suites are Wormhole-galaxy + 1D submeshes + `num_links=4`; on a BH `(4,8)` galaxy the only coverage is a few 2026 repro tests (some written for *this* tt-xla issue, some expected-to-hang) plus `moe_gpt_e2e` (which hangs on BH), and the gpt-oss reference model **disables the fused path on Blackhole entirely**.

tt-metal ref: `v0.74.0-dev20260621-14-g3a5f80334c1`. tt-xla config read from generated ttnn IR (`…runeee1_g1…mlir`) + runtime.

---

## 0. The structural finding first (this dominates everything below)

| Where tt-metal actually tests the MoE ops | Arch | Mesh | num_links |
|---|---|---|---|
| `test_moe_compute_6U.py` (standard nightly) | **Wormhole galaxy (6U)** | 1D: `1x8`, `1x16`, `16x1`, `8x1` | **4** |
| `test_selective_combine_6U.py` | **Wormhole galaxy** | `1x8`, `1x16` | **4** |
| `test_all_to_all_dispatch_metadata_6U.py` | **Wormhole 6U** (torus-descriptor gated) | `1x8`, `1x16`, `16x1` | **4** |
| `test_all_to_all_dispatch_6U.py` / `_combine_6U.py` | **Wormhole galaxy (TG)** | `8x4`, `8x8`, `8x16` | **4** |
| `moe_compute` **BH Loudbox** (`*-bh_lb`) | Blackhole **box** (8 chips) | `1x8`, `8x1` | **2** |
| `test_moe_gpt_e2e.py` | arch-agnostic → runs on attached galaxy | **`4x8`** | **2** (upstream had **4**) |
| `*_4x8_*_repro.py` (moe_compute / a2a-metadata / chain) | **Blackhole galaxy** | **`4x8`** | **1** |

- **No standard tt-metal MoE suite runs the fused decode on the full 2-D `(4,8)` Blackhole galaxy.** The moe_compute repro header states this verbatim ("not tested on the full 2D (4,8) galaxy by the standard suite").
- The gpt-oss **reference model** gates the fused throughput path off on BH: `throughput_experts_supported_on_arch()` = `not is_blackhole()`; the demo `pytest.skip`s it and runs only the batch-1 low-latency path. ⟹ **validated on Wormhole galaxy only.**
- ⟹ tt-xla is running this on hardware (BH `(4,8)` galaxy) where tt-metal's fused MoE path has essentially no green test coverage — consistent with the fabric-hang escalation.

---

## 1. `all_to_all_dispatch_metadata` (token dispatch)

| Dimension | tt-xla `(4,8)` original | tt-xla `(8,4)` current | tt-metal `moe_gpt_e2e` (`4x8`) | tt-metal BH `(4,8)` repro | tt-metal `_6U` unit (WH) |
|---|---|---|---|---|---|
| mesh | `4x8` | `8x4` | `4x8` | `4x8` | `1x8`/`1x16`/`16x1` |
| cluster_axis | **0** ✅ | 1 ⚠️ | **0** | **0** (B3 pass) / 1 (B1 hang) | 0/1 |
| fabric | **FABRIC_1D_RING** ✅ | FABRIC_1D_RING | **FABRIC_1D_RING** | FABRIC_1D_RING (B3) / FABRIC_2D (B1 hang) | 1D_RING/1D |
| dispatch_algorithm | **SPARSE_UNICAST** ✅ | SPARSE_UNICAST | **SPARSE_UNICAST** | SHORTEST_PATH | SHORTEST_PATH |
| num_links | `nullopt` → auto ⚠️ | auto | **2** (upstream 4) | **1** | **4** |
| num_devices (axis) | 4 ✅ | 4 | 4 | 4 | 8/16 |
| worker_mode | DIRECT (default) ✅ | DIRECT | DIRECT | DIRECT | DIRECT/MUX (perf) |
| persistent | yes ✅ | yes | yes | yes | yes |
| expert_mapping value | **global** (#9 fix) ✅ | global | **global** | global (B3) / axis-local = hang (B4) | global |
| dispatch_core_axis | **unset** (default) ⚠️ | unset | **ROW** | ROW | ROW |
| dtypes | bf16 / uint16 ✅ | same | bf16 / uint16 | same | same |
| hidden / top_k | 2880 / 4 ✅ | same | 2880 / 4 | 2880 / 4 | 7168 / 8 |

**Match verdict:** tt-xla `(4,8)` original == the tested BH path (`moe_gpt_e2e` + B3 repro) on cluster_axis, fabric, dispatch algo, worker mode, persistent, expert-mapping, model shape. Divergences: **num_links** (auto vs pinned 1–2), **dispatch_core_axis** (unset vs `ROW`), and the **current `(8,4)`/ca=1** experiment which no tt-metal test covers.

---

## 2. `moe_compute` (tilize → matmul → combine)

| Dimension | tt-xla `(4,8)` orig | tt-xla `(8,4)` cur | tt-metal `4x8` BH repro | tt-metal single-card gpt_oss | tt-metal `_6U` (WH) |
|---|---|---|---|---|---|
| mesh | `4x8` | `8x4` | **`4x8`** ✅ | `1x1` | `1x8`/`8x1` |
| cluster_axis | **0** ✅ | 1 ⚠️ | **0** | None (compute_only) | 1 / 0 |
| topology | **Ring** ✅ | Ring | **Ring** | n/a | Ring / Linear |
| num_links | default ⚠️ | default | **1** | n/a | **4** |
| activation | **swiglu** ✅ | swiglu | swiglu | swiglu | swiglu (gpt_oss) |
| has_bias | **true** ✅ | true | true | true | true (gpt_oss) |
| intermediate_size | **2880** ✅ | 2880 | 2880 | 2880 | 2880 (gpt_oss) |
| output_height_shard_dim | **4** ✅ | 4 | (repro) | 4 | 4 |
| mux_core_range | `(1,1)-(3,3)` ✅ | same | `(1,1)-(3,3)` | — | `(1,1)-(3,3)` |
| experts_per_device | **see note** ⚠️ | see note | 2 (repro) | 4 | 4 (gpt_oss) |
| tilize drain core | `(11,9)` ⚠️ | `(11,9)` | (config-derived) | — | (config-derived) |

**Match verdict:** tt-xla `(4,8)` matches the BH `(4,8)` moe_compute repro on mesh/cluster_axis/topology/activation/bias/intermediate/shard-dim/mux. That repro is the **only** moe_compute coverage on a BH galaxy. Standard moe_compute nightly is WH-only + 1D + `num_links=4`.

> **⚠️ experts_per_device — the one unconfirmed dimension.** tt-metal's gpt_oss moe_compute uses `experts_per_device = 4` (single-card / e2e) or `2` (the reduced `4x8` repro). tt-xla's per-device gate/up weight tensor is `8x1x32x6x2912x128` (bfp_bf4). The leading `8` *looks* like epd=8, but the tiled/folded ttnn layout (dims `32` and `6` also fold in) means it does **not** reverse-engineer cleanly to an epd count from the IR alone. The **model shape is identical** to tt-metal's gpt_oss (128 experts, hidden=intermediate=2880, top-4), and the a2a `num_devices=4` matches — so if there's any divergence it is in the **expert sharding factor**, not the collective topology. **This is the single dimension worth confirming directly** (e.g., print `experts_per_device` in the tt-xla moe_compute lowering vs. tt-metal's `experts_total // num_devices`).

---

## 3. The combine (`selective_reduce_combine`, run *inside* `moe_compute`)

| Dimension | tt-xla | tt-metal `moe_gpt_e2e` | tt-metal `_6U` (WH) |
|---|---|---|---|
| op | `selective_reduce_combine` (via moe_compute) ✅ | `selective_reduce_combine` | `selective_reduce_combine` |
| topology | **Ring** ✅ | **Ring** | Ring |
| cluster_axis | 0 / (1) | 0 | 1 |
| num_links | default ⚠️ | 2 | 4 |
| token/data parallel cores | from moe_compute mux `(1,1)-(3,3)` | COMBINE_H=4 / COMBINE_W=3 | 4 / 4 |
| hidden / k | 2880 / 4 ✅ | 2880 / 4 | 7168 / {1,2,8} |

Note: gpt-oss uses **`selective_reduce_combine`**, not `all_to_all_combine`. `all_to_all_combine` and `reduce_scatter` appear only in the **deepseek** decode chain (a different model), so they are not part of the tt-xla gpt_oss path being compared.

---

## 4. Bottom line — same config or not?

**Same (once reverted to `(4,8)`/`cluster_axis=0`):** fabric `FABRIC_1D_RING`, `SPARSE_UNICAST`, combine `Ring`, `WorkerMode.DIRECT`, persistent a2a, **global** expert-mapping (#9), and the full gpt-oss model shape (128 experts / 2880 / top-4 / swiglu+bias / `output_height_shard_dim=4` / mux `(1,1)-(3,3)`). This is exactly tt-metal's `moe_gpt_e2e` + `4x8` BH repro config.

**Divergences:**
1. **Mesh/axis (current):** tt-xla is presently on the experiment `(8,4)`/`cluster_axis=1` — **no tt-metal test uses this**; the reference is `(4,8)`/`cluster_axis=0`. (Both hang identically — proven — so this doesn't change the outcome, but it is off-reference.)
2. **num_links:** tt-xla passes `std::nullopt` → `get_num_links()` auto (counts routing planes); tt-metal **pins** it — `1` on BH galaxy repros, `2` in `moe_gpt_e2e`, `4` on WH. tt-xla's auto value is not pinned to the BH-safe `1`.
3. **dispatch_core_axis:** tt-xla doesn't set it (uses the PJRT/tt-metal default); `moe_gpt_e2e` sets `DispatchCoreAxis.ROW`.
4. **experts_per_device:** unconfirmed (§2 note) — the one dimension to verify directly.

**Coverage gap (the real headline):** the config tt-xla runs — fused MoE decode on a **2-D `(4,8)` Blackhole galaxy** — is **not covered by any standard tt-metal test** (those are Wormhole-galaxy + 1D + `num_links=4`), only by 2026 repro tests (some of which are the tt-xla issue-#7 hang repros), while `moe_gpt_e2e` itself hangs on BH and the reference model disables the path on BH. So "are we running the tested config?" → **the closest tt-metal config is validated on Wormhole, not on the Blackhole galaxy we run on.**
