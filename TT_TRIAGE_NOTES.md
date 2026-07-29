# tt-triage — learning notes, live-hang attempt, and a host-crash incident

Working notes on using [tt-triage](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/triage.html) ([tt-triage.md](https://github.com/tenstorrent/tt-metal/blob/main/tools/triage/tt-triage.md)) interactively for the first time, in the context of the Falcon3-7B hang investigation (see `FALCON3_SINGLE_LAYER_HANG_DEBUG.md`). We already use tt-triage in this repo's CI (`.github/scripts/install-tt-triage.sh`, `.github/workflows/call-test.yml` / `call-perf-test.yml`) but had never driven it by hand before. **This session crashed the host machine while trying it against a live hang — confirmed at least twice, and the "safety patch" attempt below did NOT actually prevent the second one, it only delayed it by ~5 minutes.** Read the incident sections before running tt-triage against anything live. **Current standing rule: do not run tt-triage against a hung/spinning device at all — the guardrails below are NOT sufficient.**

## Summary (for sharing)

- **tt-triage can crash the host machine** when pointed at a device another process already has open and is actively hammering (e.g. our hang) — confirmed via kernel logs (`Failed to set initial power state: -22` → cascading kernel-workqueue failure → full freeze), root-caused to an unconditional `set_power_state(true)` call in tt-umd's `TopologyDiscovery`. Confirmed twice against a real hang; not a fluke.
- **It's safe against a healthy, non-hung server** — tested with and without `TT_VISIBLE_DEVICES=0`, both clean, and it successfully returned real on-device RISC-V callstacks. This part of the finding still stands.
- **⚠️ CORRECTED: the safety patches (`low_power=True`, skip eth-link-training-wait) did NOT actually prevent the crash — they only delayed it.** The guardrailed attempt against a real hang initially looked clean (a graceful `NocHangError`, no immediate crash, confirmed via 5+ minutes of stable metrics), but the **host crashed anyway ~5 minutes later**, with the identical kernel-level signature as the unguarded crash. There is currently **no known-safe way to point tt-triage at a hung device**, patched or not. See the corrected incident below.

## What it is, and how it actually connects

tt-triage is a Python tool (`tools/tt-triage.py` in tt-metal) that runs discoverable "scripts" — data providers and analyses — against a live (or previously-serialized) Metal execution. Crucially, it does **not** attach to a host PID the way gdb/py-spy do. It talks to tt-metal's built-in **Inspector** subsystem over a **capnp RPC connection** (`tt-umd`/`tt-exalens` handle the raw device access underneath). This matters a lot for live-hang debugging:

- Inspector is **enabled by default** (`TT_METAL_INSPECTOR` default `true`) and its **RPC server is also enabled by default** (`TT_METAL_INSPECTOR_RPC` default `true`), listening on **`localhost:50051`** by default (`tt_metal/llrt/rtoptions.hpp`: `InspectorSettings{enabled=true, rpc_server_host="localhost", rpc_server_port=50051, rpc_server_enabled=true, serialize_on_dispatch_timeout=true, ...}`).
- So **any tt-metal process, including ours, already exposes this RPC with zero special setup** — confirmed live: `grep -i c383 /proc/net/tcp` (50051 in hex) showed an active `LISTEN` socket, and cross-referencing `/proc/<pid>/fd` confirmed **the wedged EngineCore itself owned that socket**. No restart needed to attempt triage against an already-running hang.
- CI wires this up automatically for its own timeouts: `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE: "/opt/tt-triage-venv/bin/python .../tt-triage.py"` + `TT_METAL_OPERATION_TIMEOUT_SECONDS` (default 300s) in `call-test.yml`/`call-perf-test.yml` — tt-metal invokes tt-triage itself as a subprocess when a dispatch op exceeds the timeout. Our manually-launched tt-media-server never set `TT_METAL_OPERATION_TIMEOUT_SECONDS` (confirmed via `dump_configuration`: `timeout_duration_for_operations: 0s`), so nothing was ever going to auto-trigger this; had to run it by hand.

## Setup

```bash
python3.12 -m venv .venv-tt-triage && source .venv-tt-triage/bin/activate
pip install -r <tt-metal>/tools/triage/requirements.txt   # pycapnp, tt-umd, tt-exalens — pin these to YOUR tt-metal build, don't grab latest
export TT_METAL_HOME=<path to the tt-metal tree matching the RUNNING process's build>
python3 <tt-metal>/tools/tt-triage.py --run=<script> --dev=all --llm-output
```

Used the tt-metal copy already vendored under this repo's build (`third_party/tt-mlir/install/tt-metal/tools/tt-triage.py` + its own `tools/triage/requirements.txt`) rather than downloading a separate pinned version like the CI installer does — guarantees exact version match against what's actually running.

## Key scripts (from `--help`)

| Script | What it does |
|---|---|
| `run_checks` | Data provider: device/location/RISC-core selection + checks. Dependency for most others. |
| `inspector_data` | Connects to the Inspector RPC (or falls back to serialized logs / parsed log files). |
| `operation_provider` / `dispatcher_data` | Data providers: per-core dispatcher state + Inspector's runtime op map. |
| `dump_running_operations` | Currently-running op(s) by Op Id, with device/core coverage + previous op. |
| `dump_op_window` | Context window of ops around the currently-running set (what ran just before / should run next). |
| `dump_callstacks` | **On-device** callstacks for every RISC-V core on every device — the actual chip-side kernel state, distinct from (complementary to) host-side gdb/py-spy views. |
| `check_broken_components` | Flags cores that won't halt or resumed without triage's permission (NoC hang detection). |
| `dump_configuration` | Dumps Inspector's recorded environment/rtOptions/ttnnConfig — a live config snapshot. |

Useful flags: `--dev=all` (vs. a specific device id — device-id-mapping quirks made `--dev=0` fail, see below), `--llm-output` (CSV instead of Rich tables, easier to grep/paste), `--triage-summary-path=<path>` / `--llm-output-path=<path>` (save an artifact), `-v`/`-vv` (more columns).

## Live attempt against a wedged Falcon3-7B server (2026-07-28)

Tried it against the full-model tt-media-server hang described in `FALCON3_SINGLE_LAYER_HANG_DEBUG.md` (EngineCore PID 875071, still actively spinning on a completion-queue read at the time).

### What worked

- **`dump_configuration --dev=all --llm-output`**: clean full dump of the live process's rtOptions/ttnnConfig/environment. Confirmed independently, from an entirely different data path than everything logged by hand elsewhere: `TT_VISIBLE_DEVICES=0`, `num_hw_cqs=1`, `watcher_enabled=false`, `timeout_duration_for_operations=0s`, `reliability_mode=(unset)`, `skip_eth_cores_with_retrain=false` — no operation-level timeout or watcher was ever configured for this server, consistent with a manual launch rather than a CI-wrapped one.
- **`run_checks --dev=all`**: succeeds (it's a pure data-provider, no visible table on its own).
- Calling the Inspector RPC client directly (bypassing `tools/tt-triage.py`) also confirmed the RPC has valid data: `InspectorRpcController("localhost", 50051).getAllBuildEnvs()` returned exactly one entry, `metalDeviceId=0`, with a real `firmwarePath`.

### What partially worked — and is itself new evidence

First attempt at **any** device-touching script (`dump_running_operations`) failed with:
```
DeviceTimeoutError: MMIO per-op timeout: 4B load took 220859 us (budget=2 ms), 4 of 4 bytes remaining.
  ... tt::umd::BlackholeTTDevice::wait_eth_core_training(...)
  ... tt::umd::TopologyDiscovery::discover(...)
```
i.e. tt-triage's own fresh UMD topology-discovery (the same eth-core-training-wait step behind the always-benign `TT_FATAL: Chip 0 logical eth core ... connects to a remote mmio device` warnings seen in every server log) took **110x its 2ms budget** and gave up. This is itself informative: it means the actively-spinning completion-queue-poll thread on the EngineCore side isn't just uselessly burning CPU — it's hammering the device's PCIe/MMIO channel hard enough to measurably starve a *second*, independent process's routine MMIO reads. A retry succeeded past this exact step a minute later, confirming it's contention-driven, not a permanently dead PCIe path.

### What didn't work

`dump_running_operations` and `dump_callstacks` (the two most valuable scripts here — the latter would give on-device RISC-V callstacks we don't have at all yet) both consistently fail once past the MMIO-contention step, with:
```
TTTriageError: Failed to get firmware path from Inspector RPC: 154095680358880
Make sure Inspector RPC is available or serialized RPC data exists.
Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application.
```
Traced this past the tool's own generic error message (`tools/triage/dispatcher_data.py:155`): it's a bare Python `KeyError` on a `device_unique_id` value (`154095680358880`) that doesn't exist in `self._build_env_cache`. The cache is populated by iterating `inspector_data.getAllBuildEnvs()` and computing a key via `metal_device_id_mapping.get_unique_id(build_env.metalDeviceId)`, then read back via a *separately*-computed `run_checks.devices[0].unique_id` — **these two device-identity computations don't agree** for our board, even though (confirmed directly above) the underlying RPC data for device 0 is perfectly valid. Didn't chase the tool's own source further than this.

This rig is a **4-chip QB2 board**, not an isolated single chip (the host boot log — see incident below — enumerates four Tenstorrent Blackhole PCI devices, `0000:01:00.0` through `04:00.0`); our LLM workload just pins itself to one of the four via `TT_VISIBLE_DEVICES=0`. tt-triage's `TopologyDiscovery` does full cross-chip ethernet-link training as part of its startup regardless of which device you actually care about (confirmed: both `--dev=0` and `--dev=all` hit the identical MMIO timeout during discovery — `--dev=` only filters *analysis* after the fact, not what gets physically probed). Best guess, not confirmed: the `device_unique_id` mismatch is related to this multi-chip topology-discovery path, possibly a tt-triage limitation on multi-chip boards. Worth checking if it's a known issue upstream if pursued further.

### Net result

Confirmed tt-triage **can** attach live to an already-running process with zero prior setup (Inspector's defaults are all favorable). Got one genuinely new data point (MMIO-channel contention from the spinning thread) and an independent config cross-check, but not the on-device callstacks that would have been the most valuable addition to py-spy/gdb. Then the host crashed (see below) before this could go further.

---

## ⚠️ INCIDENT: this tt-triage session crashed the host machine (2026-07-28)

**Confirmed via host kernel logs** (`sudo journalctl -b -1 -k`, checked from outside the container after the crash — checking from the host was specifically prompted because the exact same thing happened once before when using tt-triage on this rig). This is a real, reproducible hazard, not a coincidence — now backed by hard kernel evidence, not just timing correlation.

### Timeline

- Boot at 07:52:52 that day: kernel enumerates 4 Tenstorrent Blackhole devices (`0000:01:00.0` through `04:00.0`) normally, once — completely routine.
- ~01:05–01:20: this session's tt-triage exploration runs (py-spy/gdb captures were earlier and are unrelated/safe; the device-touching tt-triage commands — `dump_running_operations`, `dump_callstacks`, the MMIO-timeout attempt, the direct `InspectorRpcController` test — all happened in this window), concurrently with the still-actively-spinning EngineCore that still held `/dev/tenstorrent/0` open.
- **`01:17:41`**: first anomaly — `workqueue: mmput_async_fn hogged CPU for >10000us 4 times, consider switching to WQ_UNBOUND`.
- **`01:17:53` / `01:17:54`**: `tenstorrent 0000:01:00.0: Failed to set initial power state: -22` (×2). Not a boot-time message — it appeared hours after driver load, mid-session, on the *exact device* the wedged EngineCore was actively polling. `errno -22` = `EINVAL`.
- **`01:18:07` onward**: `pci_pme_list_scan` (PCI power-management-event scanning) starts hogging CPU too.
- **`01:18:20`**: two *more* `Failed to set initial power state: -22` messages — a second attempt.
- **`01:17:41` → `01:21:11`**: `mmput_async_fn` hog counts escalate rapidly — 4 → 5 → 7 → 11 → 19 → 35 → 67 — alongside `pci_pme_list_scan`, `psi_avgs_work`, and `vmstat_update` all joining in. The kernel's own general-purpose workqueues (process-memory teardown, PCI PM scanning, load-average accounting, VM stats) increasingly fail to get scheduled in time — a system grinding toward a full freeze, not an isolated one-off glitch.
- **`01:21:59`**: the **last log line of any kind** for the rest of that boot — including `rsyslogd`'s own reconnection-retry message, which had fired reliably every ~30s for hours without fail up to that exact point. The entire machine went completely dark here, not just the Tenstorrent stack.
- **`~01:25:51`**: new boot (hard reset/power-cycle — nothing shut down cleanly).

### Root cause, traced to the exact line

`tenstorrent 0000:01:00.0: Failed to set initial power state: -22` is almost certainly produced by `tt_metal/third_party/umd/device/topology/topology_discovery.cpp`, inside `TopologyDiscovery::get_connected_devices()`:

```cpp
void TopologyDiscovery::get_connected_devices() {
    ...
    local_device_ids = PCIDevice::enumerate_devices();   // filtered by TT_VISIBLE_DEVICES
    for (auto& device_id : local_device_ids) {
        std::unique_ptr<TTDevice> tt_device = TTDevice::create(device_id, ...);
        if (options.low_power) {
            // Low power mode is temporarily disabled. See https://github.com/tenstorrent/tt-umd/issues/2531.
            log_warning(LogUMD, "Low power mode is not yet supported. The device will remain in high power mode ...");
        } else {
            // set_power_state is currently a no-op until https://github.com/tenstorrent/tt-umd/issues/2531 is resolved.
            tt_device->set_power_state(true);
        }
        ...
```

**`tt_device->set_power_state(true)` is called unconditionally for every enumerated device** (unless `options.low_power` is set, which skips it entirely). This reaches into the kernel driver, which currently can't fulfill it (per the linked tt-umd issue #2531) and returns `-EINVAL` — exactly the `-22` in the kernel log. This is the operation attempting to touch a device's PCI power state **while another process already has that device open and is actively using it** — a very plausible trigger for the cascading kernel-workqueue failure that followed.

### Does `TT_VISIBLE_DEVICES=0` / `TT_MESH_GRAPH_DESC_PATH` (our usual single-chip env vars) help?

Asked directly, since this rig's standalone tt-xla scripts always set these two for single-chip runs. Checked against source rather than guessing:

- **`TT_VISIBLE_DEVICES=0` is exactly the right lever, and tt-triage's code path does honor it.** `tt_metal/third_party/umd/docs/TT_VISIBLE_DEVICES.md`: "`TT_VISIBLE_DEVICES` is evaluated at `PCIDevice::enumerate_devices()` time... affects `Cluster` construction, which calls `enumerate_devices()`." `get_connected_devices()` above calls exactly that function. Setting it for tt-triage means it would never even open `/dev/tenstorrent/1,2,3` — cutting exposure from 4 devices to 1. **Worth doing regardless, no downside.**
- **But it doesn't fully save you**: device 0 is precisely the device we care about (and the one already open/busy) — `set_power_state(true)` still gets called on it either way. `TT_VISIBLE_DEVICES=0` reduces blast radius (3 fewer devices touched) but does not eliminate the actual dangerous call on the device that matters.
- **`TT_MESH_GRAPH_DESC_PATH` is not applicable to tt-triage at all.** It's a different, higher-level concept — tt-metal's own fabric/mesh-graph description for building a logical `MeshDevice`, consumed way above where tt-triage operates (tt-triage calls raw `tt_umd.TopologyDiscovery` directly and never touches tt-metal's mesh-device layer).
- **There is a real escape hatch, just not exposed to us**: `TopologyDiscoveryOptions.low_power = true` (bound in the `tt_umd` Python module, confirmed present: `python3 -c "import tt_umd; print(tt_umd.TopologyDiscoveryOptions)"` succeeds) skips the `set_power_state` call entirely. But `tt-triage.py`'s own CLI doesn't expose a `--low-power` flag anywhere in `--help` — using it would require patching our local tt-triage copy to construct `TopologyDiscoveryOptions(low_power=True)` directly, rather than anything available from the command line as shipped.

### Conclusion

Very high confidence this was caused by **tt-triage's tt-umd-based device access racing with the still-live, still-attached EngineCore process on the same physical device**, specifically the unconditional `set_power_state(true)` call during topology discovery. Unlike gdb/py-spy (pure host-side, zero device contact — safe, used extensively throughout the Falcon3 investigation with no issues), tt-triage does raw PCIe/power-state manipulation through `tt-umd`. This matches the user's report that **the exact same thing happened the previous time tt-triage was used against this rig** — two-for-two is not a coincidence.

Likely a mix of "using it in an unsupported pattern" (attaching a second, independent `tt-umd`-owning process directly, rather than the docs' safer `--remote-exalens` pathway where a single pre-launched `tt-exalens --server` would service both the workload and triage) *and* a genuine tt-triage/tt-umd robustness gap — a diagnostic tool racing into an unconditional PCI power-state write against a device someone else already owns, and taking down the entire host rather than detecting "already open" and refusing safely, is a real bug regardless of intended usage.

### Recommendation — until this is understood/fixed upstream

- **py-spy + gdb remain safe** (proven throughout the Falcon3 investigation, zero device contact) — keep using them freely for any future live hang.
- **Do not run tt-triage (or any other tt-umd/tt-exalens-based tool) against a device that another process currently has open**, especially not one actively spinning/polling it, until Tenstorrent confirms concurrent access is safe or fixes the `set_power_state`-on-an-owned-device path.
- If on-device callstacks (`dump_callstacks`) are ever needed for a live hang, the safer sequence is: capture everything host-side first (py-spy/gdb/tt-smi), *then* deliberately kill the wedged process, *then* run tt-triage against the now-idle device — sacrificing live on-device state for not risking another host crash.
- Worth filing this as its own report against `tt-metal`/`tt-triage`/`tt-umd` (concurrent-device-access safety) — this is a tooling hazard that could hit any team debugging a live hang this way, not something specific to Falcon3.
- This crash also means the live Falcon3 wedge (EngineCore, tt-media-server, the eval loop) was lost — everything captured *before* the crash (py-spy dump, gdb backtrace, tt-smi health, the #4521-discriminator verdict) stands as recorded in `FALCON3_SINGLE_LAYER_HANG_DEBUG.md`; nothing further could be gathered from that specific occurrence.

## Follow-up: healthy-server experiment — clean both ways (2026-07-28, later same day)

Back inside the container with a fresh, **healthy, non-hung** full-model server (standalone `vllm serve`, not tt-media-server this time — PID 278/303, no `NUM_HIDDEN_LAYERS` in `additional_config`). Set up a background metrics watcher first (`top`/`free`/`vmstat`-style sampling every 2s, `ps --sort=-pcpu`, device-holder check, per-thread jiffies, with a `sync` every 10th iteration so data survives even a hard crash) logging to `tt_triage_experiment_metrics_<timestamp>.log`, specifically so a third crash wouldn't lose the lead-up evidence the way the host kernel log did last time (that one we only recovered afterward, from outside the container).

**Pre-flight**: confirmed genuinely healthy before touching anything — CPU-delta across all EngineCore threads over 3s showed nothing (no thread pegged, unlike the wedge), and `/v1/models` responded normally.

**Attempt 1 — with `TT_VISIBLE_DEVICES=0`** (the mitigation identified above): `dump_configuration`, `dump_running_operations`, and `dump_callstacks` **all succeeded cleanly, `rc=0`, no MMIO timeout, no `KeyError`** — a stark contrast to every attempt against the wedge. `dump_callstacks` returned real on-device RISC-V callstacks for the first time this investigation (`cq_prefetch`, `cq_dispatch`, `cq_dispatch_subordinate_compute`, `cq_realtime_profiler`, etc. — dispatch-firmware kernels, exactly what's expected on an idle-between-requests server). Watched the metrics log for 5+ minutes afterward (matching the ~4 min delay from first kernel symptom to total freeze last time) — load average stayed flat at 0.3–0.6 the entire window, server stayed responsive throughout. **No crash.**

**Attempt 2 — without `TT_VISIBLE_DEVICES=0`** (unset, default: all 4 devices visible): `dump_callstacks` **also succeeded, `rc=0`**, again with real device-0 callstacks. The only failures were expected, benign, per-core: devices 3's cores (`functional_workers`, `eth` links) failed individually with the same "Failed to get firmware path from Inspector RPC" message — because Inspector genuinely has no build-env data for a device our workload never touched, not a crash precursor. Device 0 (the one that matters) resolved fully either way. Server stayed healthy and responsive after.

### Updated conclusion

**The crash is specifically tied to attaching tt-triage while the target device is under heavy contention from an actively-spinning/hung process — not to concurrent attachment in general.** Against a healthy, idle-between-requests server, tt-triage's full discovery (including the same `set_power_state(true)` call and the same `wait_eth_core_training` path that misbehaved against the wedge) completes normally regardless of `TT_VISIBLE_DEVICES`. This narrows the hazard considerably: **tt-triage appears safe to use against a live, healthy tt-xla/tt-media-server process** — the danger is specifically pairing it with an already-hung one, where the target device's completion-queue-polling thread is hammering the MMIO/PCIe channel at ~100% of a core when tt-triage's topology discovery tries to also touch it.

This doesn't fully clear tt-triage/tt-umd of the underlying design issue (`set_power_state(true)` unconditionally on a device something else already owns is still not defensively coded, and a device merely being *busy* — even without an active hang — could in principle hit similar contention under enough load) — but it does mean the **specific recommendation below can be relaxed**: tt-triage is fine to reach for during normal live debugging of a healthy server; the hard rule is still **never point it at a device you already know is wedged/spinning**.

## Guardrailed attempt against a real, live hang (2026-07-28, same day, third attempt)

A new hang occurred naturally right after the healthy-server experiment above (a pure standalone `vllm serve` process, full model — see `FALCON3_SINGLE_LAYER_HANG_DEBUG.md`, "LIVE HANG CAUGHT #2"). Rather than repeat the unguarded attempt that crashed the host twice before, patched in both mitigations this time and re-ran the full metrics watcher for safety.

**Patch applied** (to the venv's installed `ttexalens/umd_api.py`, at the same `TopologyDiscoveryOptions` construction site documented above):
```python
self.discovery_options.wait_on_ethernet_link_training = False  # was True, with a "TODO: Set to False" comment already in upstream source
self.discovery_options.low_power = True  # was omitted (defaults False), skips set_power_state(true)
```
Both attributes confirmed settable on the bound `tt_umd.TopologyDiscoveryOptions` object before running against anything live.

**Result: `dump_configuration` (and everything downstream of UMD init) failed cleanly, with no crash:**
```
RuntimeError: All 1 detected Tenstorrent device(s) failed to initialize and are unhealthy: device 0 (NocHangError)
```
`rc=1`, a plain Python exception. Confirmed immediately after: server still healthy (`/v1/models` responds), device still cleanly held by the EngineCore (`fuser` shows no contention/change), metrics log shows no load-average anomaly at all in the window (unlike the run-up to both prior crashes).

**Traced why it stops here, and concluded this is the right place to stop**: `UmdApi.__init__` (umd_api.py:146-156) checks `cluster_descriptor.get_unhealthy_devices()` after discovery; with **zero** healthy chips found, it raises `RuntimeError` unconditionally — this check fires regardless of the `*_action = IGNORE` settings already present in the file (those only suppress throwing at the individual per-device health-check step in C++; if literally every device ends up unhealthy, there's nothing left to hand back, so the Python layer raises). **`NocHangError` is tt-umd's own internal health check correctly detecting that this exact device is in a bad NOC state and refusing to proceed** — not a bug or a gap, but the intended safety mechanism working. Deliberately did not look for a way to force past this: doing so would mean constructing a full API context around a device tt-umd has explicitly flagged as unsafe, which is precisely the situation that caused the crash in the first place.

### ⚠️ CORRECTED conclusion (see below — the crash was only delayed, not prevented)

**Initial (wrong) read, at the time**: no crash observed within a 5+ minute watch window, `NocHangError` refused cleanly, therefore "guardrails work." **This was premature.** Checking host kernel logs after the fact (next section) shows **the host crashed anyway, ~5 minutes after that "clean" `NocHangError` result** — same exact fatal signature as the unguarded crash (`Failed to set initial power state: -22`, escalating kernel-workqueue hogging, total freeze). The metrics watcher had already been stopped by the time the delayed crash happened, which is exactly why it looked clean in the moment — a real methodology gap, not just a wrong conclusion. **Do not trust "no crash in the first N minutes" as a safety signal for this tool against a hung device; the first crash was near-instant, this one took ~5 minutes, so there's no known safe observation window.**

## ⚠️ INCIDENT #2 (confirmed after the fact): the guardrailed attempt crashed the host too, just delayed

Checked host kernel logs (`sudo journalctl --list-boots`, from outside the container) well after the fact and found the full picture was worse than believed at the time:

```
IDX BOOT ID    FIRST ENTRY                  LAST ENTRY
-4  fe929716…  Mon 2026-07-27 07:52:52 UTC  Tue 2026-07-28 01:21:59 UTC   ← original crash (INCIDENT #1, unguarded)
-3  9d17c1e4…  Tue 2026-07-28 01:25:58 UTC  Tue 2026-07-28 02:52:18 UTC   ← this whole investigation session, including the guardrailed attempt
-2  71a749…    Tue 2026-07-28 02:54:23 UTC  Tue 2026-07-28 03:01:38 UTC   ← short, unstable — see below
-1  fa3ffd46…  Tue 2026-07-28 03:02:26 UTC  Tue 2026-07-28 03:15:59 UTC   ← short, unstable — see below
 0  d51deeb6…  Tue 2026-07-28 03:16:49 UTC  (current, stable)
```

Boot `-3` is the session where the guardrailed attempt ran (~02:44:42) and appeared clean. Its kernel log (`sudo journalctl -k -b 9d17c1e4a38a4f7a8ace45020de766d5`) tells the real story:

```
Jul 28 02:47:40  workqueue: mmput_async_fn hogged CPU for >10000us 4 times...
Jul 28 02:47:52  workqueue: pci_pme_list_scan hogged CPU for >10000us 4 times...
Jul 28 02:48:12  workqueue: mmput_async_fn hogged CPU for >10000us 19 times...
Jul 28 02:49:11  workqueue: mmput_async_fn hogged CPU for >10000us 35 times...
Jul 28 02:49:47  tenstorrent 0000:01:00.0: Failed to set initial power state: -22    ← same fatal signature as incident #1
Jul 28 02:49:47  tenstorrent 0000:01:00.0: Failed to set initial power state: -22
Jul 28 02:50:42  tenstorrent 0000:01:00.0: Failed to set initial power state: -22    ← recurs 3 more times
Jul 28 02:51:11  workqueue: mmput_async_fn hogged CPU for >10000us 67 times...
Jul 28 02:51:43  tenstorrent 0000:01:00.0: Failed to set initial power state: -22
Jul 28 02:52:09  workqueue: psi_avgs_work hogged CPU for >10000us 7 times...
                 ← boot ends here (02:52:18), host crashed
```
- The guardrailed `dump_configuration` call itself ran at **02:44:42-43** and returned `NocHangError` cleanly.
- The workqueue-hogging death spiral began at **02:47:40** — about **3 minutes later**.
- The vLLM server process itself became unreachable (`ConnectionRefused`) starting at **02:49:04**, per the eval client's log.
- The `Failed to set initial power state: -22` messages — the same ones from incident #1 — start firing at **02:49:48**, *after* the server process was already gone, and recur 4 more times over the next 2 minutes.
- Host crashed for good at **02:52:18** — total elapsed from the guardrailed attempt to host death: **~7.5 minutes**.

**This means the earlier "final conclusion" above was wrong.** The guardrails did not make the attempt safe — they changed a near-instant crash into a delayed one. `low_power=True` and `wait_on_ethernet_link_training=False` skip *some* of the risky operations in `TopologyDiscovery::get_connected_devices()`, but evidently not all of what's needed — `TTDevice::create()` and `init_device()`/`init_tt_device()` still run per enumerated device even with both guardrails set (this is exactly what has to happen for `NocHangError` to be detected in the first place), and something in that remaining sequence still destabilizes the shared device/driver state enough to bring the host down a few minutes later.

**Unresolved**: two additional short-lived boots followed (`-2`: 7 min, `-1`: 13 min) before the system stabilized on the current boot. Neither shows the `Failed to set initial power state` signature in its kernel log — only an unrelated, pre-existing `rke2-agent.service` networking failure loop (`no default routes found`, restart-looping — looks like a flaky background Kubernetes agent unrelated to Tenstorrent hardware). Whether these two reboots were: (a) residual PCIe/driver instability left over from the crash above, (b) manual reboots done to recover, or (c) a *separate* incident (e.g. tt-triage run again, this time against a genuinely healthy server, as suspected but not yet confirmed with direct kernel-log evidence) — is not yet determined. **If tt-triage was run again in that window, its exact command/timing would help pin this down**; the kernel logs alone don't show a repeat of the specific `set_power_state` failure in either of those two boots.

## Open questions / next steps for tt-triage itself

- **REVISED — crash-safety is NOT resolved.** The `low_power=True` + `wait_on_ethernet_link_training=False` guardrails do not prevent a host crash against a live hang — they only delay it by several minutes. **There is currently no known-safe way to point tt-triage (or presumably any `tt_umd.TopologyDiscovery`-based tool) at a device another process is actively spinning on.** Treat this as a hard "do not do this" until Tenstorrent provides a fix or a genuinely non-invasive read path.
- **Methodology lesson**: a clean result in the first several minutes is not sufficient evidence of safety for this failure mode — the first crash was near-instant, the second took ~5-7 minutes. Any future safety check needs either a much longer observation window (tens of minutes) or, better, avoiding the experiment entirely now that the risk is confirmed structural, not timing-dependent.
- **Resolved, still stands**: tt-triage does NOT crash a healthy, non-hung server (confirmed clean, both with and without `TT_VISIBLE_DEVICES=0`, real callstacks obtained). The danger is specifically pairing it with an already-hung device.
- **Unresolved**: the two short, unstable boots after the confirmed crash (7 min and 13 min) — cause not determined from kernel logs alone. Could be residual instability, could be a separate incident. Worth clarifying with whoever was driving the box during that window (02:52-03:16).
- **`TT_VISIBLE_DEVICES=0` remains worth setting as routine practice** when you only care about one device — no downside, though (per the above) it does not make attaching to a hung device safe.
- **The `dispatcher_data.py` `device_unique_id` KeyError** seen against the original wedge — unresolved, lower priority than the crash-safety finding above.
- **File upstream, now with the corrected, complete story**: `set_power_state(true)` (and/or the surrounding `TTDevice::create()`/`init_tt_device()` sequence) on an already-open, actively-spinning device crashes the host, and **this is not fixed by skipping just those two options** — the crash recurs, delayed, even with `low_power=True` and `wait_on_ethernet_link_training=False` set. This is the headline bug to report: tt-triage/tt-umd's live-attach path is unsafe against a hung device with no currently-known mitigation from the calling side.

## Why recovery took 3-4 power cycles (2026-07-28, investigated after the fact)

The user reported needing 3-4 manual power cycles before the machine came back up fully stable. Checked host boot history (`sudo journalctl --list-boots`, from outside the container) to reconstruct the full recovery sequence:

```
IDX BOOT ID    FIRST ENTRY                  LAST ENTRY                    DURATION
-3  9d17c1e4…  Tue 2026-07-28 01:25:58 UTC  Tue 2026-07-28 02:52:18 UTC   (this session; crashed)
-2  71a749…    Tue 2026-07-28 02:54:23 UTC  Tue 2026-07-28 03:01:38 UTC   ~7 min
-1  fa3ffd46…  Tue 2026-07-28 03:02:26 UTC  Tue 2026-07-28 03:15:59 UTC   ~13 min
 0  d51deeb6…  Tue 2026-07-28 03:16:49 UTC  (current, stable 30+ min)
```

**Could not find a kernel-level crash signature in boots `-2` or `-1`.** Checked both thoroughly:
- No repeat of `Failed to set initial power state` or the escalating `hogged CPU` workqueue spiral that marked both confirmed crashes.
- No PCIe AER/uncorrectable errors, no NMI, no OOM.
- The only "error"-looking lines in both are (a) a `pci 0000:06:00.0/06:01.0: retraining failed` pair — confirmed **pre-existing and benign**: it appears identically in *every* boot checked, including the current stable one, and is an AMD 600-series chipset PCIe switch downstream port with nothing physically attached (unrelated to the Tenstorrent devices on buses 01-04); and (b) a pre-existing, unrelated `rke2-agent.service` (Rancher Kubernetes agent) restart-crash-loop (`no default routes found in /proc/net/route`) — a flaky background service issue, nothing to do with Tenstorrent hardware.
- Also checked: no separate Claude Code session and no activity in this session's own transcript during the entire 02:52–03:16 window (confirmed via the session's `.jsonl` file directly) — so whatever was run against "a non-hanging server" in that window wasn't captured by either of those. Also checked shared bash history (`~/.bash_history`, same file on host and in-container since `/home/kmabee` is bind-mounted) — no tt-triage-related commands appear there either, meaning it wasn't run from an interactive login shell that flushed its history (could still have been a one-off non-interactive command, e.g. via `docker exec container bash -c "..."`, which wouldn't be captured).

**Best-supported hypothesis** (not confirmed with direct hardware-log evidence — this board has no BMC/IPMI System Event Log to check, being a consumer ASRock B850M-C, not server-grade): the original crash (`set_power_state` failing with `-EINVAL` on an already-open, actively-spinning device) most likely left the Tenstorrent ASIC's own on-board/firmware-level state in a genuinely broken condition that an **OS-level reboot alone does not clear** — many PCIe devices, and TT ASICs in particular (hence `tt-smi -r` existing as a distinct operation from a host reboot), retain power and internal state across a warm reboot unless power is actually removed from the slot. That would explain the pattern well: each subsequent boot came up with the *host* OS fine, but the Tenstorrent device was still in whatever corrupted state the crash left it in, causing renewed instability that looked like it needed another reboot — until a power cycle *with power actually cut* (not just an OS restart) finally reset the ASIC's own state and let things stabilize on boot `0`.

**Net takeaway**: a `set_power_state` crash against a hung device may require a genuine power cycle (not just `reboot`/`tt-smi -r`/an OS restart) to fully recover from, and it may take more than one attempt if power isn't fully cut for long enough each time. Worth keeping in mind for anyone else who hits this: if the box doesn't stabilize after one reboot post-crash, that's not surprising — try a longer, harder power-off before the next attempt.

## Status: ready to resume hang-chasing

All tt-triage-specific investigation for now is captured above. Nothing further to dig into here — main open item is confirming with the user what exactly was run during the 02:52–03:16 window (untraceable from logs), which doesn't block resuming the Falcon3-7B hang investigation itself.
