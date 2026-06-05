# Config Update (proposed) — flux_2_dev vae

```
Provenance:
  tt-xla SHA       : b96e57426
  tt-foundry SHA   : 092701d87d (third_party/tt_forge_models @ flux branch)
  Generated        : 2026-06-05 10:18
  Source skill     : model-bringup-config-update
  Result classified from : json_report + stdout (1 passed, run_graph_test PCC 0.99)
```

**Result: PASSED** on arch **p150** (single_device VAE component).
Test surface: **pipeline component** → edit component test file only; **no runner YAML**.

`supported_archs` = `[p150]` (from `state.arch_results`; n150 not runnable on this Blackhole host — skipped by host probe, not a failure).

## Change: tests/torch/models/flux_2_dev/test_vae_decoder.py

Add the standard `record_test_properties` marker (matching `wan/test_wan_vae.py`)
so the VAE decoder is tagged `bringup_status=BringupStatus.PASSED`. The test
currently carries no bringup_status marker.

```diff
 import pytest
 import torch
 import torch_xla
 import torch_xla.runtime as xr
-from infra import Framework, run_graph_test
+from infra import Framework, RunMode, run_graph_test
 from infra.evaluators import ComparisonConfig, PccConfig
+from utils import BringupStatus, Category

 from tests.infra.testers.compiler_config import CompilerConfig
 from third_party.tt_forge_models.flux_2_dev.pytorch import ModelLoader, ModelVariant


 @pytest.mark.nightly
 @pytest.mark.model_test
 @pytest.mark.single_device
+@pytest.mark.record_test_properties(
+    category=Category.MODEL_TEST,
+    model_info=ModelLoader.get_model_info(ModelVariant.FLUX2_DEV_VAE),
+    run_mode=RunMode.INFERENCE,
+    bringup_status=BringupStatus.PASSED,
+)
 def test_vae_decoder():
```

PCC enforcement already present (no change needed):
```python
comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
```

## weight_fit.json
Set vae component `supported_archs: ["p150"]` (mirror).

## state.json
`stage: "passed"`, `details.supported_archs: ["p150"]`.

## No runner YAML change
`test_path` is under `tests/torch/models/` → runner
`test_config_inference_single_device.yaml` is **not** touched.
