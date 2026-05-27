# tt-mlir fix for Z-Image RoPE complex gather (#4756)

PyTorch repro tests are on tt-xla branch `akannan/zimage_shlo_bug`.

The SHLO→TTIR fix lives in **tt-mlir** (not tt-xla):

| Repo | Branch | Base commit |
|------|--------|-------------|
| [tt-mlir](https://github.com/tenstorrent/tt-mlir) | `akannan/zimage_shlo_bug` | `297f7eb6` |

## Build tt-mlir (other machine)

```bash
cd tt-xla/third_party/tt-mlir/src/tt-mlir   # or your clone
git fetch origin akannan/zimage_shlo_bug
git checkout akannan/zimage_shlo_bug
source env/activate
cmake -G Ninja -B build
cmake --build build --target ttmlir-opt -j$(nproc)
```

Phase 1b check:

```bash
build/bin/ttmlir-opt --stablehlo-complex-data-type-conversion \
  -o /tmp/complex_gather_out.mlir \
  test/ttmlir/Dialect/StableHLO/ComplexDataTypeConversion/complex_gather.mlir
grep 'complex<f32>' /tmp/complex_gather_out.mlir && echo FAIL || echo OK
```

## Build tt-xla PJRT against that tt-mlir

Use the same tt-mlir pin as a fresh tt-xla clone (`3ac5318` in `third_party/CMakeLists.txt` on `main`) **or** point ExternalProject at your checkout:

```bash
cd tt-xla
source venv/activate
cmake -G Ninja -B build \
  -DTTMLIR_SOURCE_DIR_OVERRIDE=/path/to/tt-mlir-with-akannan/zimage_shlo_bug
cmake --build build -j$(nproc)
```

If PJRT link fails with `createTTNNAllocateDistributedOpBuffers` undefined, do a **clean** full build (see successful flow in a fresh clone) and align `TT_MLIR_VERSION` with `main` (`3ac5318`), then cherry-pick or merge `akannan/zimage_shlo_bug` on top of that tt-mlir tree.

## Pytest (device)

```bash
source venv/activate
python -m pytest -svv tests/torch/models/z_image/test_rope_embedder_op_sanity.py -k gather_complex_polar_table_only
python -m pytest -svv tests/torch/models/z_image/test_rope_repro_standalone.py -k rope_index_axis1
```

After the fix, `compile_fails` tests may report **DID NOT RAISE** (compile succeeded); update test expectations accordingly.
