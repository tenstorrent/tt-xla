# tt-mlir fix for Z-Image RoPE complex gather (#4756)

PyTorch tests on tt-xla branch `akannan/zimage_shlo_bug` expect **tt-mlir** on branch
`akannan/zimage_shlo_bug`. Without that branch, RoPE gather/index tests fail TT compile
with **Error code 13** (`complex<f32>` legalization).

## Checkout tt-mlir fix

```bash
cd tt-xla/third_party/tt-mlir/src/tt-mlir
git fetch origin akannan/zimage_shlo_bug
git checkout akannan/zimage_shlo_bug
```

## Build tt-mlir + PJRT

```bash
cd tt-mlir
source env/activate
cmake -G Ninja -B build
cmake --build build --target TTMLIRCompiler -j$(nproc)
cmake --install build --component SharedLib

cd ../../..   # tt-xla root
source venv/activate
cmake -G Ninja -B build
cmake --build build -j$(nproc)
python -m pip install -e python_package
```

Do **not** use `TTMLIR_SOURCE_DIR_OVERRIDE` when tt-mlir already lives at
`third_party/tt-mlir/src/tt-mlir`.

## Pytest

```bash
python -m pytest -svv tests/torch/models/z_image/test_rope_embedder_op_sanity.py
python -m pytest -svv tests/torch/models/z_image/test_rope_repro_standalone.py
python -m pytest -svv tests/torch/models/z_image/test_transformer_slice.py -k single_chip
```

Full `test_transformer.py` remains `xfail` until the end-to-end model passes.
