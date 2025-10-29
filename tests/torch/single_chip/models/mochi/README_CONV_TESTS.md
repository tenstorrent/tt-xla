# Mochi Decoder Conv3D Test Configuration

This directory contains tools and data for testing Conv3D operations from the Mochi decoder model.

## Files

- **`extract_convolutions.py`**: Script to extract convolution operations from TTIR files
- **`locations.json`**: Unique convolution configurations extracted from `mochi_decoder.ttir`
- **`generate_test_params.py`**: Script to generate pytest parameters from `locations.json`
- **`mochi_decoder.ttir`**: TTIR file of the Mochi decoder model

## Workflow

### 1. Extract Convolutions from TTIR

```bash
python3 extract_convolutions.py mochi_decoder.ttir --unique --save-json
```

This will:
- Extract all convolution operations from the TTIR file
- Deduplicate based on tensor shapes, attributes, and layout
- Save unique configurations to `locations.json`
- Track all occurrences of each unique convolution

**Result**: Found 8 unique convolutions from 57 total operations

### 2. Generate Test Parameters

```bash
python3 generate_test_params.py > test_params_output.txt
```

This generates pytest parametrize format for the test configurations.

### 3. Update Test File

The test file `/localdev/vkovinic/tt-metal/tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`
has been updated with the 8 unique Mochi decoder convolution configurations in `test_conv3d_mochi_shapes`.

## Unique Convolution Configurations

| ID | Input Shape | Out Channels | Kernel | Stride | Padding | Occurrences |
|----|------------|--------------|--------|--------|---------|-------------|
| conv_in | (1, 12, 7, 60, 106) | 768 | (1,1,1) | (1,1,1) | (0,0,0) | 1 |
| conv_768 | (1, 768, 9, 62, 108) | 768 | (3,3,3) | (1,1,1) | (0,0,0) | 18 |
| conv_512 | (1, 512, 23, 122, 214) | 512 | (3,3,3) | (1,1,1) | (0,0,0) | 8 |
| conv_256_t22 | (1, 256, 22, 242, 426) | 256 | (3,3,3) | (1,1,1) | (0,0,0) | 6 |
| conv_256_t24 | (1, 256, 24, 242, 426) | 256 | (3,3,3) | (1,1,1) | (0,0,0) | 6 |
| conv_128_t15 | (1, 128, 15, 482, 850) | 128 | (3,3,3) | (1,1,1) | (0,0,0) | 6 |
| conv_128_t17 | (1, 128, 17, 482, 850) | 128 | (3,3,3) | (1,1,1) | (0,0,0) | 6 |
| conv_128_t16 | (1, 128, 16, 482, 850) | 128 | (3,3,3) | (1,1,1) | (0,0,0) | 6 |

**Total convolutions in model**: 57 (these 8 unique configs account for all of them)

## Running Tests

Currently all tests will skip because blocking parameters need to be determined:

```bash
pytest -v /localdev/vkovinic/tt-metal/tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py::test_conv3d_mochi_shapes
```

### Determining Optimal Blocking

To find optimal blocking parameters for each configuration:

1. Use `test_conv3d_sweep_blocks` as a reference for block size sweeping
2. Update blocking parameters in the test file for each configuration
3. Blocking format: `(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)`

Example blocking configurations from similar shapes:
- `(128, 96, 1, 2, 16)` for 768 channels
- `(128, 128, 1, 8, 4)` for 512 channels
- `(128, 128, 1, 2, 16)` for 128 channels

## Notes

- All convolutions use `padding_mode="zeros"` with `padding=(0,0,0)`
- All convolutions use `stride=(1,1,1)`
- Most convolutions are 3x3x3 kernels, except conv_in which is 1x1x1
- The most common configuration (conv_768) appears 18 times in the model
- Padding is handled separately before the convolution in the TTIR (via `pad(functional.py:5209)`)

## TODO

- [ ] Determine optimal blocking parameters for each configuration
- [ ] Run sweep tests to find best blocking for each shape
- [ ] Update test file with determined blocking parameters
- [ ] Run full test suite and verify PCC values
