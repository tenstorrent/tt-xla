# Reference Outputs for LLM Accuracy Testing

This directory contains reference output files (`.refpt`) used for token accuracy testing in tt-xla LLM benchmarks.

**`.refpt` files are generated on-demand** during accuracy tests and are gitignored. Only the source text corpus (`tale-of-two-cities.txt.bz2`) is tracked in git.

## What are .refpt Files?

`.refpt` (reference point) files contain precomputed ground truth data for accuracy testing. Each file stores:

- **reference_tokens**: Full token sequence from "Tale of Two Cities" text corpus
- **top1_tokens**: Top 1 predicted token at each position (argmax from reference model)
- **top5_tokens**: Top 5 predicted tokens at each position (from HuggingFace model on CPU)

## File Format

Each `.refpt` file is a PyTorch serialized dictionary:

```python
{
    'reference_tokens': torch.Tensor,  # Shape: [1, total_length]
    'top1_tokens': torch.Tensor,       # Shape: [total_length-1]
    'top5_tokens': torch.Tensor,       # Shape: [total_length-1, 5]
    'library_versions': {              # For regeneration detection
        'torch': str,                  # e.g., "2.7.0"
        'transformers': str,           # e.g., "4.57.1"
    }
}
```

## On-Demand Generation

When an accuracy test runs, the system automatically:

1. Checks if the `.refpt` file exists for the model being tested
2. If it exists, checks whether `torch` and `transformers` versions match the current environment
3. If the file is missing or versions differ, generates a new `.refpt` on CPU before proceeding

This means:
- **First run** for a model is slower (~30-60s for CPU reference generation)
- **Subsequent runs** reuse the cached `.refpt` file and are fast
- **Library version upgrades** automatically trigger regeneration — no manual intervention needed

The generation logic lives in `llm_utils/reference_generator.py`.

## How They're Used

During accuracy testing:

1. The first half of `reference_tokens` is used as the input prompt (prefill)
2. The second half serves as ground truth for validation
3. **Teacher forcing** is applied: model predictions are stored, but ground truth tokens are fed as input for the next iteration
4. TOP1 and TOP5 accuracy are computed by comparing predictions against reference data

**TOP1 Accuracy**: Percentage where predicted token matches `top1_tokens` (argmax from reference model)
**TOP5 Accuracy**: Percentage where predicted token is in any of the top 5 from `top5_tokens`

## Filename Convention

The filename must match what `TokenAccuracy.get_model_name_from_variant()` returns:

- Extract the model name from HuggingFace path (everything after last `/`)
- Example: `meta-llama/Llama-3.2-1B-Instruct` -> `Llama-3.2-1B-Instruct.refpt`

## Running Accuracy Tests

Accuracy tests are run by passing `--accuracy-testing` to any LLM test:

```bash
# Single model in accuracy mode
pytest -svv tests/benchmark/test_llms.py::test_llama_3_2_1b --accuracy-testing --batch-size 16 --output-file results.json
```

## Source Text

The reference data is generated from "A Tale of Two Cities" by Charles Dickens, stored in `tale-of-two-cities.txt.bz2`. This file is tracked in git.
