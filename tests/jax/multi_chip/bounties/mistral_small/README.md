mistral_nnx is a Flax NNX implementation of Mistral models.

Copied from https://github.com/yiding/mistral-nnx

## Support

The model has been tested with mistral-small and ministral-8b. Note that
Ministral-8b uses sliding window, which is not supported, so long sequences may
not behave correctly.

### Supported Features

Attention: Grouped Query Attention (GQA), but implementation should also support MHA.

Sliding window: not supported.

KV Cache: Very rudimentary KV Cache, supports batch-size 1.

## Usage

### Config

The model uses configuration from the model repo from Huggingface Hub.

### Weights

Model supports loading the safetensor weights from huggingface hub. Weights
loaded this way need to be transposed or reshaped before usage. There's
additional code to save the transformed weights into an orbax checkpoint, which
loads much faster.

### Inference

MistralModel supports basic forward pass using `MistralModel.__call__()`, which
takes an array of a batch of tokens `(batch_size, seq_len)`.

It also supports KV cache with batch-size 1 using `MistralModel.decode()`.
`generate.py` contains code to setup and run inference with kv cache.

### Parallelism

The model parameter initializers has been annotated with Flax [local axis
annotations](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#logical-axis-annotation).

The parameter loading functions `MistralModel.load` and
`MistralModel.load_from_hf_pt_model` can be passed a `mesh` and `sharding_rules`
mapping that maps the logical axis names to mesh axis names, this annotates the
parameter arrays with jax sharding constraints and ensure the params are loaded
to the correct devices.

An example usage is provided in `test_model_implementation.py`.


### JIT

`nnx.jit` has high CPU overhead, and has been observed to sometimes have high
memory overhead as well. A more performant alternative is to use `jax.jit` with
`nnx.split`.

Examples of this method are provided in `generate.py` and
`test_model_implementation.py`.
