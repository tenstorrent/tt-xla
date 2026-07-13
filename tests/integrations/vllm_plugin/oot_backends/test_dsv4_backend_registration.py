# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Registration test for the DeepSeek-V4 attention backend.

This verifies the OOT *registration wiring* — that importing the plugin registers
``TTDeepseekV4AttentionBackend`` in vLLM's attention-backend registry and that
vLLM can resolve it back to our class with the expected ``AttentionBackend``
interface. It is a device-free unit test; attention numerics are covered
separately by ``test_dsv4_swa_attention_impl.py``.

DSV4 is a sparse-MLA model, so the plugin overrides vLLM's sparse-MLA enum slot
(``AttentionBackendEnum.FLASHMLA_SPARSE``) with our backend — mirroring how the
dense MLA backend overrides ``FLASH_ATTN_MLA`` (see ``vllm_tt/__init__.py``).
"""
import pytest

# Importing the plugin runs the module-level register_backend() calls. vLLM also
# auto-loads it as a platform plugin, so this is idempotent.
import vllm_tt  # noqa: F401,E402
from vllm.v1.attention.backend import AttentionBackend, MLAAttentionImpl  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm_tt.attention_impls.attention import TTAttentionBackend  # noqa: E402
from vllm_tt.attention_impls.attention_dsv4 import (  # noqa: E402
    TTDeepseekV4AttentionBackend,
    TTDeepseekV4AttentionBackendImpl,
)
from vllm_tt.attention_impls.attention_mla import TTMLAAttentionBackend  # noqa: E402

_DSV4_BACKEND_PATH = (
    "vllm_tt.attention_impls.attention_dsv4.TTDeepseekV4AttentionBackend"
)


@pytest.mark.push
def test_dsv4_backend_registered_with_vllm():
    """The DSV4 attention backend is registered and resolves to our class."""
    backend_enum = AttentionBackendEnum.FLASHMLA_SPARSE

    # 1. The plugin overrode the sparse-MLA enum slot.
    assert backend_enum.is_overridden(), (
        "DSV4 backend was not registered — vllm_tt/__init__.py must call "
        "register_backend(AttentionBackendEnum.FLASHMLA_SPARSE, ...)."
    )
    assert backend_enum.get_path() == _DSV4_BACKEND_PATH

    # 2. vLLM resolves the registered class path to our backend class.
    resolved = backend_enum.get_class()
    assert resolved is TTDeepseekV4AttentionBackend
    assert issubclass(resolved, AttentionBackend)


@pytest.mark.push
def test_dsv4_backend_interface():
    """The registered backend exposes the AttentionBackend interface vLLM uses."""
    backend = AttentionBackendEnum.FLASHMLA_SPARSE.get_class()

    assert backend.get_name() == "DEEPSEEK_V4"

    impl_cls = backend.get_impl_cls()
    assert impl_cls is TTDeepseekV4AttentionBackendImpl
    assert issubclass(impl_cls, MLAAttentionImpl)

    # A metadata builder must be resolvable (reused from the non-MLA path).
    assert backend.get_builder_cls() is not None

    # DSV4 SWA cache uses a 64-token block (sparse_swa.py:74).
    assert backend.get_page_size(vllm_config=None) == 64

    # MLA latent KV-cache: (num_blocks, num_kv_heads == 1, block_size,
    # head_size = kv_lora_rank + qk_rope_head_dim). Mirrors the V3 MLA shape.
    assert backend.get_kv_cache_shape(
        num_blocks=4, block_size=64, num_kv_heads=1, head_size=576
    ) == (4, 1, 64, 576)

    # MLA requires a single latent KV head.
    with pytest.raises(AssertionError):
        backend.get_kv_cache_shape(
            num_blocks=4, block_size=64, num_kv_heads=2, head_size=576
        )


@pytest.mark.push
def test_dsv4_registration_does_not_clobber_existing_backends():
    """Registering DSV4 must not disturb the existing TT backend registrations."""
    assert AttentionBackendEnum.FLASH_ATTN_MLA.get_class() is TTMLAAttentionBackend
    assert AttentionBackendEnum.CUSTOM.get_class() is TTAttentionBackend
    # The three TT backends are distinct classes.
    assert (
        len(
            {
                AttentionBackendEnum.FLASHMLA_SPARSE.get_class(),
                AttentionBackendEnum.FLASH_ATTN_MLA.get_class(),
                AttentionBackendEnum.CUSTOM.get_class(),
            }
        )
        == 3
    )
