from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LLaMAConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
    self,
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_sequence_length=2048,
    rms_norm_eps=1e-5,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=-1,
    bos_token_id=128000,
    eos_token_id=128001,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    tie_word_embeddings=False,
    gradient_checkpointing: bool = False,
    rope_theta: float = 500000.0,
    rope_scaling=None,
    **kwargs,
):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.max_position_embeddings = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.gradient_checkpointing = gradient_checkpointing
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

