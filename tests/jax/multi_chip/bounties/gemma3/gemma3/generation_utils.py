from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

import jax
import jax.numpy as jnp
from flax import nnx

from gemma3.base import BaseModel

# TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    # Define a TypeVar for documenting the intended usage context,
    # i.e., classes using this mixin should inherit from BaseModel.
    T_BaseModel = TypeVar("T_BaseModel", bound="BaseModel")
    # Define a base class for type checking
    _Base = BaseModel
else:
    # Runtime doesn't strictly need the bound, but define T anyway
    T_BaseModel = TypeVar("T_BaseModel")
    # Use object at runtime
    _Base = object

# Constants for numerical stability
EPSILON = 1e-9


def temperature_scale(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Scales logits by temperature.

    Args:
        logits: Logits to scale. Shape: (..., vocab_size)
        temperature: Temperature value. Higher values make the distribution flatter (more random),
                     lower values make it peakier (more deterministic). Must be positive.

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    # Prevent division by zero by adding a small epsilon if temperature is zero
    safe_temperature = max(temperature, EPSILON)
    return logits / safe_temperature


def top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Masks logits outside the top k values.

    Sets logits not in the top k to negative infinity.

    Args:
        logits: Logits to filter. Shape: (..., vocab_size)
        k: Number of top logits to keep.

    Returns:
        Filtered logits.
    """
    if k <= 0:
        # If k is 0 or negative, mask all logits
        return jnp.full_like(logits, -jnp.inf)

    # Ensure k is not larger than the vocabulary size
    k = min(k, logits.shape[-1])

    # Get top-k values
    top_k_values = jax.lax.top_k(logits, k=k)[0]
    kth_value = top_k_values[..., -1:]

    # Create a mask where logits >= kth_value are True
    mask = logits >= kth_value

    # Set logits below the threshold to -inf
    return jnp.where(mask, logits, -jnp.inf)


def top_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Shape (..., vocab_size)
        p: Probability threshold (0 < p <= 1)

    Returns:
        Filtered logits with -inf for tokens outside the top-p nucleus
    """
    if not 0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits

    # Convert to probabilities
    probs = nnx.softmax(logits, axis=-1)

    # Sort probabilities in descending order
    sorted_probs = jnp.sort(probs, axis=-1)[..., ::-1]

    # Calculate cumulative probabilities and create mask
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    sorted_mask = cumulative_probs <= p

    # Always include at least the top token
    sorted_mask = sorted_mask.at[..., 0].set(True)

    # Find the minimum probability within the nucleus
    threshold = jnp.min(
        jnp.where(sorted_mask, sorted_probs, jnp.ones_like(sorted_probs)), axis=-1, keepdims=True
    )

    # Apply threshold to original probabilities
    # Keep tokens whose probability is >= threshold
    mask = probs >= threshold

    # Apply mask to logits
    return jnp.where(mask, logits, -jnp.inf)


def min_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """Masks logits below a probability threshold derived from the max probability (min_p sampling).
    Filters out tokens with probability less than p * max_probability.

    Args:
        logits: Logits to filter. Shape: (..., vocab_size)
        p: Probability threshold factor (0 < p <= 1).

    Returns:
        Filtered logits.
    """
    if not 0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    probs = nnx.softmax(logits, axis=-1)
    max_prob = jnp.max(probs, axis=-1, keepdims=True)
    threshold = max_prob * p

    # Identify indices corresponding to max probability
    max_prob_indices = probs >= (max_prob - EPSILON)

    if p == 1.0:
        # When p=1.0, keep just the max probability tokens
        mask = ~max_prob_indices
    else:
        # Otherwise, keep max prob tokens and tokens above the threshold
        mask_below_threshold = probs < threshold
        mask = jnp.where(max_prob_indices, False, mask_below_threshold)

    # Apply the mask to the original logits
    return jnp.where(mask, -jnp.inf, logits)


def sample_logits(
    logits: jnp.ndarray,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    do_sample: bool = True,
) -> jnp.ndarray:
    """Samples a token index from logits using specified filtering and temperature.

    Applies filtering methods (top_k, top_p, min_p) and temperature scaling,
    then samples from the resulting distribution or takes the argmax.

    Args:
        logits: Raw logits from the model. Shape: (..., vocab_size)
        rng_key: JAX PRNG key for sampling.
        temperature: Temperature scaling factor.
        top_k: If set, keep only top k logits.
        top_p: If set, keep smallest set of logits whose cumulative probability exceeds p.
        min_p: If set, keep logits with probability >= max_prob * p.
        do_sample: If True, sample using categorical distribution.
                    If False, take argmax (greedy decoding).

    Returns:
        Sampled token indices. Shape: (...)
    """
    if not do_sample:
        # Greedy decoding
        return jnp.argmax(logits, axis=-1)

    # 1. Apply temperature scaling
    if temperature != 1.0 and temperature > 0:
        scaled_logits = temperature_scale(logits, temperature)
    else:
        scaled_logits = logits

    # Store the scaled logits as the potential fallback
    logits_for_fallback = scaled_logits

    # 2. Apply filtering
    filtered_logits = scaled_logits
    # Apply filtering in a specific order (min_p -> top_k -> top_p is one common order)
    # Note: The order can matter. Min_p focuses on dynamic range,
    # while top_k/top_p on absolute ranks/mass.
    if min_p is not None and 0 < min_p < 1.0:
        filtered_logits = min_p_logits(filtered_logits, min_p)
    if top_k is not None and top_k > 0:
        filtered_logits = top_k_logits(filtered_logits, top_k)
    if top_p is not None and 0 < top_p < 1.0:  # top_p=1 means no filtering
        filtered_logits = top_p_logits(filtered_logits, top_p)

    # 3. Sample or take argmax, handling the edge case for sampling

    all_filtered_infinite = jnp.all(filtered_logits == -jnp.inf, axis=-1, keepdims=True)

    # Determine the logits to actually sample from:
    # Use the fallback (scaled, unfiltered) if all filtered are -inf
    final_logits_for_sampling = jnp.where(
        all_filtered_infinite,
        logits_for_fallback,  # Fallback to pre-filter (but post-temp) logits
        filtered_logits,  # Otherwise, use the filtered logits
    )

    # Sample using the chosen logits
    sampled_indices = jax.random.categorical(rng_key, final_logits_for_sampling, axis=-1)

    return sampled_indices


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Creates a causal attention mask for a given sequence length.

    Args:
        seq_len: The length of the sequence.

    Returns:
        A causal attention mask of shape [seq_len, seq_len].
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask


class GenerationMixin(_Base):
    """Mixin that adds text generation capabilities,
    including sampling with temperature, top-k, top-p,
    and min-probability filtering, for CausalLMs."""

    def _generate_scan_logic(
        self: "GenerationMixin",  # self is needed for model call
        initial_input_ids: jnp.ndarray,
        initial_finished_sequences: jnp.ndarray,
        initial_rng: jax.Array,
        initial_seq_len: int,
        # --- Static arguments (used by JIT, must be passed regardless) ---
        max_length: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        min_p: float | None,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int | None,
    ) -> jnp.ndarray:
        """The core autoregressive generation logic using lax.scan.
        This function itself is NOT jitted here."""

        batch_size = initial_input_ids.shape[0]
        output_ids = jnp.full((batch_size, max_length), pad_token_id, dtype=initial_input_ids.dtype)
        output_ids = output_ids.at[:, :initial_seq_len].set(initial_input_ids)

        def scan_step(carry: dict, _: Any) -> tuple[dict, None]:
            current_output_ids = carry["output_ids"]
            current_length = carry["current_length"]
            step_rng = carry["rng"]
            current_finished = carry["finished"]

            next_rng = step_rng
            sampling_rng = step_rng
            if do_sample:  # Relies on do_sample being static *when jitted*
                sampling_rng, next_rng = jax.random.split(step_rng)

            # This mask tells the model which tokens are valid.
            attention_mask = (jnp.arange(max_length) < current_length).astype(jnp.int32)[None, :]

            # Call the model, passing the attention mask
            # Assume the model returns a dictionary with 'logits'
            # Also assume the model should run deterministically during generation
            logits = self(  # type: ignore[operator]
                input_ids=current_output_ids,
                attention_mask=attention_mask,
                deterministic=True,  # Generation should be deterministic (no dropout)
            )

            # Get logits for the *next* token prediction (at index current_length - 1)
            next_token_logits = jax.lax.dynamic_slice_in_dim(
                logits, current_length - 1, 1, axis=1
            ).squeeze(axis=1)

            # Sample the next token
            next_token = sample_logits(
                logits=next_token_logits,
                rng_key=sampling_rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                do_sample=do_sample,
            )
            next_token = next_token.astype(current_output_ids.dtype)

            # Determine the token to actually write (handling EOS and padding)
            output_token = next_token  # Default
            next_finished = current_finished
            if eos_token_id is not None:  # Relies on eos_token_id being static *when jitted*
                newly_finished = (next_token == eos_token_id) & (~current_finished)
                next_finished = current_finished | newly_finished
                # If already finished, write pad_token_id, otherwise write the sampled token
                output_token = jnp.where(current_finished, pad_token_id, next_token)

            # Update the output sequence
            updated_output_ids = jax.lax.dynamic_update_slice_in_dim(
                current_output_ids, output_token[:, None], current_length, axis=1
            )

            # Prepare carry for the next step
            next_carry = {
                "output_ids": updated_output_ids,
                "current_length": current_length + 1,
                "rng": next_rng,
                "finished": next_finished,
            }
            return next_carry, None

        initial_carry = {
            "output_ids": output_ids,
            "current_length": jnp.array(initial_seq_len),
            "rng": initial_rng,
            "finished": initial_finished_sequences,
        }
        num_steps_to_generate = max_length - initial_seq_len

        # Run the scan only if needed
        if num_steps_to_generate > 0:
            final_carry, _ = jax.lax.scan(
                scan_step, initial_carry, None, length=num_steps_to_generate
            )
            final_output_ids = final_carry["output_ids"]
        else:
            # If no steps needed (initial_seq_len == max_length), return initial output_ids
            final_output_ids = output_ids

        return cast(jnp.ndarray, final_output_ids)

    # Define the compiled version of the scan logic
    # This uses partial to pre-apply jax.jit with static arguments
    # Note: Compiling happens when this method definition is executed.
    _generate_compiled = partial(
        jax.jit,
        # Specify arguments that control the computation graph structure
        static_argnames=(
            "self",  # Need self for model call inside scan_step
            "max_length",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "do_sample",
            "pad_token_id",
            "eos_token_id",
            "initial_seq_len",
        ),
    )(_generate_scan_logic)  # Apply JIT to the core logic function

    def generate(
        self: "GenerationMixin",
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        do_sample: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        rng: jax.Array | None = None,
        use_jit: bool = False,
    ) -> jnp.ndarray:
        """Generate tokens autoregressively with various sampling methods.

        Args:
            input_ids: Initial token IDs of shape [batch_size, seq_len].
            attention_mask: Optional attention mask
            max_length: Maximum length for generated sequences.
            temperature: Temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
            top_p: If specified, only sample from the smallest set of logits
                  whose cumulative probability exceeds p.
            min_p: If specified, only consider logits with prob >= min_p * prob(max logit).
            do_sample: If True, use sampling; otherwise use greedy/beam search.
            pad_token_id: Token ID to use for padding.
            eos_token_id: Token ID that signals the end of generation.
            rng: Optional PRNG key for sampling.
            use_jit: If True, use jax.jit to compile the generate function.

        Returns:
            Generated token IDs of shape [batch_size, max_length].
        """

        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}")
        if not isinstance(temperature, (float, int)) or temperature <= 0:  # noqa: UP038
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if top_p is not None and (not isinstance(top_p, float) or not 0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        _top_p = top_p if top_p != 1.0 else None  # Handle p=1.0 case internally
        if min_p is not None and (not isinstance(min_p, float) or not 0 < min_p <= 1.0):
            raise ValueError(f"min_p must be in (0, 1], got {min_p}")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}"
            )

        # Handle RNG key
        _rng = rng
        if do_sample:
            if _rng is None:
                print("Warning: No RNG key provided for sampling, using default key 0.")
                _rng = jax.random.PRNGKey(0)  # Use seed 0
            # Ensure rng is a JAX key
            if isinstance(_rng, int):
                _rng = jax.random.PRNGKey(_rng)
            elif not isinstance(_rng, jax.Array):
                raise ValueError(f"Invalid rng provided: {_rng}. Expected JAX PRNGKey or seed.")
        elif _rng is None:  # Provide a dummy key if not sampling and None was passed
            _rng = jax.random.PRNGKey(0)
        elif isinstance(_rng, int):  # Ensure key even if not sampling but seed provided
            _rng = jax.random.PRNGKey(_rng)

        # Resolve pad_token_id
        _pad_token_id = pad_token_id
        if _pad_token_id is None:
            # Safely access config attribute
            config = getattr(self, "config", None)
            _pad_token_id = getattr(config, "pad_token_id", 0) if config else 0
        if not isinstance(_pad_token_id, int):
            raise ValueError(f"pad_token_id must be an integer, got {_pad_token_id}")

        # Resolve eos_token_id
        _eos_token_id = eos_token_id
        if _eos_token_id is not None and not isinstance(_eos_token_id, int):
            raise ValueError(f"eos_token_id must be an integer or None, got {_eos_token_id}")

        # Get initial sequence length and batch size
        batch_size, initial_seq_len = input_ids.shape

        # Handle cases where input is already long enough
        if initial_seq_len >= max_length:
            print(f"""Warning: Initial sequence length ({initial_seq_len}) \
                is >= max_length ({max_length}). \
                Returning truncated input.""")
            return input_ids[:, :max_length]

        # Track whether each sequence is finished
        finished_sequences = jnp.zeros((batch_size,), dtype=jnp.bool_)

        if _eos_token_id is not None:
            # Check if the *last* token of the input is EOS
            finished_sequences = jnp.where(
                initial_seq_len > 0,
                input_ids[:, -1] == _eos_token_id,
                jnp.zeros_like(finished_sequences),  # Ensure correct shape if seq_len is 0
            )

        # --- Conditionally Call Jitted or Non-Jitted Core Logic ---
        # common_args = {
        #     "initial_input_ids": input_ids,
        #     "initial_finished_sequences": finished_sequences,
        #     "initial_rng": _rng,
        #     "initial_seq_len": initial_seq_len, # Pass as static arg for JIT
        #     "max_length": max_length,
        #     "temperature": float(temperature), # Ensure float
        #     "top_k": top_k,
        #     "top_p": _top_p, # Use resolved top_p
        #     "min_p": min_p,
        #     "do_sample": do_sample,
        #     "pad_token_id": _pad_token_id, # Use resolved pad id
        #     "eos_token_id": _eos_token_id, # Use resolved eos id
        # }

        if use_jit:
            # Call the pre-compiled method
            final_output_ids = self._generate_compiled(
                initial_input_ids=input_ids,
                initial_finished_sequences=finished_sequences,
                initial_rng=_rng,
                initial_seq_len=initial_seq_len,
                max_length=max_length,
                temperature=float(temperature),
                top_k=top_k,
                top_p=_top_p,
                min_p=min_p,
                do_sample=do_sample,
                pad_token_id=_pad_token_id,
                eos_token_id=_eos_token_id,
            )
        else:
            # Call the raw logic method directly
            final_output_ids = self._generate_scan_logic(
                initial_input_ids=input_ids,
                initial_finished_sequences=finished_sequences,
                initial_rng=_rng,
                initial_seq_len=initial_seq_len,
                max_length=max_length,
                temperature=float(temperature),
                top_k=top_k,
                top_p=_top_p,
                min_p=min_p,
                do_sample=do_sample,
                pad_token_id=_pad_token_id,
                eos_token_id=_eos_token_id,
            )

        return cast(jnp.ndarray, final_output_ids)
