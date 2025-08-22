#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real LoRA training on GPT-2 with TinyStories dataset.
Fun, engaging stories instead of boring Wikipedia!
Focuses on MLP layers only as originally requested.
"""
import warnings
import jax
import jax.numpy as jnp
import optax
import numpy as np
from transformers import FlaxGPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

import lorax
from lorax import LORA_FULL, LORA_FREEZE


def load_tinystories_data(tokenizer, max_length=128, num_samples=1000):
    """Load and tokenize TinyStories data - much more engaging than Wikipedia!"""
    print("📚 Loading TinyStories dataset...")

    # Load dataset - these are simple, fun stories!
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Get story texts
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 100]
    texts = texts[:num_samples]  # Limit for demo

    print(f"📊 Loaded {len(texts)} story samples")
    print(f"📝 Sample story: {texts[0][:300]}...")

    # Tokenize
    print("🔤 Tokenizing stories...")
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )

    input_ids = tokenized["input_ids"]
    print(f"✅ Tokenized data shape: {input_ids.shape}")

    return input_ids


def create_batches(data, batch_size=8):
    """Create training batches."""
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def main():
    print("🚀 Starting GPT-2 LoRA Training on TinyStories")

    # Load model and tokenizer
    print("🤖 Loading GPT-2 model...")
    model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load real story data
    text_data = load_tinystories_data(tokenizer, max_length=128, num_samples=500)
    train_batches = create_batches(text_data, batch_size=4)
    print(f"📦 Created {len(train_batches)} training batches")

    # LoRA spec: ONLY MLP layers (as originally requested!)
    def decision_fn(path, param):
        path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)

        # Only apply LoRA to MLP layers
        if ".mlp." in path_str and (
            "c_fc.kernel" in path_str or "c_proj.kernel" in path_str
        ):
            rank = 16  # LoRA rank
            print(f"✅ Applying LoRA rank {rank} to MLP layer: {path_str}")
            return rank
        else:
            # Freeze everything else (attention, embeddings, etc.)
            return LORA_FREEZE

    # Create LoRA spec and parameters
    print("⚙️ Setting up LoRA configuration...")
    lora_spec = lorax.simple_spec(
        model.params, decision_fn=decision_fn, tune_vectors=False
    )
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(42))

    # Setup optimizer with proper parameter freezing
    optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.01)
    optimizer = lorax.wrap_optimizer(optimizer, lora_spec)
    opt_state = optimizer.init(lora_params)

    # Wrapped model
    lora_model = lorax.lora(model)

    # Training loss function (next-token prediction)
    def loss_fn(lora_params, batch):
        input_ids = batch[:, :-1]  # Input: all tokens except last
        targets = batch[:, 1:]  # Target: all tokens except first

        # Forward pass
        logits = lora_model(input_ids, params=lora_params).logits

        # Cross-entropy loss
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        target_logprobs = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)

        # Mask padding tokens (assuming tokenizer.pad_token_id is the pad token)
        pad_mask = (targets != tokenizer.pad_token_id).astype(jnp.float32)
        loss = -jnp.sum(target_logprobs.squeeze(-1) * pad_mask) / jnp.sum(pad_mask)

        return loss

    # JIT compiled training step
    @jax.jit
    def train_step(lora_params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(lora_params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, lora_params)
        new_params = optax.apply_updates(lora_params, updates)
        return new_params, new_opt_state, loss

    # Training loop
    print("🎯 Starting training on real text data...")
    for epoch in range(10):  # Train for 3 epochs
        epoch_losses = []

        for batch_idx, batch in enumerate(
            train_batches[:20]
        ):  # Train on first 20 batches
            lora_params, opt_state, loss = train_step(lora_params, opt_state, batch)
            epoch_losses.append(float(loss))

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx:2d}: Loss = {loss:.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"📊 Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Test generation
    print("\n🔮 Testing story generation...")
    test_prompt = "Once upon a time, there was a little"
    test_tokens = tokenizer.encode(test_prompt, return_tensors="np")

    # Generate a few tokens
    generated_logits = lora_model(test_tokens, params=lora_params).logits
    next_token_id = jnp.argmax(generated_logits[0, -1])
    next_token = tokenizer.decode([next_token_id])

    print(f"📝 Input: '{test_prompt}'")
    print(f"🎯 Next predicted token: '{next_token}'")

    # Merge LoRA back into regular weights
    print("\n🔄 Merging LoRA parameters...")
    merged_params = lorax.merge_params(lora_params)

    # Verify merge works
    orig_logits = model(test_tokens, params=merged_params).logits
    lora_logits = lora_model(test_tokens, params=lora_params).logits
    merge_error = jnp.max(jnp.abs(orig_logits - lora_logits))

    print(f"✅ LoRA merge verification - Max error: {merge_error:.2e}")
    print("🎉 LoRA training completed successfully!")


if __name__ == "__main__":
    main()
