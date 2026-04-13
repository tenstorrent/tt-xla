# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeBERTa-v3-base Sentence Similarity demo on Tenstorrent hardware.

Encodes sentences using DeBERTa-v3-base on a TT device, computes cosine
similarity between them, and validates against CPU reference output.
"""

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import tt_torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "microsoft/deberta-v3-base"
SEQ_LEN = 128  # padded to multiple of 32 for tile alignment

SENTENCES = [
    "A dog is playing fetch in the park with its owner.",
    "A puppy is running around outside chasing a ball.",
    "The quarterly earnings report exceeded analyst expectations.",
    "She enjoys reading books about ancient history.",
    "He likes to learn about events from the distant past.",
]


def mean_pool(hidden_states, attention_mask):
    """Mean pooling over token embeddings, masked by attention_mask."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode(model, input_ids, attention_mask):
    """Encode a batch of sentences into normalized embeddings."""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    embeddings = mean_pool(outputs[0], attention_mask)
    return F.normalize(embeddings, p=2, dim=1)


def print_similarity_matrix(labels, sim_matrix):
    """Pretty-print a similarity matrix."""
    # Header
    col_w = 8
    label_w = 45
    header = " " * label_w + "".join(f"[{i}]{' ' * (col_w - len(str(i)) - 2)}" for i in range(len(labels)))
    print(header)

    for i, label in enumerate(labels):
        short = label[:label_w - 4] + "..." if len(label) > label_w - 1 else label
        row = f"{short:<{label_w}}"
        for j in range(len(labels)):
            row += f"{sim_matrix[i][j]:>{col_w}.3f}"
        print(row)


def main():
    xr.set_device_type("TT")
    device = xm.xla_device()

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.config.return_dict = False
    model.eval()

    # Tokenize all sentences as a batch
    inputs = tokenizer(
        SENTENCES, return_tensors="pt", padding="max_length",
        max_length=SEQ_LEN, truncation=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # --- TT device ---
    print(f"Running on device: {device}")
    compiled_model = torch.compile(model, backend="tt")
    compiled_model = compiled_model.to(device)

    tt_embeddings = encode(
        compiled_model,
        input_ids.to(device),
        attention_mask.to(device),
    ).cpu()

    # --- CPU reference ---
    model_cpu = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model_cpu.config.return_dict = False
    model_cpu.eval()
    cpu_embeddings = encode(model_cpu, input_ids, attention_mask)

    # --- Similarity matrices ---
    tt_sim = (tt_embeddings @ tt_embeddings.T).float()
    cpu_sim = (cpu_embeddings @ cpu_embeddings.T).float()

    print("\n--- TT Device Similarity Matrix ---")
    print_similarity_matrix(SENTENCES, tt_sim)

    print("\n--- CPU Reference Similarity Matrix ---")
    print_similarity_matrix(SENTENCES, cpu_sim)

    # --- Validation ---
    pcc = torch.corrcoef(torch.stack([
        tt_embeddings.flatten().float(),
        cpu_embeddings.flatten().float(),
    ]))[0, 1].item()
    max_diff = (tt_sim - cpu_sim).abs().max().item()

    print(f"\nEmbedding PCC: {pcc:.6f}  {'PASS' if pcc > 0.99 else 'WARN'}")
    print(f"Max similarity diff: {max_diff:.4f}")

    # Spot-check: semantically related pairs should rank higher than unrelated ones
    dog_pair = tt_sim[0][1].item()       # dog <-> puppy (related)
    history_pair = tt_sim[3][4].item()   # history <-> past events (related)
    dog_earnings = tt_sim[0][2].item()   # dog <-> earnings (unrelated)
    history_earnings = tt_sim[3][2].item()  # history <-> earnings (unrelated)
    print(f"\nSanity checks (related pairs should score higher than unrelated):")
    print(f"  'dog' <-> 'puppy':       {dog_pair:.3f}  (related)")
    print(f"  'history' <-> 'events':  {history_pair:.3f}  (related)")
    print(f"  'dog' <-> 'earnings':    {dog_earnings:.3f}  (unrelated)")
    print(f"  'history' <-> 'earnings': {history_earnings:.3f}  (unrelated)")

    if dog_pair > dog_earnings and history_pair > history_earnings:
        print("  All sanity checks PASSED")
    else:
        print("  WARN: unexpected similarity ranking")

    print("\nDone.")


if __name__ == "__main__":
    main()
