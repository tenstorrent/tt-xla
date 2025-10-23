import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor


# Adapted from: https://huggingface.co/Qwen/Qwen3-Embedding-4B#transformers-usage
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def qwen_embedding():
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity."),
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B', torch_dtype=torch.bfloat16)
    model.eval()
    model.compile(backend="tt") # Compile for TT device

    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )

    # Move inputs and model to device
    # Note: the 0 refers to the first TT device in the system. It is optional.
    # If omitted, the first TT device in the system will be used by default.
    device = torch_xla.device(0)
    batch_dict = batch_dict.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        # Move outputs back to host
        last_hidden_state = outputs.last_hidden_state.to("cpu")
        attention_mask = batch_dict['attention_mask'].to("cpu")
        embeddings = last_token_pool(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T)
        print(scores.tolist())


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen_embedding()