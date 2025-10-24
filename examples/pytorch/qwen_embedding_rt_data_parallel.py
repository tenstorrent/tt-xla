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


def run_qwen_on_single_chip(queries, documents, tokenizer, process_id):
    print(f"Process {process_id} Queries: {queries}")
    print(f"Process {process_id} Documents: {documents}")
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B', torch_dtype=torch.bfloat16)
    model.eval()
    model.compile(backend="tt") # Compile for TT device

    device = torch_xla.device(process_id*2)
    
    input_texts = queries + documents
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch_dict = batch_dict.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
    last_hidden_state = outputs.last_hidden_state.to("cpu")
    attention_mask = batch_dict['attention_mask'].to("cpu")
    embeddings = last_token_pool(last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T)
    return (process_id, scores)

def qwen_embedding():
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, "What is the capital of Canada?"),
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "What is the capital of France?"),
        get_detailed_instruct(task, "What is the capital of Germany?"),
    ]

    documents = [
        "The capital of Canada is Ottawa.",
        "The capital of China is Beijing.",
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
    ]

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
    
    results = []
    for i in range(4):
        results.append(run_qwen_on_single_chip([queries[i]], [documents[i]], tokenizer, i))

    for process_id, scores in results:
        print(f"Process {process_id} scores: {scores.tolist()}\n\n")


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen_embedding()