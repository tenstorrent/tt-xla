import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
EXAMPLE_PROMPT = """
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """

def init_torch_model(model_name: str, config):
    """
    Initialize the PyTorch model with the given configuration.
    """
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    return torch_model

def prepare_torch_input(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    return tokenizer, inputs.input_ids, inputs.attention_mask

def run_torch_model(torch_model, input_ids, attention_mask):
    """
    Run the PyTorch model with the given input IDs and attention mask.
    """
    print("🏢 Generating HF Model output...")
    outputs = torch_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs

def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    config = AutoConfig.from_pretrained(
        model_name,
        # num_hidden_layers=4, # for easeier testing
        torch_dtype=torch.float32,
    )
    tokenizer, input_ids, attention_mask = prepare_torch_input(model_name, prompt)

    torch_model = init_torch_model(model_name, config)
    torch_output = run_torch_model(torch_model, input_ids, attention_mask)
    
    torch_result = tokenizer.batch_decode(torch_output, skip_special_tokens=False)
    print("🈵 Decoded output:", torch_result[0])

if __name__ == "__main__":
    torch_output = run_test(
        model_name=MODEL_NAME,
        prompt=EXAMPLE_PROMPT
    )
    
    