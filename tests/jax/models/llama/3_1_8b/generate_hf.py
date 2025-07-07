import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import fire

def hf_load(model_id: str = "meta-llama/Meta-Llama-3.1-8B"):
    print("🔧 Loading Hugging Face model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    # Print all special tokens and their IDs
    for token_name in tokenizer.special_tokens_map:
        token_str = tokenizer.special_tokens_map[token_name]
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"{token_name}: {token_str} -> {token_id}")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    return tokenizer, model

def main(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B",
    prompt: str = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n"
    "F: #### 18 \n\n"
    "Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\n"
    "A: The cost of the house and repairs came out to 80,000 + 50,000 = $<<80000+50000=130000>>130,000\n"
    "He increased the value of the house by 80,000 * 1.5 = <<80000*1.5=120000>>120,000\n"
    "So the new value of the house is 120,000 + 80,000 = $<<120000+80000=200000>>200,000\n"
    "So he made a profit of 200,000 - 130,000 = $<<200000-130000=70000>>70,000\n"
    "F: #### 70000\n\n"
    "Q: A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars.  If the rink has 75 cars in total how many yellow cars do they have?\n"
    "A:"
),
    max_gen_len: int = 128,
    temperature: float = 0.01,
    top_p: float = 0.99
):
    
    #example prompts
    #In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?
    #A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
    #A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars.  If the rink has 75 cars in total how many yellow cars do they have?
    
    #answers
    # 60
    # 3
    # 23
    
    tokenizer, model = hf_load(model_id)
    print(tokenizer.eos_token, tokenizer.pad_token)
    print(tokenizer.special_tokens_map)

    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    print("eos_token_id in input_ids:", tokenizer.eos_token_id in input_ids[0])

    print("✍️ Generating...")
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_gen_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(input_ids,attention_mask=attention_mask, generation_config=generation_config)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()

    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()

    print("\n🧠 Output:\n", result) 

if __name__ == "__main__":
    fire.Fire(main)
