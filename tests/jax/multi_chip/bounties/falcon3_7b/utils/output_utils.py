def strip_output(result, prompt: str = "") -> str:
    """
    Strip the output of the model to remove any special tokens or leading text.
    """
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()
    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()
    return result

def compare_results(torch_result: str, flax_result: str) -> str:
    print("HF Result:\n", torch_result)
    print("Flax Result:\n", flax_result)
    if torch_result == flax_result:
        return "✅ Outputs match!"
    else:
        return "❌ Outputs do not match!"
