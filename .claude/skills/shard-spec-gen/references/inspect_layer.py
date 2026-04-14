from third_party.tt_forge_models.<model>.<task>.pytorch.loader import ModelLoader, ModelVariant
import torch

loader = ModelLoader(variant=ModelVariant.<VARIANT>)
config = loader.load_config()

# attention/MLP: instantiate the layer directly — fast, no full model load
from transformers.models.<arch>.modeling_<arch> import <LayerClass>
layer = <LayerClass>(config).to(torch.bfloat16)
print("attrs:", [n for n, _ in layer.named_parameters()])
print("hidden_size:", config.hidden_size)
print("num_attention_heads:", getattr(config, "num_attention_heads", "N/A"))
print("num_key_value_heads:", getattr(config, "num_key_value_heads", "N/A"))
print("intermediate_size:", getattr(config, "intermediate_size", "N/A"))

# MoE: load one transformer layer only
# config.num_hidden_layers = 1  # or: ModelLoader(num_layers=1)
# model = loader.load_model(dtype_override=torch.bfloat16)
# mlp = model.model.layers[0].mlp
# print(type(mlp), [n for n, _ in mlp.named_parameters()][:20])
# print("has shared_expert:", hasattr(mlp, "shared_expert"))
# print("num experts:", len(mlp.experts) if hasattr(mlp, "experts") else 0)
