import torch

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

tt_output = torch.load("DeepSeek_params/tt_final_output.pt")
gpu_output = torch.load("DeepSeek_params/final_output.pt")

pcc = compute_pcc(tt_output, gpu_output)
print(f"PCC between TT-XLA and GPU outputs: {pcc}")