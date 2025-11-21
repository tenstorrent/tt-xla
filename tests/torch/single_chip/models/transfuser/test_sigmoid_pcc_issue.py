

import torch
from loguru import logger

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    
    logger.info("denom={}",denom)

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
        
def test_sigmoid_pcc():
    
    device_sigmoid = torch.load("device_out_comb_0.pt", map_location="cpu")
    golden_sigmoid = torch.load("golden_out_comb_0.pt", map_location="cpu")

    allclose_result = torch.allclose(device_sigmoid, golden_sigmoid, atol=1e-2, rtol=1e-2) # same values of atol,rtol used in tt-xla
    
    logger.info("allclose_result={}",allclose_result)
    
    logger.info("PCC = {}",compute_pcc(device_sigmoid,golden_sigmoid))
    
    
    logger.info("--------------------------Contents----------------------------------------")
    
    # set print options to show everything
    torch.set_printoptions(
        edgeitems=10000,   # print all items, not truncated
        threshold=1000000, # no summarization
        linewidth=2000     # avoid line wrapping
    )
    
    print("--------------------------Golden----------------------------------------")
    print(golden_sigmoid)
    
    print("--------------------------xla output----------------------------------------")
    print(device_sigmoid)
    
    

    
    
    
    
    
    
        
        
    