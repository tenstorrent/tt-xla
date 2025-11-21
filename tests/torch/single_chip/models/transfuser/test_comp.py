
import torch 
from loguru import logger

def test_comp():
    
    
    a = torch.load("device_out_comb_0.pt",map_location="cpu")
    b = torch.load("device_out_sig.pt",map_location="cpu")
    
    logger.info("all close?={}",torch.allclose(a,b))
    
    c = torch.load("device_out_comb_1.pt",map_location="cpu")
    d = torch.load("device_out_conv.pt",map_location="cpu")
    
    logger.info("all close?={}",torch.allclose(c,d))
    
     
    e = torch.load("golden_out_comb_0.pt",map_location="cpu")
    f = torch.load("golden_out_sig.pt",map_location="cpu")
    
    logger.info("all close?={}",torch.allclose(e,f))
    
    g = torch.load("golden_out_comb_1.pt",map_location="cpu")
    h = torch.load("golden_out_conv.pt",map_location="cpu")
    
    logger.info("all close?={}",torch.allclose(g,h))
    
    
    logger.info("a={}",a)
    logger.info("b={}",b)
    
    logger.info("c={}",c)
    logger.info("d={}",d)
    
    logger.info("e={}",e)
    logger.info("f={}",f)
    logger.info("g={}",g)
    logger.info("h={}",h)

    