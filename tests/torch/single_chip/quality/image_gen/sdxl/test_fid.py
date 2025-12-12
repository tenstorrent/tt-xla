import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from pipeline import SDXLPipeline, SDXLConfig, save_image
from data import CocoDataset
from metrics import FIDMetric

def test_fid_sdxl():
    xr.set_device_type("TT")

    pipeline = SDXLPipeline(config=SDXLConfig(width=512, height=512))
    pipeline.setup(warmup=True)
    dataset = CocoDataset()

    images = []
    for caption in dataset.captions:
        img = pipeline.generate(caption, seed=42)
        img = img.permute(0, 3, 1, 2) # (B, H, W, 3) -> (B, 3, H, W)
        images.append(img)
    
    images = torch.stack(images, dim=0)

    fid_metric = FIDMetric(dataset.statistics_mean, dataset.statistics_cov)

    fid = fid_metric.compute(images)
    print(f"FID: {fid}")


if __name__ == "__main__":
    test_fid_sdxl()
        
    