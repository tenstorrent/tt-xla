import pandas as pd
import numpy as np
import urllib.request
from io import BytesIO
from typing import List
    
class CocoDataset:
    """
    Dataset class for the COCO 2014 dataset. This is a common dataset used for evaluation of image generation models.
    A nice perk is that it comes with a set of captions and FID data statistics that can be used for evaluation.
    """
    N_SUBSET_SAMPLES = 10
    COCO_CAPTIONS_URL = "https://raw.githubusercontent.com/mlcommons/inference/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
    COCO_STATISTICS_URL = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz"

    def __init__(self, dataset_type: str = "subset", seed: int = 42):
        assert dataset_type in ["subset"], "Only subset is supported for now"
        self.size = dataset_type
        self.seed = seed
        self._captions = None
        self.sample_ids = self._get_sample_ids()
        self._statistics = None

    @property
    def captions(self) -> List[str]:
        if self._captions is None:
            self._load_captions()
        if self.size == "subset" and hasattr(self, "sample_ids"):
            return [self._captions[i] for i in self.sample_ids]
        return self._captions

    @property
    def statistics_mean(self) -> np.ndarray:
        if self._statistics is None:
            with urllib.request.urlopen(CocoDataset.COCO_STATISTICS_URL) as f:
                self._statistics = np.load(BytesIO(f.read()))
        return self._statistics['mu']

    @property
    def statistics_cov(self) -> np.ndarray:
        if self._statistics is None:
            with urllib.request.urlopen(CocoDataset.COCO_STATISTICS_URL) as f:
                self._statistics = np.load(BytesIO(f.read()))
        return self._statistics['sigma']

    
    def _get_sample_ids(self) -> List[int]:
        if self.size == "subset":
            rng = np.random.default_rng(self.seed)
            all_caption_ids = list(range(len(self.captions)))
            return rng.choice(all_caption_ids, size=self.N_SUBSET_SAMPLES, replace=False)
        else:
            raise NotImplementedError(f"Invalid dataset type: {self.size}")

    def _load_captions(self) -> None:
        df = pd.read_csv(CocoDataset.COCO_CAPTIONS_URL, sep='\t')
        self._captions = df['caption'].tolist()

if __name__ == "__main__":
    dataset = CocoDataset()
    print(dataset.captions[:10])