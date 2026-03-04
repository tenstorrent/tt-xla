import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import torch.nn.functional as F


def test_sanity1():

    class sanity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.01

        def forward(self, cls_logits ):
            
            pred_scores = F.softmax(cls_logits, dim=-1)
            
            return pred_scores
            
    model = sanity()
    model.eval()
    
    cls_logits = torch.load('cls_logits_org.pt')
    logger.info("cls_logits={}",cls_logits)
    logger.info("cls_logits.shape={}",cls_logits.shape)
    logger.info("cls_logits.dtype={}",cls_logits.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[cls_logits],
    )

    tester.test(workload)
    


def test_sanity2():

    class sanity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.01

        def forward(self, cls_logits ):
            
            pred_scores = F.softmax(cls_logits, dim=-1)
            
            for scores in pred_scores:

                for label in range(1, 2):
                    score = scores[:, label]
                
            return score


    model = sanity()
    model.eval()
    
    cls_logits = torch.load('cls_logits_org.pt')
    logger.info("cls_logits={}",cls_logits)
    logger.info("cls_logits.shape={}",cls_logits.shape)
    logger.info("cls_logits.dtype={}",cls_logits.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[cls_logits],
    )

    tester.test(workload)
    
def test_sanity3():

    class gt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.01

        def forward(self, score ):

            keep_idxs = score > self.score_thresh
            
            return keep_idxs


    model = gt()
    model.eval()
    
    score = torch.load('score.pt',map_location="cpu")
    logger.info("score={}",score)
    logger.info("score.shape={}",score.shape)
    logger.info("score.dtype={}",score.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[score],
    )

    tester.test(workload)
    

def test_sanity4():

    class gt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.01

        def forward(self, scores ):
            score = scores[:, 1]

            keep_idxs = score > self.score_thresh
            
            return keep_idxs


    model = gt()
    model.eval()
    
    scores = torch.load('scores.pt',map_location="cpu")
    logger.info("scores={}",scores)
    logger.info("scores.shape={}",scores.shape)
    logger.info("scores.dtype={}",scores.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[scores],
    )

    tester.test(workload)