import torchax
import tt_xla.utils as ttxla_utils
import tt_xla.dist as ttxla_dist
from torchax.tensor import j2t, t2j

import torch
import torch.distributed as dist

def test_spawn(input):
    def test_all_reduce_spmd(index, a):
      b = torch.mul(a, a)
      dist.all_reduce(b, dist.ReduceOp.SUM)
      return b
    return ttxla_dist.spawn(test_all_reduce_spmd, [input])

if __name__ == "__main__":
    # Initialize torchax and tt_xla
    torchax.enable_globally()
    ttxla_utils.initialize(use_shardy=True, backend="cpu,tt")
    ttxla_dist.init_process_group()

    input = torch.randn(1024, 1024)
    res = test_spawn(input)
    print(j2t(res))

    ttxla_dist.destroy_process_group()
