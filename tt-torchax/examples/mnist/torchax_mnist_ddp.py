
import tt_xla.utils as ttxla
from tt_xla.tt_torchax import TorchAXDDPInference

import torch
from mnist_utils import MNISTModel, TestLoader

def mnist_ddp_spmd(): 
    test_loader = TestLoader(16)

    inputs = []
    for data, target in test_loader:
        inputs.append(data)
        break
    input = inputs[0]

    m = MNISTModel()
    m.eval()

    m_scripted = torch.jit.script(m)
    output_cpu = m_scripted(input)

    ddp_model = TorchAXDDPInference(m_scripted)
    sharded_input = ddp_model.shard_input(input)
    output_ddp = ddp_model(sharded_input)

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output ddp tt-xla ----") 
    print(output_ddp)

if __name__ == "__main__":
    ttxla.initialize()
    mnist_ddp_spmd()
