
import tt_xla.utils as ttxla
from tt_xla.tt_torchax import TorchAXSingleInference

from mnist_utils import MNISTModel, TestLoader

def mnist_single_main():
    test_loader = TestLoader()

    inputs = []
    for data, target in test_loader:
        inputs.append(data)
        break
    input = inputs[0]

    m = MNISTModel()
    m.eval()

    output_cpu = m(input)

    m = TorchAXSingleInference(m)
    output_tt = m(input)

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output tt ----")
    print(output_tt)

if __name__ == "__main__":
    ttxla.initialize()
    mnist_single_main()
