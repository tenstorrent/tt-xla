# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import model_pt
import ttnn
import utils
from model_ttnn import CLIPVisionEncoderAndResamplerTTNN
from weights_loader import load_inputs_for__main

_CONST_EVAL_CACHE = {}


def main():
    # Get PyTorch golden output
    pt_input = model_pt.get_input()
    pt_output = model_pt.run_pytorch_inference(input_tensor=pt_input)

    # Load TTNN inputs
    load_inputs_for__main_0 = load_inputs_for__main()
    model = CLIPVisionEncoderAndResamplerTTNN(
        load_inputs_for__main_0, _CONST_EVAL_CACHE
    )

    # Run TTNN model
    for i in range(3):
        start_time = time.time()

        # Run TTNN model
        out_ttnn_device = model(load_inputs_for__main_0[390])[0]

        # Get outputs
        out_ttnn_host = ttnn.from_device(out_ttnn_device, blocking=True)
        end_time = time.time()

        # Calculate duration and PCC
        duration = (end_time - start_time) * 1000
        pcc = utils.calculate_pcc(pt_output, ttnn.to_torch(out_ttnn_host))

        # Print results
        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tPCC: {pcc:.6f}")

    return 0


if __name__ == "__main__":
    main()
