import random

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class AddOne(torch.nn.Module):
    def forward(self, *tensors):
        # sum = 0
        # for t in tensors:
        #     sum += t
        
        return tuple((t + t for t in tensors))


def main():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Option to use random shapes
    use_random_shapes = True
    # Shape range for random shapes (min_dim, max_dim)
    min_dim = 32
    max_dim = 128
    # Number of dimensions for random shapes
    num_dims = 2

    # Create tensors
    num_tensors = 500
    if use_random_shapes:
        tensors = []
        for _ in range(num_tensors):
            shape = tuple(random.randint(min_dim, max_dim) for _ in range(num_dims))
            tensors.append(torch.zeros(shape, dtype=random.choice([torch.float32, torch.bfloat16])))
    else:
        tensors = [torch.zeros(64, 127) for _ in range(num_tensors)]
    
    # Move all tensors to device
    tensors_on_device = [t.to(device) for t in tensors]
    
    # Compile model and move to device
    compiled_model = torch.compile(AddOne(), backend="tt")
    compiled_model = compiled_model.to(device)
    
    # Add one to all tensors at once
    print(f"Processing all {num_tensors} tensors", flush=True)
    results = compiled_model(*tensors_on_device)
    print(f"Completed processing all {num_tensors} tensors", flush=True)
    return results


if __name__ == "__main__":
    xr.set_device_type("TT")
    results = main()
    print(f"Returned {len(results)} results", flush=True)


    # results = main()
    # print(f"Returned {len(results)} results", flush=True)

