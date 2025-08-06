import torch
import pytest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import copy
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import os
from loguru import logger
import sys

class SimplifiedMnistModel(torch.nn.Module):
    def __init__(self):
        super(SimplifiedMnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # No dropout as random generation is not supported yet.
        # self.dropout1 = nn.Dropout(0)
        # self.dropout2 = nn.Dropout(0)
        self.fc1 = nn.Linear(36864, 128)
        # self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # No max pool as max pool with indices is not supported yet
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # # #x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class TrainTestBase:
    from typing import Type

    def __init__(self, device, model, optimizer=None, loss_fn=None):
        self.device = device
        # dtype should be set from user code, not here
        self.model = copy.deepcopy(model).to(self.device).train()

        if optimizer is not None:
            self.optimizer = optimizer(self.model.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_fn = loss_fn if loss_fn is not None else None

        self.compiled_step_fn = torch_xla.compile(self.run_step, name="train_step")
        self.losses = []

    def update_target(self, target_output):
        if target_output is not None:
            if isinstance(target_output, torch.Tensor):
                self.target = target_output
            elif isinstance(target_output, (list, tuple)):
                self.target = type(target_output)(t for t in target_output)
            elif isinstance(target_output, dict):
                self.target = {k: v for k, v in target_output.items()}
            else:
                raise TypeError(f"Unsupported target type: {type(target_output)}")
        else:
            self.target = None

    def run_forward(self):
        result = self.model(self.test_input)
        return result

    def run_backward(self, result, target):
        loss = self.loss_fn(result, target)
        loss.backward()
        return loss
    
    def step_optimizer(self):
        raise NotImplementedError("Subclasses must implement the step_optimizer method.")

    def run_step(self, test_input, target_output, output_sharding=None):
        # Run forward pass
        self.test_input = test_input
        self.update_target(target_output)
        self.optimizer.zero_grad()
        result = self.run_forward()
        if output_sharding is not None:
            xs.mark_sharding(self.target, self.mesh, output_sharding)

        # Run backward pass
        loss = self.run_backward(result, self.target)
        
        # self.losses.append(loss.item())

        # Step optimizer
        self.step_optimizer()
        return loss

    def run(self, test_input, target_output, output_sharding=None):
        loss = self.compiled_step_fn(test_input, target_output, output_sharding)
        # xm.mark_step()
        return loss

    def get_parameters(self):
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def set_device_loader(self, loader, mesh, input_sharding_spec=None):
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.spmd as xs

        self.mesh = mesh
        input_sharding = None
        if mesh is not None:
            if input_sharding_spec is not None:
                input_sharding = xs.ShardingSpec(mesh, input_sharding_spec)
            else:
                # Default to sharding the batch dimension
                input_sharding = xs.ShardingSpec(mesh, ("data", None, None, None))
        self.train_device_loader = pl.MpDeviceLoader(
            loader,
            self.device,
            input_sharding=input_sharding,
        )
        # Shard the input's batch dimension along the `data` axis, no sharding along other dimensions

    def get_losses(self):
        return self.losses

class TrainTestCpu(TrainTestBase):
    def __init__(self, model, optimizer=None, loss_fn=None):
        super().__init__("cpu", model, optimizer, loss_fn)

    def step_optimizer(self):
        self.optimizer.step()
        

class TrainTestXLA(TrainTestBase):
    def __init__(self, device, model, optimizer=None, loss_fn=None):
        super().__init__(device, model, optimizer, loss_fn)
    
    def step_optimizer(self):
        xm.optimizer_step(self.optimizer)
        torch_xla.sync()


def setup_training_environment():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

def setup_mesh():
    num_devices = xr.global_runtime_device_count()
    print(f"Running on {num_devices} devices")
    mesh_shape = (num_devices, 1, 1, 1)
    axis_names = ('data', 'c', 'h', 'w')
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    print(f"Mesh shape: {mesh_shape}, Device IDs: {device_ids}")
    return mesh

def create_data_loader(batch_size=32, dtype=torch.bfloat16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize the MNIST dataset
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in loader:
        data, target = batch
        data = data.to(dtype)
        yield data, target
    

def run_training_on_xla(model, mesh, batch_size=32, num_steps=16) :
    xla_runner = TrainTestXLA(
            device=torch_xla.device(), model=model, optimizer=None, loss_fn=nn.NLLLoss()
        )
    loader = create_data_loader(batch_size)
    xla_runner.set_device_loader(
        loader, mesh, input_sharding_spec=("data", None, None, None)
    )
    params = []
    for step, (data, target) in enumerate(xla_runner.train_device_loader):
        if step >= num_steps:
            break
        xla_runner.run(data, target, output_sharding=("data", ))
        params.append(xla_runner.get_parameters())
        print(f"Step {step + 1}/{num_steps} completed")
    
    return xla_runner

def run_training_on_cpu(model, batch_size=32, num_steps=16):
    cpu_runner = TrainTestCpu(model=model, optimizer=None, loss_fn=nn.NLLLoss())
    
    loader = create_data_loader(batch_size)

    step_count = 0
    for data, target in loader:
        if step_count >= num_steps:
            break
        
        data = data.to('cpu')
        target = target.to('cpu')
        
        cpu_runner.run(data, target)
        print(f"CPU Step {step_count + 1}/{num_steps} completed")
        step_count += 1
    
    return cpu_runner

def test_training_on_multiple_devices():
    setup_training_environment()
    mesh = setup_mesh()
    model = SimplifiedMnistModel()
    model = model.to(torch.bfloat16)
    initial_params = model.state_dict()
    
    batch_size = 8
    num_steps = 2

    xla_runner = run_training_on_xla(model, mesh, batch_size=batch_size, num_steps=num_steps)
    
    xla_params = xla_runner.get_parameters()
    xla_params_on_cpu = {k: v.cpu() for k, v in xla_params.items()}
    print(f'xla_looses: {xla_runner.get_losses()}')

    cpu_runner = run_training_on_cpu(model, batch_size=batch_size, num_steps=num_steps)
    cpu_params = cpu_runner.get_parameters()
    print(f'cpu_looses: {cpu_runner.get_losses()}')

    make_assert = False 
    for key in initial_params.keys():
        if key in xla_params_on_cpu and key in cpu_params:
            xla_param = xla_params_on_cpu[key]
            cpu_param = cpu_params[key]
            
            are_they_close = torch.allclose(xla_param, cpu_param, atol=0.1)

            if not are_they_close:
                abs_diff = torch.abs(xla_param - cpu_param)
                
                max_diff = torch.max(abs_diff)
                mean_diff = torch.mean(abs_diff)
                
                max_diff_idx = torch.argmax(abs_diff)
                
                xla_val_at_max_diff = xla_param.flatten()[max_diff_idx]
                cpu_val_at_max_diff = cpu_param.flatten()[max_diff_idx]

                error_message = (
                    f"\nParameter '{key}' does not match!\n"
                    f"  - Max difference: {max_diff.item():.6f}\n"
                    f"  - Mean difference: {mean_diff.item():.6f}\n"
                    f"  - Location of max diff (flattened index): {max_diff_idx.item()}\n"
                    f"  - XLA value at location: {xla_val_at_max_diff.item():.6f}\n"
                    f"  - CPU value at location: {cpu_val_at_max_diff.item():.6f}"
                )
                print(error_message)
                make_assert = True
            else:
                abs_diff = torch.abs(xla_param - cpu_param)
                max_diff = torch.max(abs_diff)
                print(f"Parameter '{key}' matches. Max difference: {max_diff.item():.6f}")
        else:
            raise KeyError(f"Parameter {key} not found in both XLA and CPU parameters")
    
    assert not make_assert, "Some parameters do not match between XLA and CPU training."
