# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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
        x = F.relu(x)
        # #x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
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
            self.optimizer = torch.optim.Adam(self.model.parameters())

        self.loss_fn = loss_fn if loss_fn is not None else None

        self.compiled_step_fn = torch_xla.compile(self.run_step, name="train_step")

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
        xm.optimizer_step(self.optimizer)
        torch_xla.sync()

    def run_step(self, test_input, target_output):
        # Run forward pass
        self.test_input = test_input
        self.update_target(target_output)
        self.optimizer.zero_grad()
        result = self.run_forward()

        # Run backward pass
        loss = self.run_backward(result, self.target)

        # Step optimizer
        self.step_optimizer()
        return loss

    def run(self, test_input, target_output):
        loss = self.compiled_step_fn(test_input, target_output)
        # xm.mark_step()
        return loss

    def get_parameters(self):
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def set_device_loader(self, loader, mesh):
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.spmd as xs

        self.mesh = mesh
        self.train_device_loader = pl.MpDeviceLoader(
            loader,
            self.device,
            input_sharding=xs.ShardingSpec(mesh, ("data", None, None, None)),
        )
        # Shard the input's batch dimension along the `data` axis, no sharding along other dimensions


class TrainTestCpu(TrainTestBase):
    def __init__(self, model, optimizer=None, loss_fn=None):
        super().__init__("cpu", model, optimizer, loss_fn)


class TrainTestXLA(TrainTestBase):
    def __init__(self, device, model, optimizer=None, loss_fn=None):
        super().__init__(device, model, optimizer, loss_fn)


os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
    os.environ["LOGGER_LEVEL"] = "DEBUG"

    from torch_xla.experimental import plugins

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__),
                "/localdev/sshon/tt-xla/build/src/tt/pjrt_plugin_tt.so",
            )

    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.use_spmd()
    torch_xla.sync(True, True)


def training_on_single_device():
    num_steps = 1  # 1 input, 1 target. could be adjusted
    torch.manual_seed(1)
    model = SimplifiedMnistModel()
    model = model.to(torch.bfloat16)
    model = model.train()

    inputs_and_targets = []
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)
    data_iterator = iter(dataloader)
    inputs = []
    targets = []
    for _ in range(num_steps):
        test_input, target = next(data_iterator)
        print(f"Test input shape: {test_input.shape}, Target shape: {target.shape}")
        test_input = test_input.to(torch.bfloat16)
        inputs.append(test_input)
        targets.append(target)
        inputs_and_targets.append((test_input, target))

    xla_runner = TrainTestXLA(
        device=torch_xla.device(), model=model, optimizer=None, loss_fn=nn.NLLLoss()
    )
    for input, target in inputs_and_targets:
        xla_runner.run(input.to(torch_xla.device()), target.to(torch_xla.device()))

    params = xla_runner.get_parameters()
    params = {
        name: param.to("cpu") for name, param in params.items() if param.requires_grad
    }

    cpu_runner = TrainTestCpu(model=model, optimizer=None, loss_fn=nn.NLLLoss())
    for input, target in inputs_and_targets:
        cpu_runner.run(input, target)

    params_cpu = cpu_runner.get_parameters()
    params_cpu = {
        name: param for name, param in params_cpu.items() if param.requires_grad
    }

    for name, param_xla in params.items():
        param_cpu = params_cpu[name]
        assert torch.allclose(param_xla, param_cpu, atol=0.1), (
            f"Parameter '{name}' mismatch!\n"
            f"Max difference: {torch.abs(param_xla - param_cpu).max()}"
        )


def training_on_multiple_devices():
    num_steps = 32
    setup_tt_environment()
    torch.manual_seed(1)
    model = SimplifiedMnistModel()
    model = model.to(torch.bfloat16)
    model = model.train()

    inputs_and_targets = []
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)
    data_iterator = iter(dataloader)
    inputs = []
    targets = []
    for _ in range(num_steps):
        test_input, target = next(data_iterator)
        print(f"Test input shape: {test_input.shape}, Target shape: {target.shape}")
        test_input = test_input.to(torch.bfloat16)
        inputs.append(test_input)
        targets.append(target)
        inputs_and_targets.append((test_input, target))

    xr.use_spmd(True)
    num_devices = xr.global_runtime_device_count()
    print(f"Running on {num_devices} devices")
    mesh_shape = (num_devices, 1, 1, 1)
    axis_names = ("data", "c", "h", "w")
    device_ids = np.arange(num_devices).reshape(mesh_shape)
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    print(f"Mesh shape: {mesh_shape}, Device IDs: {device_ids}")

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=num_steps)

    xla_ddp = TrainTestXLA(
        device=torch_xla.device(), model=model, optimizer=None, loss_fn=nn.NLLLoss()
    )
    xla_ddp.set_device_loader(loader, mesh)
    # for step, (input, target) in enumerate(xla_ddp.train_device_loader):
    #     xla_ddp.run(input.to(torch_xla.device()), target.to(torch_xla.device()))
    for step, (input, target) in enumerate(loader):
        input_xla = input.to(torch_xla.device())
        target_xla = target.to(torch_xla.device())
        xs.mark_sharding(input_xla, mesh, ("data", None, None, None))
        xs.mark_sharding(target_xla, mesh, ("data",))
        xla_ddp.run(input_xla, target_xla)

    print("Training complete. Saving model parameters.")
    params = xla_ddp.get_parameters()
    params = {
        name: param.to("cpu") for name, param in params.items() if param.requires_grad
    }
    # print(f'params = {params}')

    cpu_runner = TrainTestCpu(model=model, optimizer=None, loss_fn=nn.NLLLoss())
    for input, target in inputs_and_targets:
        cpu_runner.run(input, target)

    params_cpu = cpu_runner.get_parameters()
    params_cpu = {
        name: param for name, param in params_cpu.items() if param.requires_grad
    }
    # print(f'params cpu = {params_cpu}')

    for name, param_xla in params.items():
        param_cpu = params_cpu[name]
        assert torch.allclose(param_xla, param_cpu, atol=0.1), (
            f"Parameter '{name}' mismatch!\n"
            f"Max difference: {torch.abs(param_xla - param_cpu).max()}"
        )

        print(
            f"{name} param has good param Max difference: {torch.abs(param_xla - param_cpu).max()}"
        )


def check_4d():
    num_steps = 2
    setup_tt_environment()
    torch.manual_seed(1)
    model = SimplifiedMnistModel()
    model = model.to(torch.bfloat16)
    model = model.train()

    inputs_and_targets = []
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)
    data_iterator = iter(dataloader)
    for _ in range(num_steps):
        test_input, target = next(data_iterator)
        print(f"Test input shape: {test_input.shape}, Target shape: {target.shape}")
        test_input = test_input.to(torch.bfloat16)
        test_input = torch.reshape(
            test_input,
            (
                1,
                28 * 28,
            ),
        )
        inputs_and_targets.append((test_input, target))

    xr.use_spmd(True)
    num_devices = xr.global_runtime_device_count()
    print(f"Running on {num_devices} devices")
    mesh_shape = (num_devices, 1, 1, 1)
    axis_names = ("data", "c", "h", None)
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    print(f"Mesh shape: {mesh_shape}, Device IDs: {device_ids}")

    from torch.utils.data import Dataset

    inputs_b = torch.cat(
        [x for x, _ in inputs_and_targets], dim=0
    )  # (num_steps*batch_size, 1, 28, 28)
    inputs_b = torch.reshape(inputs_b, (2, 1, 28, 28))
    in_features = inputs_b.shape[-1]  # 28
    out_features = 100
    inputs_w = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    inputs = inputs_b.to(torch_xla.device())
    inputs_w_d = inputs_w.to(torch_xla.device())
    targets = torch.cat(
        [y for _, y in inputs_and_targets], dim=0
    )  # (num_steps*batch_size,)
    xs.mark_sharding(inputs, mesh, ("data", None, None, None))
    # xs.mark_sharding(targets, mesh_shape, ('data',))
    # print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    output = torch.nn.functional.linear(inputs, inputs_w_d)
    print(f"Output shape: {output.shape}")
    c_batch_gathered = xm.all_gather(output, 0, groups=[[0, 1]], pin_layout=False)
    print(f"Gathered output shape: {c_batch_gathered.shape}")
    c_value = c_batch_gathered.to("cpu")
    print(f"c_value shape = {c_value.shape}")
    output_cpu = torch.nn.functional.linear(inputs_b, inputs_w)
    assert torch.allclose(c_value, output_cpu, atol=0.1), "not matched!"


def test_mnist_ttxla():
    # training_on_single_device()
    training_on_multiple_devices()
    # check_4d()
