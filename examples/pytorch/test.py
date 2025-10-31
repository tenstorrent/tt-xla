# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


EXPORT_PATH = 'myresnet'


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MyResnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.EXPORT_PATH = "myresnet"
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.EXPORT_PATH = "simplemodel"
        self.flatten = nn.Flatten(0)
        self.mm1 = nn.Linear(in_features=3 * 224 * 224, out_features=100, bias=True)
        self.mm2 = nn.Linear(in_features=100, out_features=90, bias=True)
        self.tail = self.Tail()

    def forward(self, x):
        x = self.flatten(x)
        x = self.mm1(x)
        # print(x.shape)
        x = self.mm2(x)
        x = self.tail(x)
        return x

    class Tail(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(False)
            self.softmax = nn.Softmax(-1)

        def forward(self, x):
            x = self.relu(x)
            x = self.softmax(x)
            return x


def get_model():
    # model: nn.Module = MyResnet()
    model: nn.Module = SimpleModel()

    model = model.eval()

    return model

def get_input():
    torch.manual_seed(4)
    input = torch.rand((1, 3, 224, 224))

    return input


def dump_tensors():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": model.EXPORT_PATH,
            "dump_inputs": True,
        }
    )
    output = model(input)
    # output = model(input).to("cpu")
    return


def dump_code():
    xr.set_device_type("TT")

    model = get_model()

    model.compile(backend="tt")

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": model.EXPORT_PATH,
            "backend": "codegen_py",
        }
    )
    output = model(input)
    # output = model(input).to("cpu")


def run_on_cpu():
    model = get_model()

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output)


def run_on_tt():
    xr.set_device_type("TT")

    model = get_model()

    model.compile(backend="tt")

    # Generate inputs.
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output)
    return


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # print("Dumping code...")
    # dump_code()

    # print("Dumping tensors...")
    # dump_tensors()

    # print("Running on cpu...")
    # run_on_cpu()

    print("Running on tt...")
    run_on_tt()