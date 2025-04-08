# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np

from .comparison import compare_atol, compare_pcc, AtolConfig, PccConfig


class InterpreterCheck:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def find_matching_files(self):
        files = os.listdir(self.folder_path)
        base_names = {}
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in [".txt", ".npy"]:
                if name not in base_names:
                    base_names[name] = []
                base_names[name].append(file)
        return {k: v for k, v in base_names.items() if len(v) > 1}

    def load_tensor(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == ".txt":
            with open(file_path, "r") as f:
                content = f.read()
                tensor_data = content.split("Tensor(")[1]
                tensor_data = tensor_data.split("shape=Shape")[0]
                return np.array(eval(tensor_data))
        elif ext == ".npy":
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def compare_tensors(self):
        matching_files = self.find_matching_files()
        for base_name, files in matching_files.items():
            tensors = []
            for file in files:
                file_path = os.path.join(self.folder_path, file)
                tensors.append(self.load_tensor(file_path))

            # compare_atol(
            #     tensors[0], tensors[1], AtolConfig(0.16)
            # )
            compare_pcc(tensors[0], tensors[1], PccConfig(0.95))


# Example usage:
# comparator = TensorComparator("/path/to/folder")
# comparison_results = comparator.compare_tensors()
# print(comparison_results)
