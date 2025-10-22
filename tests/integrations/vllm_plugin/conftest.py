# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_vllm_environment():
    """
    Set up environment variables specific to vLLM plugin tests.
    
    Sets VLLM_WORKER_MULTIPROC_METHOD=spawn to prevent OMP-related hangs in torch CPU
    execution when vLLM spawns child processes using fork. The fork method can cause
    deadlocks in multithreaded programs that use OpenMP (like PyTorch) because forked
    processes inherit the parent's thread state but only the main thread continues
    execution, leading to potential deadlocks when other threads were holding locks
    at the time of fork. Using spawn creates a fresh process which avoids this issue.
    """
    # Set the multiprocessing method to spawn to avoid OMP hangs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    yield
    
    # Optional cleanup after all tests in this directory complete
    # (keeping the env var set doesn't hurt, but we could clean it up)
    # os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
