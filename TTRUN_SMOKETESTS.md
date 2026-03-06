# Running Metal Smoketests with ttrun

This document describes how to run the metal smoketests using the `ttrun` instance installed as part of the tt-xla wheel, triggered through torch-xla.

## Overview

Instead of using `mpirun` directly, these scripts use the `ttrun.py` from the installed tt-xla wheel package. This ensures consistency with the torch-xla runtime environment.

## Prerequisites

- tt-xla wheel must be installed in the environment
- tt-metal must be built at `/workspace/tt-metal/build`
- Hostfile must exist at `/etc/mpirun/hostfile`
- Remote execution script must exist at `/workspace/tests/torch/multi_host/experimental/remote_docker.sh`

## Scripts

### 1. test_physical_discovery

Run the physical discovery test:

```bash
bash /workspace/run_physical_discovery_ttrun.sh
```

This script:
- Uses ttrun from `/workspace/venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/ttnn/ttnn/distributed/ttrun.py`
- Runs `/workspace/tt-metal/build/test/tt_metal/tt_fabric/test_physical_discovery`
- Forwards `LD_LIBRARY_PATH` to include `/workspace/tt-metal/build/lib`

### 2. run_cluster_validation

Run the cluster validation test with connectivity checking and traffic sending:

```bash
bash /workspace/run_cluster_validation_ttrun.sh
```

This script:
- Uses ttrun from the tt-xla wheel installation
- Runs `/workspace/tt-metal/build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic`
- Forwards `LD_LIBRARY_PATH` to include `/workspace/tt-metal/build/lib`

## Key Differences from mpirun

1. **ttrun wrapper**: Uses the Python ttrun script instead of calling mpirun directly
2. **Absolute paths**: All paths are absolute to avoid working directory issues
3. **LD_LIBRARY_PATH**: Explicitly forwards the library path to include tt-metal build libraries
4. **Rank bindings**: Uses dual_t3k_rank_bindings.yaml from the installed package

## Integration into GitHub Workflows

These scripts can be integrated into the GitHub workflow by replacing the mpirun commands. Example:

```yaml
- name: Run Physical Discovery Test
  run: bash /workspace/run_physical_discovery_ttrun.sh

- name: Run Cluster Validation Test
  run: bash /workspace/run_cluster_validation_ttrun.sh
```

## Troubleshooting

If tests fail:

1. **Verify paths**: Check that all absolute paths in the scripts match your environment
2. **Check LD_LIBRARY_PATH**: Ensure tt-metal libraries are accessible
3. **Hostfile**: Verify `/etc/mpirun/hostfile` contains correct host entries
4. **Rank bindings**: Confirm `dual_t3k_rank_bindings.yaml` exists and is correct for your setup
5. **Test binaries**: Ensure tt-metal tests are built and executable

## Environment Variables

The scripts use these paths by default:
- `TTRUN_SCRIPT`: `/workspace/venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/ttnn/ttnn/distributed/ttrun.py`
- `RANK_BINDING`: `/workspace/venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml`
- `HOSTFILE`: `/etc/mpirun/hostfile`
- `TT_METAL_BUILD`: `/workspace/tt-metal/build`

Modify these in the scripts if your environment differs.
