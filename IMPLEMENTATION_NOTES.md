# Multi-Host Workflow POC - Implementation Notes

## Goal
Create a pipeline that can use mpirun to reach multiple hosts and perform basic sanity checks (hostname, container verification).

## Context
Two different multi-host paradigms exist:

### 1. Manual-update-durations.yml Approach (Current)
- **Pattern**: Matrix strategy with multiple GitHub Actions runners
- Each node (forge-qbae-01, forge-qbae-02) has its own runner
- Each runner starts its own container
- Coordination via SSH between containers
- Distributed barrier: workers pause, controller waits and orchestrates
- No assumed shared filesystem (though `_work` is shared in practice)

### 2. Multi-host-physical.yaml Approach (Target)
- **Pattern**: Single runner on controller node
- Only the controller runs GitHub Actions runner
- Workers are "live" but not directly targetable by GitHub Actions
- Single container on controller with mpirun capabilities
- Uses `/etc/mpirun/hostfile` to discover worker nodes
- Mounts SSH keys for keyless SSH to workers
- Shared NFS at `_work` between all nodes
- Example: `run_dual_t3k_tests.sh` uses mpirun to execute across all hosts

## Implementation

### POC Workflow: `.github/workflows/multihost-poc.yml`

**Purpose**: Minimal workflow to validate multi-host setup can reach all nodes via mpirun

**Key components**:
1. **Runner targeting**: `multi-host-t3000` label (controller node)
2. **Container setup**:
   - Uses `tt-xla-ci-ubuntu-22-04:latest` image
   - Mounts `/etc/mpirun:/etc/mpirun:ro` for hostfile
   - Mounts `~/.ssh:~/.ssh:ro` for keyless SSH
   - Privileged mode with host networking and PID namespace
3. **Hostname check**:
   - Reads `/etc/mpirun/hostfile` to show available hosts
   - Runs `mpirun hostname` to get hostname from all nodes
   - Checks for `/.dockerenv` to verify container execution

**MPI configuration**:
```bash
mpirun --hostfile /etc/mpirun/hostfile \
  --mca btl_tcp_if_exclude docker0,lo \
  <command>
```

## Key Differences from Original Approach

| Aspect | manual-update-durations.yml | multihost-poc.yml |
|--------|----------------------------|-------------------|
| Runners | Multiple (1 per node) | Single (controller) |
| Containers | 1 per node | 1 on controller |
| Coordination | SSH + barrier files | mpirun |
| Worker access | GitHub Actions direct | mpirun from controller |
| Filesystem | Independent (with shared `_work`) | Shared NFS assumed |

## Next Steps

After validating the POC works:

1. **Container coordination**: Test mpirun launching containers on worker nodes
2. **Advanced MPI tests**: Run actual distributed workloads (not just hostname)
3. **Port full workflow**: Migrate the test logic from manual-update-durations.yml
4. **Environment variables**: Properly configure distributed runtime variables:
   - `TT_DISTRIBUTED_WORKER_PATH`
   - `TT_RUNTIME_ENABLE_DISTRIBUTED`
   - `TT_DISTRIBUTED_RANK_BINDING`
   - `TT_DISTRIBUTED_CONTROLLER_HOST_NAME`
   - `TT_DISTRIBUTED_BTL_TCP_IF_INCLUDE`
   - `TT_DISTRIBUTED_HOSTS_LIST`

## Testing the POC

To test this workflow:
1. Push the workflow file to the repository
2. Go to Actions tab in GitHub
3. Select "Multi-Host POC - Hostname Check"
4. Click "Run workflow"
5. Verify output shows hostnames from all nodes in the cluster

Expected output:
```
Hostfile found at /etc/mpirun/hostfile:
<host1>
<host2>

Running hostname via mpirun across all hosts...
<host1>
<host2>

Checking for container environment (.dockerenv)...
<host1>: Running in Docker container
<host2>: Running in Docker container
```

## Notes

- The POC assumes keyless SSH is already configured between hosts
- The Docker image must have OpenMPI installed
- Network interface exclusions (`docker0,lo`) prevent MPI from using container-only networks
- The `--privileged --pid=host` flags may need adjustment based on security requirements
