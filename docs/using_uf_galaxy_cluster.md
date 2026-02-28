# How to use the UF Galaxy cluster for development

1. Log into the UF cluster's entrypoint machine: `ssh {user}@UF-EV-B13-GWH01`
2. Check available machines: `sinfo`.
3. Reserve a machine: `srun --nodelist={machine name} --pty /bin/bash`.
    - This will make the machine show up as "allocated" for as long as the terminal with
    your `srun` command is running.
4. SSH into the reserved machine on a different terminal instance.
5. [Only need to do this once] Create an ssh key within the `/data/{user}/.ssh` folder
(**not** under `~/.ssh`, since the HOME folder on all Slurm machines is considered to be
ephemeral).
6. Depending on your preferred development style, do one of the following:
    - **Prefer terminal development**: Run the `scripts/run_tt_xla_container.sh` script
    to create a docker container. You can do your development work there.
    - **Prefer VS Code/Cursor development**:
        1. Open your preferred editor and connect to the reserved machine.
        2. A pop-up should show up asking if you would like to re-open this project in
        a dev container, select "Yes". If nothing pops up then you can open up the command
        palette (`Cmd+Shift+P`) and select the `Dev Containers: Reopen in Container` option.
