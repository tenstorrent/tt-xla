---
name: tt-forge-model-remediate
description: Remediate a tt-forge-model pytest in a compile-only environment.
model: Opus
allowed-tools: Bash, Read, Grep, Edit, Task
---

venv
- cache
- python venv
- compile only / system desc
- random weights
- clearing

The goal of this task is to remediate any test or environment level failures by
executing the test through our xla compiler infra.

Before starting any work, please ensure the following environment variables are set!
If it's not, this skill cannot be invoked.
- TT_COMPILE_ONLY_SYSTEM_DESC
- TT_XLA_ROOT 

## Available skill scripts

These scripts are located in a subdirectory of this skill.

- `scripts/setup_venv.sh`: Sets up the proper environment.
- `scripts/run.sh`: Activates environment and forwards all args to pytest.
- `scripts/teardown_venv.sh`: Teardown the venv.
- `scripts/activate`: Source the python virtual environment.

## Phase 1. Setup the venv.

Invoke the `scripts/setup_venv.sh` script to setup a local venv.

## Phase 2. Run the model.

Run the model with the run script: `scripts/run.sh $0`

## Phase 3. Diagnose and resolve all failures.

If the test fails because of a python dependency issue please update (or create) the
requirements.txt file that lives next to the test.

Please commit independent, atomic, fixes for each issue that arises with the
test.  In a loop let's:
- Double check the changes look good.
- Fix anything that looks out of place.
- Use git add to add any new requirements.txt files or any other new files you
  introduced.
- Run `source scripts/activate && pre-commit run --all-files` to reformat the code.
- Git commit the changes to checkpoint our progress with a short commit message.
  Describing the fix made for this single issue.
