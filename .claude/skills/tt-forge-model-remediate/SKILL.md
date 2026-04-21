---
name: tt-forge-model-remediate
description: Remediate a tt-forge-model pytest in a compile-only environment.
allowed-tools: Bash, Read, Grep, Edit, Task
---

The goal of this task is to remediate any test or environment level failures by
executing the test through our xla compiler infra.

Before starting any work, please ensure the following environment variables are set!
If it's not, this skill cannot be invoked.
- TT_XLA_ROOT 

Do not bother exploring the project's structure, only if the run command fails
and we need remediation should we start understanding the test and the project.

## Available skill scripts

These scripts are located in a subdirectory of this skill.

- `scripts/setup_venv.sh`: Sets up the proper environment.
- `scripts/run.sh`: Activates environment and forwards all args to pytest.
- `scripts/log_status.sh`: Logs that the remediation failed given the test name
  and a status string.
- `scripts/teardown_venv.sh`: Teardown the venv.
- `scripts/activate`: Source the python virtual environment.

## Phase 1. Git pull.

Ensure you have the latest changes for this branch:
- If the current branch exists on the remote: `git pull`
- If it does not, create one: `git push -u origin HEAD`

## Phase 2. Setup the venv.

Invoke the `scripts/setup_venv.sh` script to setup a local venv.

## Phase 3. Run the model & Diagnose and resolve all failures.

Determine if this model can run on a Tenstorrent quietbox system via one of the
following modes:
- `single_device`: Can fit models up to 32B parameters.
- `tensor_parallel`: Can fit models up to 120B parameters.

The test string `%0` should be of the form.  If the model should be run
`tensor_parallel` be sure to update the run string below by substituting
`single_device` with `tensor_parallel`.

Example:
- Before: `tests/runner/test_models.py::test_all_models_torch[dreamshaper_xl/pytorch-v2.1-Turbo-DPM-SDE-single_device-inference]`
- After: `tests/runner/test_models.py::test_all_models_torch[dreamshaper_xl/pytorch-v2.1-Turbo-DPM-SDE-tensor_parallel-inference]`

If the model is too large to run in either of these configs, then we skip step 4
and straight to step 5 to log the model is too large.

## Phase 4. Run the model & Diagnose and resolve all failures.

Run the model with the run script: `scripts/run.sh $0`

If the test exits normally and "PASSED" is shown in the output, then there is nothing
to do and we're done! Goto phase 5.

If the test fails because of a python dependency issue please update (or create) the
requirements.txt file that lives next to the test.

If python dependencies do not resolve, it might be best to start with a fresh
venv because we are reusing the same venv between skill invocations.  To do that
simply run `scripts/teardown_venv.sh` followed by `scripts/setup_venv.sh`.

Please commit independent, atomic, fixes for each issue that arises with the
test.  In a loop let's:
- Double check the changes look good.
- Fix anything that looks out of place.
- Use git add to add any new requirements.txt files or any other new files you
  introduced.
- Run `source scripts/activate && pre-commit run --all-files` to reformat the code.
- Git commit the changes to checkpoint our progress with a short commit message.
  Describing the fix made for this single issue.
- Git push (potentially with `-u origin HEAD` if the remote branch doesn't yet
  exist) all of the changes once it's done.
- If the remediation fails after 20 attempts goto phase 5.
- Once the test is passing (exits normally and "PASSED" is shown in the output) go to phase 5.
  If you have not reached 20 attempts, and the model hasn't passed, stay in this phase!

## Phase 5. Log status

Finally we want to log the status of the test:
- If the test passed despite initial issues use: `scripts/log_status.sh $0 PASS`
- If the remediation fails after 20 attempts let's log the error and stop this
  skill: `scripts/log_status.sh $0 MAX_FAILED_ATTEMPTS`
- If the model is too large to run on this target: `scripts/log_status.sh $0 MODEL_TOO_LARGE`
