---
name: bringup-hf-model
description: Bring up a new model from a huggingface link.
model: Opus
allowed-tools: Bash, Read, Grep, Edit, Task
---

The goal of this task is to create a new testcase for a huggingface model.

## Available skill scripts

These scripts are located in a subdirectory of this skill.

- `scripts/create_venv.sh`: Creates a new python venv.

## Phase 1. Create a new venv with tenstorrent pjrt plugin installed.

Use the script `scripts/create_venv.sh` to create a new python venv.

## Phase 2. Review the huggingface model card.

Review this huggingface model card: $0

## Phase 3. Explore the forge models repo for comparable examples.

First check if this model already exists in the repo, if it does, let's early
out of this skill, there's nothing to do.

Explore the forge models repo (this repo) for comparable examples to this model

## Phase 4. Introduce the new testcase.

Introduce a new testcase file for this huggingface model and follow the same test
harness conventions discovered from phase 3.

Where applicable, always use the ModelGroup enum value `ModelGroup.VULCAN`.

After adding the new test, let's checkpoint our progress:
- First run `source .venv/bin/activate && pre-commit run --all-files` to format our code.
- Next git add the newly added files and git commit them with a nice commit message.

## Phase 5. Lint and validate.

Please commit independent, atomic, fixes for each issue that arises with the
test.  In a loop let's:
- Double check the changes look good.
- Fix anything that looks out of place.
- Run `source .venv/bin/activate && pre-commit run --all-files` to reformat the code.
- Git commit the changes to checkpoint our progress with a short commit message.
  Describing the fix made for this single issue.
