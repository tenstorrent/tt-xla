#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

long_sha=$(git rev-parse HEAD)
echo "token '$GH_TOKEN'"
# Use GitHub API to check for existing artifacts
echo "Checking for workflows for sha: $long_sha"
artifacts_run_id=$(curl -L -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GH_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-xla/actions/runs?head_sha=$long_sha" | jq '.workflow_runs[] | select(.name == "On Push" or .name == "On Nightly") | .id')
if [ -z "$artifacts_run_id" ]; then
  echo "No workflow run found for commit: $long_sha"
  exit 1
fi
gh run download "$artifacts_run_id" --repo "tenstorrent/tt-xla" --dir wheels --pattern "xla-whl-release-*"
wheel_path=$(find wheels -name "*.whl" | head -n 1)
if [ -z "$wheel_path" ]; then
  echo "No wheel file found in downloaded artifacts."
  exit 1
fi
echo "Installing wheel artifact: $wheel_path"
pip install wheels/*.whl --force-reinstall
exit 0
