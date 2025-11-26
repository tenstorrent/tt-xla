#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

wheel_artifact_name="xla-whl-release-$(git rev-parse --short HEAD)"
# Use GitHub API to check for existing artifacts
response=$(curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-xla/actions/artifacts?name=$wheel_artifact_name")
total_count=$(echo "$response" | jq -r '.total_count')
if [ "$total_count" -gt 0 ]; then
  echo "exists=true" >> "$GITHUB_OUTPUT"
  # Get the download URL of the most recent artifact
  artifacts_run_id=$(echo "$response" | jq -r '.artifacts[0].workflow_run.id')

  echo "Downloading wheel artifact: $wheel_artifact_name from run ID: $artifacts_run_id"
  gh run download $artifacts_run_id --repo "tenstorrent/tt-xla" --dir wheels --name $wheel_artifact_name

  echo "Installing wheel artifact: $wheel_artifact_name"
  pip install wheels/*.whl --force-reinstall
  exit 0
else
  echo "No existing artifact found for: $wheel_artifact_name"
  exit 1
fi
