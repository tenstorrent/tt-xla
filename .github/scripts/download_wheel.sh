#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

rm -rf wheels
# If no artifact found with short SHA, try with longer SHA
for size in 7 8 9 10 11 12; do
  echo "Checking for artifact with SHA size: $size"
  wheel_artifact_name="xla-whl-release-$(git rev-parse --short=$size HEAD)"
  response=$(curl -s -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/tenstorrent/tt-xla/actions/artifacts?name=$wheel_artifact_name")
  total_count=$(echo "$response" | jq -r '.total_count')
  if [ "$total_count" -gt 0 ]; then
    echo "Found artifact!"
    artifacts_run_id=$(echo "$response" | jq -r '.artifacts[0].workflow_run.id')
    break
  fi
done

# If still no artifact found
if [ -z "$artifacts_run_id" ]; then
  echo "No artifact found."
  exit 1
fi

echo "Downloading wheel from run_id $artifacts_run_id name $wheel_artifact_name"
gh run download "$artifacts_run_id" --repo "tenstorrent/tt-xla" --dir wheels --name "$wheel_artifact_name"
echo "Installing wheel artifact"
pip install wheels/*.whl --upgrade
rm -rf wheels
exit 0
