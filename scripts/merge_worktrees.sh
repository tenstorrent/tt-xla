#!/bin/bash

set -e

for i in $(seq 0 24); do
  echo "$i"
  if [ ! -d ".worktrees/worker-$i" ]; then
    continue
  fi
  cd .worktrees/worker-$i
  git stash
  git rebase -X theirs nsmith/hf-bringup
  cd ../..
  git merge nsmith/hf-bringup-$i
done
