#!/bin/bash
set -x
git worktree list | tail -n +2 | awk '{print $1}' | xargs -r -I{} git worktree remove --force {}
rm -rf .worktrees
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git branch --list | awk '{print $1}' | grep "$BRANCH-.*" | xargs git branch -D
