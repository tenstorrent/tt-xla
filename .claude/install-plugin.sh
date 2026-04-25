#!/usr/bin/env bash
# Installs the tt-xla-dev Claude Code plugin to ~/.claude/plugins/tt-xla-dev/
# Run from the repo root: bash .claude/install-plugin.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$SCRIPT_DIR/plugins/tt-xla-dev"
PLUGIN_DEST="$HOME/.claude/plugins/tt-xla-dev"

echo "Installing tt-xla-dev plugin..."
echo "  src:  $PLUGIN_SRC"
echo "  dest: $PLUGIN_DEST"

mkdir -p "$PLUGIN_DEST"
cp -r "$PLUGIN_SRC/." "$PLUGIN_DEST/"

echo "Done. Plugin installed at $PLUGIN_DEST"
echo "Restart Claude Code to activate the plugin."
echo ""
echo "Available commands:"
echo "  /tt-xla-dev:local-review        Review staged changes before creating a PR"
echo "  /tt-xla-dev:ci-review [PR#]     Summarize CI status for a PR"
echo "  /tt-xla-dev:create-pr [area]    Create a PR with proper template and reviewers"
