#!/usr/bin/env bash
# Generate .mcp.json with absolute paths for this checkout.
# Run once after cloning: bash scripts/setup-mcp.sh

set -euo pipefail

REPO_PATH="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="${REPO_PATH}/.mcp.json.template"
OUT="${REPO_PATH}/.mcp.json"

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: .mcp.json.template not found at $TEMPLATE"
  exit 1
fi

sed "s|\${REPO_PATH}|${REPO_PATH}|g" "$TEMPLATE" > "$OUT"
echo "Written: $OUT"
echo "Repo path: $REPO_PATH"
echo
echo "Next: restart Claude Code to load the MCP servers."
echo "Verify with: /mcp"
