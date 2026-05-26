#!/usr/bin/env bash
# Restore venv/bin/python* to symlinks against /usr/bin/python3.12 (TT host default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/venv/bin"
if [[ -n "${PY312:-}" && -x "${PY312}" ]]; then
    :
elif [[ -x /usr/bin/python3.12 ]]; then
    PY312="/usr/bin/python3.12"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; exit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    PY312="$(command -v python3)"
else
    echo "error: Python 3.12 not found. Run: PY312=/usr/bin/python3.12 $0" >&2
    exit 1
fi

# Remove portable launcher / broken copies
rm -f "${BIN}/python3.12" "${BIN}/.python312_launcher_installed"

ln -sf "${PY312}" "${BIN}/python3.12"
ln -sf python3.12 "${BIN}/python"
ln -sf python3.12 "${BIN}/python3"

"${BIN}/python" -c "import sys; print('restored:', sys.executable, sys.version.split()[0])"
