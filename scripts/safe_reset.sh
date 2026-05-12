#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Safely reset all Tenstorrent devices on this host.
#
# Always kills processes that hold /dev/tenstorrent/* (pytest workers, ttrt,
# JAX/torch_xla scripts — whatever has the device open) before running
# tt-smi -r. Skipping the kill step while a process still holds the device
# can leave the card in a worse state (sometimes requiring a host reboot).
#
# In interactive mode, the script also walks the parent chain of each
# device holder and surfaces any python* ancestors (typically sweep
# orchestrators that survive the test kill); for each one it asks whether
# to kill it too. In --force mode, only device holders are killed.
#
# Usage:
#   scripts/safe_reset.sh           interactive (confirms + prompts per ancestor)
#   scripts/safe_reset.sh --force   non-interactive, device holders only

set -euo pipefail

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

command -v tt-smi >/dev/null || { echo "ERROR: tt-smi not on PATH; install it (pip install tt-smi) inside the tt-xla venv." >&2; exit 3; }

shopt -s nullglob
DEVS=( /dev/tenstorrent/* )
shopt -u nullglob
[[ ${#DEVS[@]} -gt 0 ]] || { echo "ERROR: no /dev/tenstorrent/* devices." >&2; exit 4; }

holders() { fuser "${DEVS[@]}" 2>/dev/null | tr -s ' \t' '\n' | grep -E '^[0-9]+$' | sort -un || true; }

# Walk up the parent chain from a device-holder PID, emitting each
# python*/pytest* ancestor until init or a non-python process is reached.
python_ancestors() {
    local pid=$1 ppid comm
    while ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' '); do
        [[ -z "$ppid" || "$ppid" -le 1 ]] && break
        comm=$(ps -o comm= -p "$ppid" 2>/dev/null | tr -d ' ')
        [[ "$comm" == python* || "$comm" == pytest* ]] || break
        echo "$ppid"
        pid=$ppid
    done
}

H=$(holders)
if [[ -z "$H" ]]; then
    echo "[safe_reset] no processes hold /dev/tenstorrent/*; nothing to kill."
else
    echo "[safe_reset] device holders:"
    ps -o pid,user,etime,stat,comm,args -p $H || true
    EXTRA=""
    if [[ "$FORCE" != 1 ]]; then
        [[ -t 0 ]] || { echo "[safe_reset] non-interactive; pass --force." >&2; exit 1; }
        read -rp "Kill these PIDs and continue? [y/N] " ans
        [[ "${ans,,}" =~ ^y(es)?$ ]] || { echo "Aborted."; exit 1; }
        for pid in $H; do
            for anc in $(python_ancestors "$pid"); do
                [[ " $EXTRA " == *" $anc "* ]] && continue
                echo "[safe_reset] python ancestor of $pid:"
                ps -o pid,user,etime,comm,args -p "$anc" 2>/dev/null || continue
                read -rp "Also kill $anc? [y/N] " ans
                [[ "${ans,,}" =~ ^y(es)?$ ]] && EXTRA+="$anc "
            done
        done
    fi
    TARGETS="$H $EXTRA"
    echo "[safe_reset] SIGTERM: $TARGETS"
    kill -TERM $TARGETS 2>/dev/null || true
    sleep 3
    R=""
    for p in $TARGETS; do kill -0 "$p" 2>/dev/null && R+="$p "; done
    [[ -n "$R" ]] && { echo "[safe_reset] SIGKILL: $R"; kill -KILL $R 2>/dev/null || true; }
fi

for _ in $(seq 1 10); do [[ -z "$(holders)" ]] && break; sleep 1; done
[[ -n "$(holders)" ]] && { echo "ERROR: device still held; refusing to reset." >&2; exit 5; }

echo "[safe_reset] tt-smi -r"
tt-smi -r
tt-smi -ls 2>/dev/null || tt-smi -s | head -25
echo "[safe_reset] OK."
