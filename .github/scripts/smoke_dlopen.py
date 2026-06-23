#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Post-repair smoke test for the pjrt_plugin_tt wheel.

dlopen every shipped pjrt_plugin_tt shared library in an isolated subprocess
and fail if any crashes with SIGSEGV. This catches ELF corruption -- e.g. a
patchelf that mangles the compact single-RX-segment (rosegment) layout clang/
lld emit, pulling executable .init/.plt into a non-executable RW segment so the
plugin segfaults (SEGV_ACCERR) the moment the dynamic loader maps it.

It needs NO Tenstorrent hardware: the plugin only touches the device at PJRT
client init (jax.devices('tt')), not at dlopen time, so a healthy library maps
cleanly here while a corrupt one faults during relocation. A non-SIGSEGV
failure (e.g. an unresolved symbol when a library is loaded in isolation) is
benign and ignored -- only SIGSEGV indicates corruption.
"""

import os
import subprocess
import sys
import sysconfig

SIGSEGV_RETURNCODES = (-11, 139)


def find_libs():
    purelib = sysconfig.get_paths()["purelib"]
    roots = [
        os.path.join(purelib, "pjrt_plugin_tt"),
        os.path.join(purelib, "pjrt_plugin_tt.libs"),
    ]
    libs = set()
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if ".so" not in name:
                    continue
                path = os.path.join(dirpath, name)
                if not os.path.islink(path):
                    libs.add(path)
    return sorted(libs)


def main():
    libs = find_libs()
    if not libs:
        print("ERROR: no pjrt_plugin_tt shared libraries found", file=sys.stderr)
        return 1

    segv = []
    for lib in libs:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes, sys; ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)",
                lib,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode in SIGSEGV_RETURNCODES:
            segv.append(lib)
            print(f"SEGV  {lib}")
        else:
            print(f"ok    {lib} (rc={result.returncode})")

    if segv:
        print(
            f"\nFAILED: {len(segv)} of {len(libs)} libraries segfaulted on dlopen -- "
            "ELF corruption (check the patchelf used during auditwheel repair).",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: all {len(libs)} libraries dlopen cleanly (no SIGSEGV).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
