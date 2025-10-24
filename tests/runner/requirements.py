# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import fcntl
import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Set, Tuple

# Debug flag: set TT_XLA_REQS_DEBUG=1 to see detailed output
DEBUG_ENV = "TT_XLA_REQS_DEBUG"
DISABLE_ENV = "TT_XLA_DISABLE_MODEL_REQS"


def _dbg(msg: str) -> None:
    if os.environ.get(DEBUG_ENV, "0") == "1":
        print(msg, flush=True)


class RequirementsManager:
    """Context manager to temporarily install per-model requirements and roll back.

    - Looks for a requirements.txt next to a given loader.py
    - Freezes the current environment
    - Installs required packages (supports an optional 'requirements.nodeps.txt' installed with --no-deps)
    - On exit, uninstalls newly added packages and restores changed versions
    - Uses a global file lock to serialize pip operations
    """

    def __init__(self, requirements_path: Optional[str]) -> None:
        self.requirements_path = (
            requirements_path
            if requirements_path and os.path.isfile(requirements_path)
            else None
        )
        self._before_freeze: Dict[str, str] = {}
        self._after_freeze: Dict[str, str] = {}
        self._newly_installed: Set[str] = set()
        self._changed_versions: Dict[str, str] = {}
        self._lock_file = None

    @staticmethod
    def for_loader(loader_path: str) -> "RequirementsManager":
        loader_dir = os.path.dirname(os.path.abspath(loader_path))
        req_path = os.path.join(loader_dir, "requirements.txt")
        return RequirementsManager(req_path)

    def __enter__(self) -> "RequirementsManager":
        if not self.requirements_path:
            return self

        if os.environ.get(DISABLE_ENV, "0") == "1":
            return self

        # Acquire a global lock for pip operations
        lock_path = os.path.join(
            tempfile.gettempdir(), "tt_xla_model_requirements.lock"
        )
        self._lock_file = open(lock_path, "w")
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)

        self._before_freeze = self._pip_freeze()
        _dbg(f"[Requirements] __enter__: installing -r {self.requirements_path}")
        self._pip_install_requirements(self.requirements_path)

        # Optional: a sibling file 'requirements.nodeps.txt' for packages to install without dependencies
        nodeps_path = os.path.join(
            os.path.dirname(self.requirements_path), "requirements.nodeps.txt"
        )
        if os.path.isfile(nodeps_path):
            _dbg(f"[Requirements] __enter__: installing (no-deps) -r {nodeps_path}")
            self._pip(("install", "--no-input", "--no-deps", "-r", nodeps_path))

        _dbg("[Requirements] __enter__: running pip freeze (after)")
        self._after_freeze = self._pip_freeze()
        self._compute_diffs()
        _dbg(
            f"[Requirements] __enter__: newly_installed={sorted(self._newly_installed)}"
        )
        _dbg(f"[Requirements] __enter__: changed_versions={self._changed_versions}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if not self.requirements_path:
                return

            if os.environ.get(DISABLE_ENV, "0") == "1":
                return

            # Uninstall newly installed packages
            if self._newly_installed:
                to_remove = sorted(self._newly_installed)
                _dbg(f"[Requirements] __exit__: uninstalling: {to_remove}")
                self._pip_uninstall(to_remove)

            # Restore original versions for packages that changed
            if self._changed_versions:
                pinned = [
                    f"{name}=={version}"
                    for name, version in sorted(self._changed_versions.items())
                ]
                _dbg(f"[Requirements] __exit__: restoring versions: {pinned}")
                self._pip_install(tuple(pinned))
        finally:
            # Always release the lock if held
            if self._lock_file is not None:
                try:
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                finally:
                    self._lock_file.close()
                    self._lock_file = None

    def _compute_diffs(self) -> None:
        before_keys = set(self._before_freeze.keys())
        after_keys = set(self._after_freeze.keys())

        self._newly_installed = after_keys - before_keys

        changed = {}
        for name in before_keys & after_keys:
            if self._before_freeze[name] != self._after_freeze[name]:
                changed[name] = self._before_freeze[name]
        self._changed_versions = changed
        _dbg(
            f"[Requirements] _compute_diffs: +{len(self._newly_installed)} new, ~{len(self._changed_versions)} changed"
        )

    @staticmethod
    def _pip(args: Tuple[str, ...]) -> None:
        cmd = (sys.executable, "-m", "pip") + args
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        _dbg(f"[Requirements] pip: {' '.join(map(str, cmd))}")
        subprocess.run(
            " ".join(map(str, cmd)),
            shell=True,
            check=True,
            env=env,
        )

    @classmethod
    def _pip_install_requirements(cls, requirements_path: str) -> None:
        cls._pip(
            (
                "install",
                "--no-input",
                "-r",
                requirements_path,
            )
        )

    @classmethod
    def _pip_install(cls, specs: Tuple[str, ...]) -> None:
        if not specs:
            return
        args = ("install", "--no-input") + specs
        cls._pip(args)

    @classmethod
    def _pip_uninstall(cls, packages: Tuple[str, ...] | list[str]) -> None:
        if not packages:
            return
        args = ("uninstall", "-y") + tuple(packages)
        cls._pip(args)

    @staticmethod
    def _pip_freeze() -> Dict[str, str]:
        cmd = (sys.executable, "-m", "pip", "freeze")
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        _dbg("[Requirements] pip freeze")
        proc = subprocess.run(
            " ".join(map(str, cmd)),
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return RequirementsManager._parse_freeze(proc.stdout)

    @staticmethod
    def _parse_freeze(text: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line and not line.startswith("-e ") and "@" not in line:
                try:
                    name, version = line.split("==", 1)
                    result[name.strip().lower()] = version.strip()
                except ValueError:
                    continue
        return result
