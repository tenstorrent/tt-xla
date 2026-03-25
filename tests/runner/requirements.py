# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import fcntl
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
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
    - Also looks for system-requirements.txt for system packages (e.g. ffmpeg)
    """

    # JAX test infra imports flax/transformers at module level; purging them
    # from sys.modules would break isinstance checks between old class objects
    # held by module-level variables (e.g. nnx.Module) and freshly loaded ones.
    # Entries must be import names (e.g. "PIL", not "Pillow"), since they are
    # compared against resolved import names in _purge_stale_modules.
    _JAX_PURGE_SKIP = frozenset({"flax", "transformers"})

    # Top-level directory names in RECORD that are not importable packages.
    _RECORD_SKIP = frozenset({"__pycache__", "bin", "share"})

    # Golden snapshot of the pip environment captured once per session.
    # Each forked child inherits this from the parent process.  At the
    # start of __enter__, the current on-disk state is compared against
    # the golden snapshot; if they differ (e.g. a previous child was
    # OOM-killed before __exit__ could roll back), the environment is
    # restored before proceeding.
    _golden_freeze: Optional[Dict[str, str]] = None

    @classmethod
    def capture_golden_state(cls) -> None:
        """Capture the clean pip environment in memory.  Called once per session.

        Guarded so that forked children (pytest-forked) do not re-capture:
        with --forked, session fixtures re-run inside each child process.
        Without this guard the child would overwrite the inherited golden
        freeze with the (possibly dirty) on-disk state.
        """
        if cls._golden_freeze is not None:
            return
        cls._golden_freeze = cls._pip_freeze()
        _dbg(
            f"[Requirements] Golden freeze captured: "
            f"{len(cls._golden_freeze)} packages"
        )

    @classmethod
    def _restore_environment(cls) -> None:
        """Restore the pip environment to the golden state held in memory."""
        if cls._golden_freeze is None:
            _dbg("[Requirements] No golden freeze captured; cannot heal")
            return

        _dbg("[Requirements] Restoring environment from golden freeze")

        current_freeze = cls._pip_freeze()

        # Packages not in the golden state — must be removed
        to_uninstall = sorted(
            set(current_freeze.keys()) - set(cls._golden_freeze.keys())
        )

        # Packages whose version changed or that went missing
        to_restore = []
        for name, line in cls._golden_freeze.items():
            if name not in current_freeze or current_freeze[name] != line:
                to_restore.append(line)

        if not to_uninstall and not to_restore:
            _dbg("[Requirements] Environment is clean; nothing to heal")
            return

        if to_uninstall:
            _dbg(f"[Requirements] Restoring: uninstalling {to_uninstall}")
            try:
                cls._pip_uninstall(to_uninstall)
            except Exception as e:
                warnings.warn(
                    f"[Requirements] Restoring: uninstall failed: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if to_restore:
            _dbg(f"[Requirements] Restoring: restoring {len(to_restore)} packages")
            # Install one-by-one so that a single unresolvable package
            # does not block the entire restore.
            for spec in to_restore:
                try:
                    cls._pip_install(tuple([spec]))
                except Exception as e:
                    warnings.warn(
                        f"[Requirements] Restoring: failed to install {spec}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        _dbg("[Requirements] Environment restored")

    @classmethod
    def check_and_restore_environment(cls) -> None:
        """Public entry point: restore pip env to golden state if dirty.

        Called before every test (via a pytest fixture) so that even tests
        without their own requirements.txt benefit from crash recovery.
        """
        if cls._golden_freeze is not None:
            cls._restore_environment()

    def __init__(
        self, requirements_path: Optional[str], framework: Optional[str] = None
    ) -> None:
        self.requirements_path = (
            requirements_path
            if requirements_path and os.path.isfile(requirements_path)
            else None
        )
        self.system_requirements_path = None
        if self.requirements_path:
            # Look for system-requirements.txt in the same directory
            sys_req_path = os.path.join(
                os.path.dirname(self.requirements_path), "system-requirements.txt"
            )
            if os.path.isfile(sys_req_path):
                self.system_requirements_path = sys_req_path

        self._framework = framework.lower() if framework else None
        self._before_freeze: Dict[str, str] = {}
        self._after_freeze: Dict[str, str] = {}
        self._newly_installed: Set[str] = set()
        self._changed_versions: Dict[str, str] = {}
        # Cached dist-name → import-name mapping, populated in _compute_diffs
        # while packages are still installed so __exit__ can use it after uninstall.
        self._import_names_cache: Dict[str, Set[str]] = {}
        self._lock_file = None
        self._system_packages_installed: Set[str] = set()

    @staticmethod
    def for_loader(
        loader_path: str, framework: Optional[str] = None
    ) -> "RequirementsManager":
        loader_dir = os.path.dirname(os.path.abspath(loader_path))
        req_path = os.path.join(loader_dir, "requirements.txt")
        return RequirementsManager(req_path, framework=framework)

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

        # Self-healing: compare the on-disk pip state against the golden
        # snapshot captured at session start.  If they differ, a previous
        # forked child was likely killed (e.g. OOM) before __exit__ could
        # roll back.  Restore the clean state before proceeding.
        if self._golden_freeze is not None:
            self._restore_environment()

        # Install system requirements first (if any)
        if self.system_requirements_path:
            _dbg(
                f"[Requirements] __enter__: installing system packages from {self.system_requirements_path}"
            )
            self._install_system_requirements(self.system_requirements_path)

        # Check for uninstall_first.txt to uninstall incompatible packages
        uninstall_path = os.path.join(
            os.path.dirname(self.requirements_path), "uninstall_first.txt"
        )
        if os.path.isfile(uninstall_path):
            _dbg(
                f"[Requirements] __enter__: uninstalling packages from {uninstall_path}"
            )
            with open(uninstall_path, "r") as f:
                packages_to_uninstall = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            if packages_to_uninstall:
                self._pip_uninstall(packages_to_uninstall)

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

        # Optional: a sibling file 'requirements.nodeps.nobuildisolation.txt' for packages to install without dependencies and isolated environment
        nodeps_no_build_isolation_path = os.path.join(
            os.path.dirname(self.requirements_path),
            "requirements.nodeps.nobuildisolation.txt",
        )
        if os.path.isfile(nodeps_no_build_isolation_path):
            _dbg(
                f"[Requirements] __enter__: installing (no-deps and no-build-isolation) -r {nodeps_no_build_isolation_path}"
            )
            self._pip(
                (
                    "install",
                    "--no-input",
                    "--no-deps",
                    "--no-build-isolation",
                    "-r",
                    nodeps_no_build_isolation_path,
                )
            )

        _dbg("[Requirements] __enter__: running pip freeze (after)")
        self._after_freeze = self._pip_freeze()
        self._compute_diffs()
        _dbg(
            f"[Requirements] __enter__: newly_installed={sorted(self._newly_installed)}"
        )
        _dbg(
            f"[Requirements] __enter__: changed_versions={sorted(self._changed_versions.keys())}"
        )
        self._purge_stale_modules()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if not self.requirements_path:
                return

            if os.environ.get(DISABLE_ENV, "0") == "1":
                return

            # Each rollback step is independently guarded so a failure in one
            # (e.g. network error during uninstall) doesn't prevent the others.
            if self._newly_installed:
                to_remove = sorted(self._newly_installed)
                _dbg(f"[Requirements] __exit__: uninstalling: {to_remove}")
                try:
                    self._pip_uninstall(to_remove)
                except Exception as e:
                    warnings.warn(
                        f"[Requirements] __exit__: uninstall failed: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if self._changed_versions:
                _dbg(
                    f"[Requirements] __exit__: restoring versions: {sorted(self._changed_versions.keys())}"
                )
                restore_file = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".txt",
                        delete=False,
                        prefix="tt_xla_restore_",
                    ) as f:
                        for name in sorted(self._changed_versions):
                            f.write(self._changed_versions[name] + "\n")
                        restore_file = f.name
                    self._pip_install_requirements(restore_file)
                except Exception as e:
                    warnings.warn(
                        f"[Requirements] __exit__: version restore failed: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                finally:
                    if restore_file and os.path.isfile(restore_file):
                        os.unlink(restore_file)

            self._purge_stale_modules()
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

        # Resolve dist→import names now while all packages are still installed.
        # During __exit__ some will have been uninstalled so metadata is gone.
        self._import_names_cache = {}
        for name in self._newly_installed | set(self._changed_versions.keys()):
            self._import_names_cache[name] = self._dist_to_import_names(name)

        _dbg(
            f"[Requirements] _compute_diffs: +{len(self._newly_installed)} new, ~{len(self._changed_versions)} changed"
        )
        _dbg(
            f"[Requirements] _compute_diffs: dist->import mapping={{{', '.join(f'{k}: {sorted(v)}' for k, v in sorted(self._import_names_cache.items()))}}}"
        )

    @staticmethod
    def _dist_to_import_names(dist_name: str) -> Set[str]:
        """Resolve a distribution name to its top-level import package names.

        Uses ``importlib.metadata`` to read the distribution's ``top_level.txt``
        so that packages like ``Pillow`` resolve to ``PIL``, ``scikit-learn`` to
        ``sklearn``, etc.  When ``top_level.txt`` is absent (e.g. ``PyYAML``),
        falls back to scanning installed file paths from the distribution's
        ``RECORD``.  Returns the normalised distribution name only when
        metadata is entirely unavailable (e.g. package already uninstalled).
        """
        normalized_fallback = dist_name.lower().replace("-", "_")
        try:
            dist = importlib.metadata.distribution(dist_name)

            top_level = dist.read_text("top_level.txt")
            if top_level:
                names = {
                    n.strip().lower().replace("-", "_")
                    for n in top_level.splitlines()
                    if n.strip()
                }
                if names:
                    return names

            # Fallback: scan RECORD (pip's installed-file manifest) for
            # top-level package dirs/modules.  Needed when top_level.txt is
            # absent (e.g. scikit-image ships RECORD but no top_level.txt).
            if dist.files:
                names = set()
                for f in dist.files:
                    path_str = str(f)
                    parts = path_str.split("/")
                    # Single-file module (e.g. "six.py" → import name "six")
                    if len(parts) == 1 and path_str.endswith(".py"):
                        mod = path_str[:-3]  # strip ".py" (3 chars)
                        if mod.replace("_", "").isalnum():
                            names.add(mod.lower().replace("-", "_"))
                    # Package directory (e.g. "skimage/__init__.py" → "skimage")
                    elif (
                        len(parts) > 1
                        and parts[0] not in RequirementsManager._RECORD_SKIP
                        and not parts[0].endswith((".dist-info", ".data"))
                        and parts[0].replace("_", "").isalnum()
                    ):
                        names.add(parts[0].lower().replace("-", "_"))
                if names:
                    return names

            _dbg(
                f"[Requirements] WARNING: distribution '{dist_name}' is installed but "
                f"has no top_level.txt and no usable RECORD entries; falling back to "
                f"normalised name '{normalized_fallback}'. sys.modules purge may be "
                f"incomplete if the import name differs."
            )
        except importlib.metadata.PackageNotFoundError:
            _dbg(
                f"[Requirements] WARNING: distribution '{dist_name}' not found in "
                f"metadata (already uninstalled?); falling back to normalised name "
                f"'{normalized_fallback}'."
            )
        return {normalized_fallback}

    def _purge_stale_modules(self) -> None:
        """Remove changed/new packages from sys.modules so re-imports load from disk.

        After pip install or rollback, cached module objects in sys.modules may
        point to the old version's code.  Purging forces Python to re-import
        from the updated on-disk packages the next time they are imported.

        For JAX, ``flax`` and ``transformers`` are excluded from purging because
        the JAX test infrastructure imports them at module level during test
        collection.  Purging them would create a mismatch between the old class
        objects held by module-level variables (e.g. ``nnx.Module``) and the
        freshly loaded ones, breaking ``isinstance`` checks.  Keeping them in
        ``sys.modules`` preserves the same behaviour as before sys-module
        purging was introduced — the old versions remain in memory and are used
        consistently by both the tester and the model.
        """
        affected_normalized: Set[str] = set()
        for name in self._newly_installed | set(self._changed_versions.keys()):
            affected_normalized.update(self._import_names_cache[name])

        if not affected_normalized:
            return

        skip = self._JAX_PURGE_SKIP if self._framework == "jax" else frozenset()
        if skip:
            skipped = affected_normalized & skip
            if skipped:
                _dbg(
                    f"[Requirements] JAX framework: skipping purge for {sorted(skipped)}"
                )
            affected_normalized = affected_normalized - skip

        if not affected_normalized:
            return

        purged = []
        for key in list(sys.modules.keys()):
            top_level = key.split(".")[0].lower().replace("-", "_")
            if top_level in affected_normalized:
                self._unregister_arrow_ext_types(sys.modules[key])
                purged.append(key)
                del sys.modules[key]

        if purged:
            _dbg(
                f"[Requirements] purged {len(purged)} stale module(s) from sys.modules"
            )

        importlib.invalidate_caches()

    @staticmethod
    def _unregister_arrow_ext_types(module) -> None:
        """Unregister any pyarrow extension types defined in *module*.

        pyarrow's C-level type registry survives sys.modules purges.
        Without this, re-importing the module would re-register its types
        and raise ArrowKeyError.
        """
        pa = sys.modules.get("pyarrow")
        if pa is None:
            return
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, pa.ExtensionType)
                and attr is not pa.ExtensionType
            ):
                name = f"{attr.__module__}.{attr.__qualname__}"
                try:
                    pa.unregister_extension_type(name)
                except (pa.lib.ArrowKeyError, KeyError):
                    pass

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
        """Parse ``pip freeze`` output into {normalised_name: full_freeze_line}.

        Handles all install formats: ``name==version``, ``name @ URL``,
        and ``-e ...#egg=name``.  Storing the full freeze line allows the
        rollback to use ``pip install -r`` which understands all these formats.
        """
        result: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            name = None
            if line.startswith("-e "):
                if "#egg=" in line:
                    name = line.split("#egg=")[-1].strip()
            elif "@" in line and "==" not in line:
                name = line.split("@")[0].strip()
            elif "==" in line:
                name = line.split("==", 1)[0].strip()

            if name:
                result[name.lower()] = line
        return result

    def _install_system_requirements(self, system_req_path: str) -> None:
        """Install system packages from system-requirements.txt using apt-get.

        Args:
            system_req_path: Path to system-requirements.txt file
        """
        # Read the packages from the file
        packages = []
        try:
            with open(system_req_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        packages.append(line)
        except Exception as e:
            _dbg(f"[Requirements] Failed to read {system_req_path}: {e}")
            return

        if not packages:
            return

        # Check if packages are already installed
        packages_to_install = []
        for pkg in packages:
            if not self._is_system_package_installed(pkg):
                packages_to_install.append(pkg)
            else:
                _dbg(
                    f"[Requirements] System package '{pkg}' already installed, skipping"
                )

        if not packages_to_install:
            return

        # Install packages using apt-get
        _dbg(f"[Requirements] Installing system packages: {packages_to_install}")
        try:
            # Update package list first
            subprocess.run(
                ["sudo", "apt-get", "update", "-qq"],
                check=True,
                capture_output=True,
            )

            # Install packages
            if os.geteuid() != 0 and not shutil.which("sudo"):
                _dbg("Skipping system install: no root access or sudo")
                return

            cmd = ["sudo", "apt-get", "install", "-y", "-qq"] + packages_to_install
            subprocess.run(cmd, check=True, capture_output=True)

            # Track what was installed
            self._system_packages_installed.update(packages_to_install)
            _dbg(
                f"[Requirements] Successfully installed system packages: {packages_to_install}"
            )
        except subprocess.CalledProcessError as e:
            _dbg(
                f"[Requirements] Failed to install system packages: {e.stderr.decode() if e.stderr else str(e)}"
            )
            # Don't fail the test if system packages can't be installed
            # The test might still work if the package was already available

    @staticmethod
    def _is_system_package_installed(package: str) -> bool:
        """Check if a system package is already installed using dpkg.

        Args:
            package: Name of the package to check

        Returns:
            True if package is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ["dpkg", "-s", package],
                capture_output=True,
                text=True,
                check=False,
            )
            # dpkg -s returns 0 if package is installed
            return result.returncode == 0
        except Exception:
            return False
