# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def get_file(path):
    rel_dir, file_name = os.path.split(path)
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = (
            Path(os.environ["DOCKER_CACHE_ROOT"])
            / "models/tt-ci-models-private"
            / rel_dir
        )
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_dir
    else:
        cache_dir = Path.home() / ".cache/lfcache" / rel_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / file_name
    if not file_path.exists():
        if "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path
