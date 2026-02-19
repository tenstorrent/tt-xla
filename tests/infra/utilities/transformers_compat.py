# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compatibility shims for newer transformers versions.

Import this module early (before any model code) to patch removed APIs
that third-party libraries (e.g. EasyDel) still depend on.
"""

import transformers.utils
import transformers.utils.generic


def _ensure_download_url():
    """Provide a stub for `transformers.utils.download_url` if it was removed."""
    if hasattr(transformers.utils, "download_url"):
        return

    import tempfile

    import requests

    def download_url(url, proxies=None):
        """Minimal replacement for the removed transformers.utils.download_url."""
        response = requests.get(url, proxies=proxies)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(response.content)
        tmp.close()
        return tmp.name

    transformers.utils.download_url = download_url


def _ensure_is_remote_url():
    """Provide a stub for `transformers.utils.is_remote_url` if it was removed."""
    if hasattr(transformers.utils, "is_remote_url"):
        return

    def is_remote_url(url_or_filename):
        """Minimal replacement for the removed transformers.utils.is_remote_url."""
        if not isinstance(url_or_filename, str):
            return False
        return url_or_filename.startswith(("http://", "https://", "s3://", "gs://"))

    transformers.utils.is_remote_url = is_remote_url


def _ensure_working_or_temp_dir():
    """Provide a stub for `transformers.utils.generic.working_or_temp_dir` if removed."""
    if hasattr(transformers.utils.generic, "working_or_temp_dir"):
        return

    import os
    import tempfile
    from contextlib import contextmanager

    @contextmanager
    def working_or_temp_dir(working_dir, use_temp_dir=False):
        """Minimal replacement for the removed working_or_temp_dir."""
        if use_temp_dir:
            with tempfile.TemporaryDirectory() as tmp_dir:
                yield tmp_dir
        else:
            yield working_dir if working_dir is not None else os.getcwd()

    transformers.utils.generic.working_or_temp_dir = working_or_temp_dir


def _ensure_set_token_in_kwargs():
    """Restore _set_token_in_kwargs on PretrainedConfig for EasyDel compatibility."""
    from transformers import PretrainedConfig

    if hasattr(PretrainedConfig, "_set_token_in_kwargs"):
        return

    @staticmethod
    def _set_token_in_kwargs(kwargs, token=None):
        if token is None:
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. "
                    "Please set only the argument `token`."
                )
            token = use_auth_token
        if token is not None:
            kwargs["token"] = token

    PretrainedConfig._set_token_in_kwargs = _set_token_in_kwargs


_ensure_download_url()
_ensure_is_remote_url()
_ensure_working_or_temp_dir()
_ensure_set_token_in_kwargs()
