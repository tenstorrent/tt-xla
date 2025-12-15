# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import (
    Optional,
    Union,
    Any,
    List,
    Tuple,
    Generator,
    Iterator,
    Sequence,
    Mapping,
)
from ....tools.utils import get_file
from time import time as ti
from torch.nn.modules.module import Module
import torch.distributed as dist
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from logging import FileHandler, Logger, LogRecord
from collections import defaultdict
import functools
from inspect import getfullargspec
from collections import abc
from shutil import get_terminal_size
from packaging.version import parse
import math
import torch.nn.functional as F
from numbers import Number
from torch import Tensor, BoolTensor
from scipy.stats import truncnorm
import shapely.geometry as geometry
import numbers
import cv2
import itertools
from pathlib import Path
import torch.nn as nn
import warnings
import copy
import torch
import datetime
from addict import Dict
import importlib
import pycocotools.mask as maskUtils
from importlib import import_module
import tempfile
import os
import ast
import os.path as osp
import re
from collections import OrderedDict
import inspect
from typing import Type, TypeVar
import threading
import logging
from torch.optim import Optimizer
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist
import sys
from termcolor import colored
from getpass import getuser
from socket import gethostname
import uuid
import platform
import shutil
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
import json
import numpy as np
import yaml
from collections.abc import Callable, Sized
from rich.console import Console
from rich.table import Table
import types
from argparse import Action, ArgumentParser, Namespace

from yaml import CDumper as Dumper
from yaml import CLoader as Loader
import pickle
from contextlib import contextmanager
from urllib.request import urlopen
from io import BytesIO, StringIO
from yapf.yapflib.yapf_api import FormatCode
import time

try:
    from PIL import Image
except ImportError:
    Image = None
backends: dict = {}
prefix_to_backends: dict = {}
BASE_KEY = "_base_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]
DEPRECATION_KEY = "_deprecation_"
DELETE_KEY = "_delete_"
_lock = threading.RLock()
T = TypeVar("T")
TORCH_VERSION = torch.__version__


def _accquire_lock() -> None:
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    if _lock:
        _lock.release()


class ManagerMeta(type):
    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert "name" in params_names, f"{cls} must have the `name` argument"
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    def __init__(self, name: str = "", **kwargs):
        assert (
            isinstance(name, str) and name
        ), "name argument must be an non-empty string."
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        _accquire_lock()
        assert isinstance(name, str), f"type of name should be str, but got {type(cls)}"
        instance_dict = cls._instance_dict
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)
            instance_dict[name] = instance
        elif kwargs:
            warnings.warn(
                f"{cls} instance named of {name} has been created, "
                "the method `get_instance` should not accept any other "
                "arguments"
            )
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f"Before calling {cls.__name__}.get_current_instance(), you "
                "should call get_instance(name=xxx) at least once."
            )
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        return self._instance_name


def is_distributed() -> bool:
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:

    return torch_dist.distributed_c10d._get_default_group()


def get_rank(group: Optional[ProcessGroup] = None) -> int:

    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def _get_rank():
    return get_rank()


def _get_device_id():
    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(",")
        return int(cuda_visible_devices[local_rank])


class MMFormatter(logging.Formatter):
    _color_mapping: dict = dict(
        ERROR="red", WARNING="yellow", INFO="white", DEBUG="green"
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), "blink should only be available when color is True"
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)
        self.err_format = (
            f"%(asctime)s - %(name)s - {error_prefix} - "
            "%(pathname)s - %(funcName)s - %(lineno)d - "
            "%(message)s"
        )
        self.warn_format = f"%(asctime)s - %(name)s - {warn_prefix} - %(" "message)s"
        self.info_format = f"%(asctime)s - %(name)s - {info_prefix} - %(" "message)s"
        self.debug_format = f"%(asctime)s - %(name)s - {debug_prefix} - %(" "message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class FilterDuplicateWarning(logging.Filter):
    def __init__(self, name: str = "mmengine"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def _get_world_size():
    return get_world_size()


def _get_host_info() -> str:
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {str(e)}")
    finally:
        return host


class MMLogger(Logger, ManagerMixin):
    def __init__(
        self,
        name: str,
        logger_name="mmengine",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
        distributed=False,
    ):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(MMFormatter(color=True, datefmt="%m/%d %H:%M:%S"))
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (
                log_level <= logging.DEBUG or distributed
            ) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = (
                        f"{filename}_{hostname}_device{device_id}_"
                        f"rank{global_rank}{suffix}"
                    )
                else:
                    filename = (
                        f"{filename}_device{device_id}_" f"rank{global_rank}{suffix}"
                    )
                log_file = osp.join(osp.dirname(log_file), filename)
            if global_rank == 0 or is_distributed:
                file_handler = logging.FileHandler(log_file, file_mode)
                file_handler.setFormatter(
                    MMFormatter(color=False, datefmt="%Y/%m/%d %H:%M:%S")
                )
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> "MMLogger":
        if not cls._instance_dict:
            cls.get_instance("mmengine")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        self.level = logging._checkLevel(level)
        _accquire_lock()
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(
    msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO
) -> None:
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"MMLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )


def import_modules_from_strings(imports, allow_failed_imports=False):
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f"Failed to import {imp}")
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            raise e
        else:
            return value


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class RemoveAssignFromAST(ast.NodeTransformer):
    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.key:
            return None
        else:
            return node


class BaseFileHandler(metaclass=ABCMeta):
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


def set_default(obj):
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj)} is unsupported for json dump")


class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)


class JsonHandler(BaseFileHandler):
    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("default", set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("default", set_default)
        return json.dumps(obj, **kwargs)


class PickleHandler(BaseFileHandler):

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super().dump_to_path(obj, filepath, mode="wb", **kwargs)


file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}


def is_str(x):
    return isinstance(x, str)


class BaseStorageBackend(metaclass=ABCMeta):
    _allow_symlink = False

    @property
    def allow_symlink(self):
        print_log(
            "allow_symlink will be deprecated in future",
            logger="current",
            level=logging.WARNING,
        )
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys

            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError("Please install memcached to enable MemcachedBackend.")

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(
            self.server_list_cfg, self.client_cfg
        )
        self._mc_buffer = mc.pyvector()

    def get(self, filepath: Union[str, Path]):
        filepath = str(filepath)
        import mc

        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class LocalBackend(BaseStorageBackend):
    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        with open(filepath, "rb") as f:
            value = f.read()
        return value

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        with open(filepath, encoding=encoding) as f:
            text = f.read()
        return text

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "wb") as f:
            f.write(obj)

    def put_text(
        self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8"
    ) -> None:
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "w", encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        return osp.isfile(filepath)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        yield filepath

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        return shutil.copy(src, dst)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        return shutil.copytree(src, dst)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        return self.copytree(src, dst)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        return self.copyfile(src, dst)

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        return self.copytree(src, dst)

    def remove(self, filepath: Union[str, Path]) -> None:
        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        os.remove(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        shutil.rmtree(dir_path)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            if self.isfile(src):
                self.copyfile(src, dst)
            else:
                self.copytree(src, dst)
            return False

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        if list_dir and suffix is not None:
            raise TypeError("`suffix` should be None when `list_dir` is True")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith(".") and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            entry.path, list_dir, list_file, suffix, recursive
                        )

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)


class HardDiskBackend(LocalBackend):
    def __init__(self) -> None:
        print_log(
            '"HardDiskBackend" is the alias of "LocalBackend" '
            "and the former will be deprecated in future.",
            logger="current",
            level=logging.WARNING,
        )

    @property
    def name(self):
        return self.__class__.__name__


class LmdbBackend(BaseStorageBackend):
    def __init__(self, db_path, readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please run "pip install lmdb" to enable LmdbBackend.')

        self.db_path = str(db_path)
        self.readonly = readonly
        self.lock = lock
        self.readahead = readahead
        self.kwargs = kwargs
        self._client = None

    def get(self, filepath: Union[str, Path]) -> bytes:
        if self._client is None:
            self._client = self._get_client()

        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode("ascii"))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self):
        import lmdb

        return lmdb.open(
            self.db_path,
            readonly=self.readonly,
            lock=self.lock,
            readahead=self.readahead,
            **self.kwargs,
        )

    def __del__(self):
        if self._client is not None:
            self._client.close()


def has_method(obj: object, method: str) -> bool:
    return hasattr(obj, method) and callable(getattr(obj, method))


class PetrelBackend(BaseStorageBackend):
    def __init__(
        self,
        path_mapping: Optional[dict] = None,
        enable_mc: bool = True,
        conf_path: Optional[str] = None,
    ):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError(
                "Please install petrel_client to enable " "PetrelBackend."
            )

        self._client = client.Client(conf_path=conf_path, enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        return re.sub(r"\\+", "/", filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath.replace("petrel://", "s3://")

    def get(self, filepath: Union[str, Path]) -> bytes:
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.Get(filepath)
        return value

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> str:
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.put(filepath, obj)

    def put_text(
        self,
        obj: str,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> None:
        self.put(bytes(obj, encoding=encoding), filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        if not (
            has_method(self._client, "contains") and has_method(self._client, "isdir")
        ):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        if not has_method(self._client, "isdir"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `isdir` method, please use a higher version or dev"
                " branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        if not has_method(self._client, "contains"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` method, please use a higher version or "
                "dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip("/"))

        return "/".join(formatted_paths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, src.split("/")[-1])

        if src == dst:
            raise SameFileError("src and dst should not be same")

        self.put(self.get(src), dst)
        return dst

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))

        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(self.get(src_path), dst_path)

        return dst

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, osp.basename(src))

        with open(src, "rb") as f:
            self.put(f.read(), dst)

        return dst

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = osp.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ""))
                self.copyfile_from_local(src_path, dst_path)

        return dst

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        if osp.isdir(dst):
            basename = osp.basename(src)
            if isinstance(dst, str):
                dst = osp.join(dst, basename)
            else:
                assert isinstance(dst, Path)
                dst = dst / basename

        with open(dst, "wb") as f:
            f.write(self.get(src))

        return dst

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            dst_path = osp.join(dst, path)
            mmengine.mkdir_or_exist(osp.dirname(dst_path))
            with open(dst_path, "wb") as f:
                f.write(self.get(self.join_path(src, path)))

        return dst

    def remove(self, filepath: Union[str, Path]) -> None:
        if not has_method(self._client, "delete"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `delete` method, please use a higher version or dev "
                "branch instead."
            )

        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.delete(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        for path in self.list_dir_or_file(dir_path, list_dir=False, recursive=True):
            filepath = self.join_path(dir_path, path)
            self.remove(filepath)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        if self.isfile(src):
            self.copyfile(src, dst)
        else:
            self.copytree(src, dst)
        return False

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        if not has_method(self._client, "list"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `list` method, please use a higher version or dev"
                " branch instead."
            )

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError("`list_dir` should be False when `suffix` is not None")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        if not dir_path.endswith("/"):
            dir_path += "/"

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for path in self._client.list(dir_path):
                if path.endswith("/"):
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        rel_dir = next_dir_path[len(root) : -1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            next_dir_path, list_dir, list_file, suffix, recursive
                        )
                else:
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root) :]
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)

    def generate_presigned_url(
        self, url: str, client_method: str = "get_object", expires_in: int = 3600
    ) -> str:
        return self._client.generate_presigned_url(url, client_method, expires_in)


class HTTPBackend(BaseStorageBackend):
    def get(self, filepath: str) -> bytes:
        return urlopen(filepath).read()

    def get_text(self, filepath, encoding="utf-8") -> str:
        return urlopen(filepath).read().decode(encoding)

    @contextmanager
    def get_local_path(self, filepath: str) -> Generator[Union[str, Path], None, None]:
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def _get_file_backend(prefix: str, backend_args: dict):
    if "backend" in backend_args:
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop("backend")
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


class FileClient:
    _backends = {
        "disk": HardDiskBackend,
        "memcached": MemcachedBackend,
        "lmdb": LmdbBackend,
        "petrel": PetrelBackend,
        "http": HTTPBackend,
    }

    _prefix_to_backends: dict = {
        "s3": PetrelBackend,
        "petrel": PetrelBackend,
        "http": HTTPBackend,
        "https": HTTPBackend,
    }

    _instances: dict = {}

    client: Any

    def __new__(cls, backend=None, prefix=None, **kwargs):
        print_log(
            '"FileClient" will be deprecated in future. Please use io '
            "functions in "
            "https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io",
            logger="current",
            level=logging.WARNING,
        )
        if backend is None and prefix is None:
            backend = "disk"
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones"
                f" are {list(cls._backends.keys())}"
            )
        if prefix is not None and prefix not in cls._prefix_to_backends:
            raise ValueError(
                f"prefix {prefix} is not supported. Currently supported ones "
                f"are {list(cls._prefix_to_backends.keys())}"
            )

        arg_key = f"{backend}:{prefix}"
        for key, value in kwargs.items():
            arg_key += f":{key}:{value}"
        if arg_key in cls._instances:
            _instance = cls._instances[arg_key]
        else:
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
            else:
                _instance.client = cls._prefix_to_backends[prefix](**kwargs)

            cls._instances[arg_key] = _instance

        return _instance

    @property
    def name(self):
        return self.client.name

    @property
    def allow_symlink(self):
        return self.client.allow_symlink

    @staticmethod
    def parse_uri_prefix(uri: Union[str, Path]) -> Optional[str]:
        assert is_filepath(uri)
        uri = str(uri)
        if "://" not in uri:
            return None
        else:
            prefix, _ = uri.split("://")
            if ":" in prefix:
                _, prefix = prefix.split(":")
            return prefix

    @classmethod
    def infer_client(
        cls,
        file_client_args: Optional[dict] = None,
        uri: Optional[Union[str, Path]] = None,
    ) -> "FileClient":
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            file_prefix = cls.parse_uri_prefix(uri)
            return cls(prefix=file_prefix)
        else:
            return cls(**file_client_args)

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError(
                "the backend name should be a string, " f"but got {type(name)}"
            )
        if not inspect.isclass(backend):
            raise TypeError(f"backend should be a class but got {type(backend)}")
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f"backend {backend} is not a subclass of BaseStorageBackend"
            )
        if not force and name in cls._backends:
            raise KeyError(
                f"{name} is already registered as a storage backend, "
                'add "force=True" if you want to override it'
            )

        if name in cls._backends and force:
            for arg_key, instance in list(cls._instances.items()):
                if isinstance(instance.client, cls._backends[name]):
                    cls._instances.pop(arg_key)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    overridden_backend = cls._prefix_to_backends[prefix]
                    for arg_key, instance in list(cls._instances.items()):
                        if isinstance(instance.client, overridden_backend):
                            cls._instances.pop(arg_key)
                else:
                    raise KeyError(
                        f"{prefix} is already registered as a storage backend,"
                        ' add "force=True" if you want to override it'
                    )

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        if backend is not None:
            cls._register_backend(name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding="utf-8") -> str:
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        return self.client.isfile(filepath)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        return self.client.join_path(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self, filepath: Union[str, Path]
    ) -> Generator[Union[str, Path], None, None]:
        with self.client.get_local_path(str(filepath)) as local_path:
            yield local_path

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        yield from self.client.list_dir_or_file(
            dir_path, list_dir, list_file, suffix, recursive
        )


def _register_backend(
    name: str,
    backend: Type[BaseStorageBackend],
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    global backends, prefix_to_backends

    if not isinstance(name, str):
        raise TypeError("the backend name should be a string, " f"but got {type(name)}")

    if not inspect.isclass(backend):
        raise TypeError(f"backend should be a class, but got {type(backend)}")
    if not issubclass(backend, BaseStorageBackend):
        raise TypeError(f"backend {backend} is not a subclass of BaseStorageBackend")

    if name in backends and not force:
        raise ValueError(
            f"{name} is already registered as a storage backend, "
            'add "force=True" if you want to override it'
        )
    backends[name] = backend

    if prefixes is not None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))

        for prefix in prefixes:
            if prefix in prefix_to_backends and not force:
                raise ValueError(
                    f"{prefix} is already registered as a storage backend,"
                    ' add "force=True" if you want to override it'
                )

            prefix_to_backends[prefix] = backend


def register_backend(
    name: str,
    backend: Optional[Type[BaseStorageBackend]] = None,
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    if backend is not None:
        _register_backend(name, backend, force=force, prefixes=prefixes)
        return

    def _register(backend_cls):
        _register_backend(name, backend_cls, force=force, prefixes=prefixes)
        return backend_cls

    return _register


register_backend("local", LocalBackend, prefixes="")
register_backend("memcached", MemcachedBackend)
register_backend("lmdb", LmdbBackend)
register_backend("petrel", PetrelBackend, prefixes=["petrel", "s3"])
register_backend("http", HTTPBackend, prefixes=["http", "https"])


def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    assert is_filepath(uri)
    uri = str(uri)
    if "://" not in uri:
        return ""
    else:
        prefix, _ = uri.split("://")
        if ":" in prefix:
            _, prefix = prefix.split(":")
        return prefix


backend_instances: dict = {}


def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and "backend" not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in ' "backend_args"
        )

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ""

    if enable_singleton:
        unique_key = f"{prefix}:{json.dumps(backend_args)}"
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def load(file, file_format=None, file_client_args=None, backend_args=None, **kwargs):
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args and "backend_args" cannot be set at the '
                "same time."
            )

    handler = file_handlers[file_format]
    if is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_backend.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


MODULE2PACKAGE = {
    "mmcls": "mmcls",
    "mmdet": "mmdet",
    "mmdet3d": "mmdet3d",
    "mmseg": "mmsegmentation",
    "mmaction": "mmaction2",
    "mmtrack": "mmtrack",
    "mmpose": "mmpose",
    "mmedit": "mmedit",
    "mmocr": "mmocr",
    "mmgen": "mmgen",
    "mmfewshot": "mmfewshot",
    "mmrazor": "mmrazor",
    "mmflow": "mmflow",
    "mmhuman3d": "mmhuman3d",
    "mmrotate": "mmrotate",
    "mmselfsup": "mmselfsup",
    "mmyolo": "mmyolo",
    "mmpretrain": "mmpretrain",
}


def _get_package_and_cfg_path(cfg_path: str) -> Tuple[str, str]:
    if re.match(r"\w*::\w*/\w*", cfg_path) is None:
        raise ValueError(
            "`_get_package_and_cfg_path` is used for get external package, "
            "please specify the package name and relative config path, just "
            "like `mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`"
        )
    package_cfg = cfg_path.split("::")
    if len(package_cfg) > 2:
        raise ValueError(
            "`::` should only be used to separate package and "
            "config name, but found multiple `::` in "
            f"{cfg_path}"
        )
    package, cfg_path = package_cfg
    assert (
        package in MODULE2PACKAGE
    ), f"mmengine does not support to load {package} config."
    package = MODULE2PACKAGE[package]
    return package, cfg_path


def is_installed(package: str) -> bool:
    import pkg_resources
    from pkg_resources import get_distribution

    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def package2module(package: str):
    from pkg_resources import get_distribution

    pkg = get_distribution(package)
    if pkg.has_metadata("top_level.txt"):
        module_name = pkg.get_metadata("top_level.txt").split("\n")[0]
        return module_name
    else:
        raise ValueError(f"can not infer the module name of {package}")


def get_installed_path(package: str) -> str:
    from pkg_resources import get_distribution

    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))


def _get_cfg_metainfo(package_path: str, cfg_path: str) -> dict:
    meta_index_path = osp.join(package_path, ".mim", "model-index.yml")
    meta_index = load(meta_index_path)
    cfg_dict = dict()
    for meta_path in meta_index["Import"]:
        meta_path = osp.join(package_path, ".mim", meta_path)
        cfg_meta = load(meta_path)
        for model_cfg in cfg_meta["Models"]:
            if "Config" not in model_cfg:
                warnings.warn(f"There is not `Config` define in {model_cfg}")
                continue
            cfg_name = model_cfg["Config"].partition("/")[-1]
            if cfg_name in cfg_dict:
                continue
            cfg_dict[cfg_name] = model_cfg
    if cfg_path not in cfg_dict:
        raise ValueError(f"Expected configs: {cfg_dict.keys()}, but got " f"{cfg_path}")
    return cfg_dict[cfg_path]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    cfg_file = cfg_file.split(".")[0]
    model_cfg = _get_cfg_metainfo(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg["Config"])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str) -> str:
    cfg_path = osp.join(package_path, ".mim", "configs", cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def add_args(parser: ArgumentParser, cfg: dict, prefix: str = "") -> ArgumentParser:
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(next(iter(v))), nargs="+")
        else:
            print_log(
                f"cannot parse key {prefix + k} of type {type(v)}", logger="current"
            )
    return parser


def dump(
    obj, file=None, file_format=None, file_client_args=None, backend_args=None, **kwargs
):
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                "same time."
            )

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


class Config:
    def __init__(
        self,
        cfg_dict: dict = None,
        cfg_text: Optional[str] = None,
        filename: Optional[Union[str, Path]] = None,
        env_variables: Optional[dict] = None,
    ):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("_text", text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__("_env_variables", env_variables)

    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        use_predefined_variables: bool = True,
        import_custom_modules: bool = True,
        use_environment_variables: bool = True,
    ) -> "Config":
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text, env_variables = Config._file2dict(
            filename, use_predefined_variables, use_environment_variables
        )
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            try:
                import_modules_from_strings(**cfg_dict["custom_imports"])
            except ImportError as e:
                raise ImportError("Failed to custom import!") from e
        return Config(
            cfg_dict, cfg_text=cfg_text, filename=filename, env_variables=env_variables
        )

    @staticmethod
    def fromstring(cfg_str: str, file_format: str) -> "Config":
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:

            warnings.warn('Please check "file_format", the file format may be .py')

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)

        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    @staticmethod
    def _validate_py_syntax(filename: str):
        with open(filename, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _substitute_env_variables(filename: str, temp_config_name: str):
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        regexp = r"\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}"
        keys = re.findall(regexp, config_file)
        env_variables = dict()
        for var_name, value in keys:
            regexp = (
                r"\{\{[\'\"]?\s*\$" + var_name + r"\s*\:\s*" + value + r"\s*[\'\"]?\}\}"
            )
            if var_name in os.environ:
                value = os.environ[var_name]
                env_variables[var_name] = value
                print_log(
                    f"Using env variable `{var_name}` with value of "
                    f"{value} to replace item in config.",
                    logger="current",
                )
            if not value:
                raise KeyError(
                    f"`{var_name}` cannot be found in `os.environ`."
                    f" Please set `{var_name}` in environment or "
                    "give a default value."
                )
            config_file = re.sub(regexp, value, config_file)

        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return env_variables

    @staticmethod
    def _pre_substitute_base_vars(filename: str, temp_config_name: str) -> dict:
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg: Any, base_var_dict: dict, base_cfg: dict) -> Any:
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(
        filename: str,
        use_predefined_variables: bool = True,
        use_environment_variables: bool = True,
    ) -> Tuple[dict, str, dict]:
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname
            )
            if platform.system() == "Windows":
                temp_config_file.close()
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            env_variables = dict()
            if use_environment_variables:
                env_variables = Config._substitute_env_variables(
                    temp_config_file.name, temp_config_file.name
                )
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name
            )

            base_cfg_dict = ConfigDict()
            cfg_text_list = list()
            for base_cfg_path in Config._get_base_files(temp_config_file.name):
                base_cfg_path, scope = Config._get_cfg_path(base_cfg_path, filename)
                _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                    filename=base_cfg_path,
                    use_predefined_variables=use_predefined_variables,
                    use_environment_variables=use_environment_variables,
                )
                cfg_text_list.append(_cfg_text)
                env_variables.update(_env_variables)
                duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )

                _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                base_cfg_dict.update(_cfg_dict)

            if filename.endswith(".py"):
                with open(temp_config_file.name, encoding="utf-8") as f:
                    codes = ast.parse(f.read())
                    codes = RemoveAssignFromAST(BASE_KEY).visit(codes)
                codeobj = compile(codes, "", mode="exec")
                global_locals_var = {"_base_": base_cfg_dict}
                ori_keys = set(global_locals_var.keys())
                eval(codeobj, global_locals_var, global_locals_var)
                cfg_dict = {
                    key: value
                    for key, value in global_locals_var.items()
                    if (key not in ori_keys and not key.startswith("__"))
                }
            elif filename.endswith((".yml", ".yaml", ".json")):
                cfg_dict = load(temp_config_file.name)
            for key, value in list(cfg_dict.items()):
                if isinstance(value, (types.FunctionType, types.ModuleType)):
                    cfg_dict.pop(key)
            temp_config_file.close()

            Config._parse_scope(cfg_dict)

        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = (
                f"The config file {filename} will be deprecated " "in the future."
            )
            if "expected" in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' "instead."
            if "reference" in deprecation_info:
                warning_msg += (
                    " More information can be found at "
                    f'{deprecation_info["reference"]}'
                )
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + "\n"
        with open(filename, encoding="utf-8") as f:
            cfg_text += f.read()
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("__")}

        cfg_text_list.append(cfg_text)
        cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables

    @staticmethod
    def _dict_to_config_dict(cfg: dict, scope: Optional[str] = None, has_scope=True):
        if isinstance(cfg, dict):
            if has_scope and "type" in cfg:
                has_scope = False
                if scope is not None and cfg.get("_scope_", None) is None:
                    cfg._scope_ = scope
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, "scope", scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(
                    value, scope=scope, has_scope=has_scope
                )
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            ]
        return cfg

    @staticmethod
    def _parse_scope(cfg: dict) -> None:
        if isinstance(cfg, ConfigDict):
            cfg._scope_ = cfg.scope
        elif isinstance(cfg, (tuple, list)):
            [Config._parse_scope(value) for value in cfg]
        else:
            return

    @staticmethod
    def _get_base_files(filename: str) -> list:
        file_format = osp.splitext(filename)[1]
        if file_format == ".py":
            Config._validate_py_syntax(filename)
            with open(filename, encoding="utf-8") as f:
                codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (
                        isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY
                    )

                base_code = next((c for c in codes if is_base_line(c)), None)
                if base_code is not None:
                    base_code = ast.Expression(body=base_code.value)
                    base_files = eval(compile(base_code, "", mode="eval"))
                else:
                    base_files = []
        elif file_format in (".yml", ".yaml", ".json"):
            cfg_dict = load(filename)
            base_files = cfg_dict.get(BASE_KEY, [])
        else:
            raise TypeError(
                "The config type should be py, json, yaml or "
                f"yml, but got {file_format}"
            )
        base_files = base_files if isinstance(base_files, list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str, filename: str) -> Tuple[str, Optional[str]]:
        if "::" in cfg_path:
            scope = cfg_path.partition("::")[0]
            package, cfg_path = _get_package_and_cfg_path(cfg_path)

            if not is_installed(package):
                raise ModuleNotFoundError(
                    f"{package} is not installed, please install {package} " f"manually"
                )
            package_path = get_installed_path(package)
            try:
                cfg_path = _get_external_cfg_path(package_path, cfg_path)
            except ValueError:
                cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
            except FileNotFoundError as e:
                raise e
            return cfg_path, scope
        else:
            cfg_dir = osp.dirname(filename)
            cfg_path = osp.join(cfg_dir, cfg_path)
            return cfg_path, None

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False) -> dict:
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        (dict, list) if allow_list_keys else dict
                    )
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @staticmethod
    def auto_argparser(description=None):
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def text(self) -> str:
        return self._text

    @property
    def env_variables(self) -> dict:
        return self._env_variables

    @property
    def pretty_text(self) -> str:

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict]:
        return (self._cfg_dict, self._filename, self._text, self._env_variables)

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str], dict]):
        _cfg_dict, _filename, _text, _env_variables = state
        super().__setattr__("_cfg_dict", _cfg_dict)
        super().__setattr__("_filename", _filename)
        super().__setattr__("_text", _text)
        super().__setattr__("_text", _env_variables)

    def dump(self, file: Optional[Union[str, Path]] = None):
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = super().__getattribute__("_cfg_dict").to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
            else:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith(".py"):
            with open(file, "w", encoding="utf-8") as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split(".")[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options: dict, allow_list_keys: bool = True) -> None:
        option_cfg_dict: dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__("_cfg_dict")
        super().__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys
            ),
        )


class DictAction(Action):
    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        def find_next_comma(string):
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                if (
                    (char == ",")
                    and (pre.count("(") == pre.count(")"))
                    and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: str = None,
    ):
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split("=", maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


class DefaultScope(ManagerMixin):
    def __init__(self, name: str, scope_name: str):
        super().__init__(name)
        assert isinstance(
            scope_name, str
        ), f"scope_name should be a string, but got {scope_name}"
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> Optional["DefaultScope"]:
        _accquire_lock()
        if cls._instance_dict:
            instance = super().get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: Optional[str]) -> Generator:
        if scope_name is None:
            yield
        else:
            tmp = copy.deepcopy(cls._instance_dict)
            time.sleep(1e-6)
            cls.get_instance(f"overwrite-{time.time()}", scope_name=scope_name)
            try:
                yield
            finally:
                cls._instance_dict = tmp


def init_default_scope(scope: str) -> None:
    never_created = (
        DefaultScope.get_current_instance() is None
        or not DefaultScope.check_instance_created(scope)
    )
    if never_created:
        DefaultScope.get_instance(scope, scope_name=scope)
        return
    current_scope = DefaultScope.get_current_instance()
    if current_scope.scope_name != scope:
        print_log(
            "The current default scope "
            f'"{current_scope.scope_name}" is not "{scope}", '
            "`init_default_scope` will force set the current"
            f'default scope to "{scope}".',
            logger="current",
            level=logging.WARNING,
        )
        new_instance_name = f"{scope}-{datetime.datetime.now()}"
        DefaultScope.get_instance(new_instance_name, scope_name=scope)


def build_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None,
) -> Any:
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
        )

    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )

    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be a mmengine.Registry object, " f"but got {type(registry)}"
        )

    if not (
        isinstance(default_args, (dict, ConfigDict, Config)) or default_args is None
    ):
        raise TypeError(
            "default_args should be a dict, ConfigDict, Config or None, "
            f"but got {type(default_args)}"
        )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.name} registry. "
                    f"Please check whether the value of `{obj_type}` is "
                    "correct or it was registered as expected. More details "
                    "can be found at "
                    "https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
                )
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        try:
            if inspect.isclass(obj_cls) and issubclass(obj_cls, ManagerMixin):
                obj = obj_cls.get_instance(**args)
            else:
                obj = obj_cls(**args)

            print_log(
                f"An `{obj_cls.__name__}` instance is built from "
                "registry, its implementation can be found in "
                f"{obj_cls.__module__}",
                logger="current",
                level=logging.DEBUG,
            )
            return obj

        except Exception as e:
            cls_location = "/".join(obj_cls.__module__.split("."))
            raise type(e)(f"class `{obj_cls.__name__}` in " f"{cls_location}.py: {e}")


def is_seq_of(
    seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None
) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


class Registry:
    def __init__(
        self,
        name: str,
        build_func: Optional[Callable] = None,
        parent: Optional["Registry"] = None,
        scope: Optional[str] = None,
        locations: List = [],
    ):

        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, "Registry"] = dict()
        self._locations = locations
        self._imported = False

        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope()

        self.parent: Optional["Registry"]
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @staticmethod
    def infer_scope() -> str:
        module = inspect.getmodule(sys._getframe(2))
        if module is not None:
            filename = module.__name__
            split_filename = filename.split(".")
            scope = split_filename[0]
        else:
            scope = "mmengine"
            print_log(
                'set scope as "mmengine" when scope can not be inferred. You '
                'can silence this warning by passing a "scope" argument to '
                'Registry like `Registry(name, scope="toy")`',
                logger="current",
                level=logging.WARNING,
            )

        return scope

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    @contextmanager
    def switch_scope_and_registry(self, scope: Optional[str]) -> Generator:
        with DefaultScope.overwrite_default_scope(scope):
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope_name = default_scope.scope_name
                root = self._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    registry = self
            else:
                registry = self
            yield registry

    def _get_root_registry(self) -> "Registry":
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def import_from_location(self) -> None:
        if not self._imported:
            if len(self._locations) == 0 and self.scope in MODULE2PACKAGE:
                print_log(
                    f'The "{self.name}" registry in {self.scope} did not '
                    "set import location. Fallback to call "
                    f"`{self.scope}.utils.register_all_modules` "
                    "instead.",
                    logger="current",
                    level=logging.DEBUG,
                )
                try:
                    module = import_module(f"{self.scope}.utils")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if self.scope in MODULE2PACKAGE:
                        print_log(
                            f"{self.scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{self.scope}, Please install {self.scope} by "
                            f"`pip install {MODULE2PACKAGE[self.scope]}.",
                            logger="current",
                            level=logging.WARNING,
                        )
                    else:
                        print_log(
                            f"Failed to import {self.scope} and register "
                            "its modules, please make sure you "
                            "have registered the module manually.",
                            logger="current",
                            level=logging.WARNING,
                        )
                else:
                    module.register_all_modules(False)

            for loc in self._locations:
                import_module(loc)
                print_log(
                    f"Modules of {self.scope}'s {self.name} registry have "
                    f"been automatically imported from {loc}",
                    logger="current",
                    level=logging.DEBUG,
                )
            self._imported = True

    def get(self, key: str) -> Optional[Type]:
        scope, real_key = self.split_scope_key(key)
        obj_cls = None
        registry_name = self.name
        scope_name = self.scope
        self.import_from_location()

        if scope is None or scope == self._scope:
            if real_key in self._module_dict:
                obj_cls = self._module_dict[real_key]
            elif scope is None:
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        obj_cls = parent._module_dict[real_key]
                        registry_name = parent.name
                        scope_name = parent.scope
                        break
                    parent = parent.parent
        else:
            try:
                import_module(f"{scope}.registry")
                print_log(
                    f"Registry node of {scope} has been automatically " "imported.",
                    logger="current",
                    level=logging.DEBUG,
                )
            except (ImportError, AttributeError, ModuleNotFoundError):
                print_log(
                    f"Cannot auto import {scope}.registry, please check "
                    f'whether the package "{scope}" is installed correctly '
                    "or import the registry manually.",
                    logger="current",
                    level=logging.DEBUG,
                )
            if scope in self._children:
                obj_cls = self._children[scope].get(real_key)
                registry_name = self._children[scope].name
                scope_name = scope
            else:
                root = self._get_root_registry()

                if scope != root._scope and scope not in root._children:
                    pass
                else:
                    obj_cls = root.get(key)

        if obj_cls is not None:
            print_log(
                f'Get class `{obj_cls.__name__}` from "{registry_name}"'
                f' registry in "{scope_name}"',
                logger="current",
                level=logging.DEBUG,
            )
        return obj_cls

    def _search_child(self, scope: str) -> Optional["Registry"]:
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._search_child(scope)
            if registry is not None:
                return registry

        return None

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        return self.build_func(cfg, *args, **kwargs, registry=self)

    def _add_child(self, registry: "Registry") -> None:
        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
    ) -> None:
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(
                "module must be a class or a function, " f"but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(
                    f"{name} is already registered in {self.name} "
                    f"at {existed_module.__module__}"
                )
            self._module_dict[name] = module

    def register_module(
        self,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ) -> Union[type, Callable]:
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register


def master_only(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):

        super().__init__()
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        is_top_level_module = False
        if not hasattr(self, "_params_init_info"):
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param][
                    "tmp_mean_value"
                ] = param.data.mean().cpu()

            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger="current",
                    level=logging.DEBUG,
                )

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if init_cfg["type"] == "Pretrained":
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)

                initialize(self, other_cfgs)

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    update_init_info(
                        m,
                        init_info=f"Initialized by "
                        f"user-defined `init_weights`"
                        f" in {m.__class__.__name__} ",
                    )
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            print_log(
                f"init_weights of {self.__class__.__name__} has "
                f"been called more than once.",
                logger="current",
                level=logging.WARNING,
            )

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self):
        logger = MMLogger.get_current_instance()
        with_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f"\n{name} - {param.shape}: "
                        f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(
                    f"\n{name} - {param.shape}: "
                    f"\n{self._params_init_info[param]['init_info']} \n "
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class BaseModel(BaseModule):
    def __init__(
        self,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        if data_preprocessor is None:
            data_preprocessor = dict(type="BaseDataPreprocessor")
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError(
                "data_preprocessor should be a `dict` or "
                f"`nn.Module` instance, but got "
                f"{type(data_preprocessor)}"
            )

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper
    ) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode="loss")
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)

        return loss, log_vars

    def to(self, *args, **kwargs) -> nn.Module:
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        self._set_device(torch.device("cpu"))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return

        self.apply(apply_fn)

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> Union[Dict[str, torch.Tensor], list]:
        pass

    def _run_forward(
        self, data: Union[dict, tuple, list], mode: str
    ) -> Union[Dict[str, torch.Tensor], list]:
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError(
                "Output of `data_preprocessor` should be "
                f"list, tuple or dict, but got {type(data)}"
            )
        return results


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, "ConfigDict", "Config"]] = None,
) -> "nn.Module":
    if isinstance(cfg, list):
        modules = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry("model", build_model_from_cfg)


class SyncBatchNormFunction(Function):
    @staticmethod
    def symbolic(
        g,
        input,
        running_mean,
        running_var,
        weight,
        bias,
        momentum,
        eps,
        group,
        group_size,
        stats_mode,
    ):
        return g.op(
            "mmcv::MMCVSyncBatchNorm",
            input,
            running_mean,
            running_var,
            weight,
            bias,
            momentum_f=momentum,
            eps_f=eps,
            group_i=group,
            group_size_i=group_size,
            stats_mode=stats_mode,
        )

    @staticmethod
    def forward(
        self,
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        momentum: float,
        eps: float,
        group: int,
        group_size: int,
        stats_mode: str,
    ) -> torch.Tensor:
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size
        self.stats_mode = stats_mode

        assert isinstance(
            input,
            (
                torch.HalfTensor,
                torch.FloatTensor,
                torch.cuda.HalfTensor,
                torch.cuda.FloatTensor,
            ),
        ), f"only support Half or Float Tensor, but {input.type()}"
        output = torch.zeros_like(input)
        input3d = input.flatten(start_dim=2)
        output3d = output.view_as(input3d)
        num_channels = input3d.size(1)
        mean = torch.zeros(num_channels, dtype=torch.float)
        var = torch.zeros(num_channels, dtype=torch.float)
        norm = torch.zeros_like(input3d, dtype=torch.float)
        std = torch.zeros(num_channels, dtype=torch.float)

        batch_size = input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_forward_mean(input3d, mean)
            batch_flag = torch.ones([1], dtype=mean.dtype)
        else:
            batch_flag = torch.zeros([1], dtype=mean.dtype)
        vec = torch.cat([mean, batch_flag])
        if self.stats_mode == "N":
            vec *= batch_size
        if self.group_size > 1:
            dist.all_reduce(vec, group=self.group)
        total_batch = vec[-1].detach()
        mean = vec[:num_channels]

        if self.stats_mode == "default":
            mean = mean / self.group_size
        elif self.stats_mode == "N":
            mean = mean / total_batch.clamp(min=1)
        else:
            raise NotImplementedError
        if batch_size > 0:
            ext_module.sync_bn_forward_var(input3d, mean, var)

        if self.stats_mode == "N":
            var *= batch_size
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)

        if self.stats_mode == "default":
            var /= self.group_size
        elif self.stats_mode == "N":
            var /= total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        update_flag = total_batch.clamp(max=1)
        momentum = update_flag * self.momentum
        ext_module.sync_bn_forward_output(
            input3d,
            mean,
            var,
            weight,
            bias,
            running_mean,
            running_var,
            norm,
            std,
            output3d,
            eps=self.eps,
            momentum=momentum,
            group_size=self.group_size,
        )
        self.save_for_backward(norm, std, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output: torch.Tensor) -> tuple:
        norm, std, weight = self.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        grad_input = torch.zeros_like(grad_output)
        grad_output3d = grad_output.flatten(start_dim=2)
        grad_input3d = grad_input.view_as(grad_output3d)

        batch_size = grad_input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_backward_param(
                grad_output3d, norm, grad_weight, grad_bias
            )
        if self.group_size > 1:
            dist.all_reduce(grad_weight, group=self.group)
            dist.all_reduce(grad_bias, group=self.group)
            grad_weight /= self.group_size
            grad_bias /= self.group_size

        if batch_size > 0:
            ext_module.sync_bn_backward_data(
                grad_output3d, weight, grad_weight, grad_bias, norm, std, grad_input3d
            )

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


@MODELS.register_module(name="MMSyncBN")
class SyncBatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        group: Optional[int] = None,
        stats_mode: str = "default",
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.group_size = dist.get_world_size(group)
        assert stats_mode in [
            "default",
            "N",
        ], f'"stats_mode" only accepts "default" and "N", got "{stats_mode}"'
        self.stats_mode = stats_mode
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input, got {input.dim()}D input")
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats:
            return SyncBatchNormFunction.apply(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                exponential_average_factor,
                self.eps,
                self.group,
                self.group_size,
                self.stats_mode,
            )
        else:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                exponential_average_factor,
                self.eps,
            )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"({self.num_features}, "
        s += f"eps={self.eps}, "
        s += f"momentum={self.momentum}, "
        s += f"affine={self.affine}, "
        s += f"track_running_stats={self.track_running_stats}, "
        s += f"group_size={self.group_size},"
        s += f"stats_mode={self.stats_mode})"
        return s


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input: torch.Tensor):
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print_log(
                f"Failed to convert {child} from SyncBN to BN!",
                logger="current",
                level=logging.WARNING,
            )
    del module
    return module_output


dataset_aliases = {
    "voc": ["voc", "pascal_voc", "voc07", "voc12"],
    "imagenet_det": ["det", "imagenet_det", "ilsvrc_det"],
    "imagenet_vid": ["vid", "imagenet_vid", "ilsvrc_vid"],
    "coco": ["coco", "mscoco", "ms_coco"],
    "coco_panoptic": ["coco_panoptic", "panoptic"],
    "wider_face": ["WIDERFaceDataset", "wider_face", "WIDERFace"],
    "cityscapes": ["cityscapes"],
    "oid_challenge": ["oid_challenge", "openimages_challenge"],
    "oid_v6": ["oid_v6", "openimages_v6"],
    "objects365v1": ["objects365v1", "obj365v1"],
    "objects365v2": ["objects365v2", "obj365v2"],
}


def get_classes(dataset) -> list:
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + "_classes()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset)}")
    return labels


class CheckpointLoader:
    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f"{prefix} is already registered as a loader backend, "
                    'add "force=True" if you want to override it'
                )
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True)
        )

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        for p in cls._schemes:
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger="current"):
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(
            f"Loads checkpoint by {class_name[10:]} backend from path: " f"{filename}",
            logger=logger,
        )
        return checkpoint_loader(filename, map_location)


def _load_checkpoint(filename, map_location=None, logger=None):
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(filename, map_location=map_location)
    if "state_dict" in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_k = k.replace("bn.", "batch_norm2d.")
            new_state_dict[new_k] = v
        checkpoint["state_dict"] = new_state_dict
    return checkpoint


WEIGHT_INITIALIZERS = Registry("weight initializer")


def _initialize(module, cfg, wholemodule=False):
    func = build_from_cfg(cfg, WEIGHT_INITIALIZERS)
    func.wholemodule = wholemodule
    func(module)


def _initialize_override(module, override, cfg):
    if not isinstance(override, (dict, list)):
        raise TypeError(
            f"override must be a dict or a list of dict, \
                but got {type(override)}"
        )

    override = [override] if isinstance(override, dict) else override

    for override_ in override:

        cp_override = copy.deepcopy(override_)
        name = cp_override.pop("name", None)
        if name is None:
            raise ValueError(
                '`override` must contain the key "name",' f"but got {cp_override}"
            )
        if not cp_override:
            cp_override.update(cfg)
        elif "type" not in cp_override.keys():
            raise ValueError(f'`override` need "type" key, but got {cp_override}')

        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(
                f"module did not have attribute {name}, "
                f"but init_cfg is {cp_override}."
            )


def initialize(module, init_cfg):
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(
            f"init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}"
        )

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop("override", None)
        _initialize(module, cp_cfg)

        if override is not None:
            cp_cfg.pop("layer", None)
            _initialize_override(module, override, cp_cfg)
        else:
            pass


def update_init_info(module, init_info):
    assert hasattr(
        module, "_params_init_info"
    ), f"Can not find `_params_init_info` in {module}"
    for name, param in module.named_parameters():

        assert param in module._params_init_info, (
            f"Find a new :obj:`Parameter` "
            f"named `{name}` during executing the "
            f"`init_weights` of "
            f"`{module.__class__.__name__}`. "
            f"Please do not add or "
            f"replace parameters during executing "
            f"the `init_weights`. "
        )
        mean_value = param.data.mean().cpu()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:
            module._params_init_info[param]["init_info"] = init_info
            module._params_init_info[param]["tmp_mean_value"] = mean_value


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    return get_rank(group) == 0


class HistoryBuffer:
    _statistics_methods: dict = dict()

    def __init__(
        self,
        log_history: Sequence = [],
        count_history: Sequence = [],
        max_length: int = 1000000,
    ):

        self.max_length = max_length
        self._set_default_statistics()
        assert len(log_history) == len(
            count_history
        ), "The lengths of log_history and count_histroy should be equal"
        if len(log_history) > max_length:
            warnings.warn(
                f"The length of history buffer({len(log_history)}) "
                f"exceeds the max_length({max_length}), the first "
                "few elements will be ignored."
            )
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)

    def _set_default_statistics(self) -> None:
        self._statistics_methods.setdefault("min", HistoryBuffer.min)
        self._statistics_methods.setdefault("max", HistoryBuffer.max)
        self._statistics_methods.setdefault("current", HistoryBuffer.current)
        self._statistics_methods.setdefault("mean", HistoryBuffer.mean)

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        if not isinstance(log_val, (int, float)) or not isinstance(count, (int, float)):
            raise TypeError(
                f"log_val must be int or float but got "
                f"{type(log_val)}, count must be int but got "
                f"{type(count)}"
            )
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length :]
            self._count_history = self._count_history[-self.max_length :]

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        method_name = method.__name__
        assert (
            method_name not in cls._statistics_methods
        ), "method_name cannot be registered twice!"
        cls._statistics_methods[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        if method_name not in self._statistics_methods:
            raise KeyError(
                f"{method_name} has not been registered in "
                "HistoryBuffer._statistics_methods"
            )
        method = self._statistics_methods[method_name]
        return method(self, *arg, **kwargs)

    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    def current(self) -> np.ndarray:
        if len(self._log_history) == 0:
            raise ValueError(
                "HistoryBuffer._log_history is an empty array! "
                "please call update first"
            )
        return self._log_history[-1]


class MessageHub(ManagerMixin):
    def __init__(
        self,
        name: str,
        log_scalars: Optional[OrderedDict] = None,
        runtime_info: Optional[OrderedDict] = None,
        resumed_keys: Optional[OrderedDict] = None,
    ):
        super().__init__(name)
        self._log_scalars = log_scalars if log_scalars is not None else OrderedDict()
        self._runtime_info = runtime_info if runtime_info is not None else OrderedDict()
        self._resumed_keys = resumed_keys if resumed_keys is not None else OrderedDict()

        assert isinstance(self._log_scalars, OrderedDict)
        assert isinstance(self._runtime_info, OrderedDict)
        assert isinstance(self._resumed_keys, OrderedDict)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), (
                "The type of log_scalars'value must be HistoryBuffer, but "
                f"got {type(value)}"
            )

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, (
                "Key in `resumed_keys` must contained in `log_scalars` or "
                f"`runtime_info`, but got {key}"
            )

    @classmethod
    def get_current_instance(cls) -> "MessageHub":
        if not cls._instance_dict:
            cls.get_instance("mmengine")
        return super().get_current_instance()

    def update_scalar(
        self,
        key: str,
        value: Union[int, float, np.ndarray, "torch.Tensor"],
        count: int = 1,
        resumed: bool = True,
    ) -> None:
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(
            count, int
        ), f"The type of count must be int. but got {type(count): {count}}"
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        assert isinstance(log_dict, dict), (
            "`log_dict` must be a dict!, " f"but got {type(log_dict)}"
        )
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert "value" in log_val, f"value must be defined in {log_val}"
                count = self._get_valid_value(log_val.get("count", 1))
                value = log_val["value"]
            else:
                count = 1
                value = log_val
            assert isinstance(count, int), (
                "The type of count must be int. but got " f"{type(count): {count}}"
            )
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def update_info_dict(self, info_dict: dict, resumed: bool = True) -> None:
        assert isinstance(info_dict, dict), (
            "`log_dict` must be a dict!, " f"but got {type(info_dict)}"
        )
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, (
                f"{key} used to be {self._resumed_keys[key]}, but got "
                "{resumed} now. resumed keys cannot be modified repeatedly."
            )

    @property
    def log_scalars(self) -> OrderedDict:
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        if key not in self.log_scalars:
            raise KeyError(
                f"{key} is not found in Messagehub.log_buffers: "
                f"instance name is: {MessageHub.instance_name}"
            )
        return self.log_scalars[key]

    def get_info(self, key: str) -> Any:
        if key not in self.runtime_info:
            raise KeyError(
                f"{key} is not found in Messagehub.log_buffers: "
                f"instance name is: {MessageHub.instance_name}"
            )
        return self._runtime_info[key]

    def _get_valid_value(
        self,
        value: Union["torch.Tensor", np.ndarray, np.number, int, float],
    ) -> Union[int, float]:
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            assert hasattr(value, "numel") and value.numel() == 1
            value = value.item()
        return value

    def state_dict(self) -> dict:
        saved_scalars = OrderedDict()
        saved_info = OrderedDict()

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:
                    print_log(
                        f"{key} in message_hub cannot be copied, "
                        f"just return its reference. ",
                        logger="current",
                        level=logging.WARNING,
                    )
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys,
        )

    def load_state_dict(self, state_dict: Union["MessageHub", dict]) -> None:
        if isinstance(state_dict, dict):
            for key in ("log_scalars", "runtime_info", "resumed_keys"):
                assert key in state_dict, (
                    "The loaded `state_dict` of `MessageHub` must contain "
                    f"key: `{key}`"
                )
            for key, value in state_dict["log_scalars"].items():
                if not isinstance(value, HistoryBuffer):
                    print_log(
                        f"{key} in message_hub is not HistoryBuffer, "
                        f"just skip resuming it.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    continue
                self.log_scalars[key] = value

            for key, value in state_dict["runtime_info"].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:
                    print_log(
                        f"{key} in message_hub cannot be copied, "
                        f"just return its reference.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    self._runtime_info[key] = value

            for key, value in state_dict["resumed_keys"].items():
                if key not in set(self.log_scalars.keys()) | set(
                    self._runtime_info.keys()
                ):
                    print_log(
                        f"resumed key: {key} is not defined in message_hub, "
                        f"just skip resuming this key.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    continue
                elif not value:
                    print_log(
                        f"Although resumed key: {key} is False, {key} "
                        "will still be loaded this time. This key will "
                        "not be saved by the next calling of "
                        "`MessageHub.state_dict()`",
                        logger="current",
                        level=logging.WARNING,
                    )
                self._resumed_keys[key] = value
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)


def _get_norm() -> tuple:
    if TORCH_VERSION == "parrots":
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm

        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm

        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


def _get_conv() -> tuple:
    if TORCH_VERSION == "parrots":
        from parrots.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    else:
        from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_dataloader() -> tuple:
    if TORCH_VERSION == "parrots":
        from torch.utils.data import DataLoader, PoolDataLoader
    else:
        from torch.utils.data import DataLoader

        PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_pool() -> tuple:
    if TORCH_VERSION == "parrots":
        from parrots.nn.modules.pool import (
            _AdaptiveAvgPoolNd,
            _AdaptiveMaxPoolNd,
            _AvgPoolNd,
            _MaxPoolNd,
        )
    else:
        from torch.nn.modules.pooling import (
            _AdaptiveAvgPoolNd,
            _AdaptiveMaxPoolNd,
            _AvgPoolNd,
            _MaxPoolNd,
        )
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


def has_batch_norm(model: nn.Module) -> bool:
    if isinstance(model, _BatchNorm):
        return True
    for m in model.children():
        if has_batch_norm(m):
            return True
    return False


OPTIM_WRAPPERS = Registry("optim_wrapper", scope="yoloworld")


@OPTIM_WRAPPERS.register_module()
class OptimWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        accumulative_counts: int = 1,
        clip_grad: Optional[dict] = None,
    ):
        assert (
            accumulative_counts > 0
        ), "_accumulative_counts at least greater than or equal to 1"
        self._accumulative_counts = accumulative_counts

        assert isinstance(optimizer, Optimizer), (
            "optimizer must be a `torch.optim.Optimizer` instance, but got "
            f"{type(optimizer)}"
        )
        self.optimizer = optimizer

        if clip_grad is not None:
            assert isinstance(clip_grad, dict) and clip_grad, (
                "If `clip_grad` is not None, it should be a `dict` "
                "which is the arguments of `torch.nn.utils.clip_grad_norm_` "
                "or clip_grad_value_`."
            )
            clip_type = clip_grad.pop("type", "norm")
            if clip_type == "norm":
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = "grad_norm"
            elif clip_type == "value":
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = "grad_value"
            else:
                raise ValueError(
                    'type of clip_grad should be "norm" or '
                    f'"value" but got {clip_type}'
                )
            assert clip_grad, (
                "`clip_grad` should contain other arguments "
                "besides `type`. The arguments should match "
                "with the `torch.nn.utils.clip_grad_norm_` or "
                "clip_grad_value_`"
            )
        self.clip_grad_kwargs = clip_grad
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        self._max_counts = -1
        self._remainder_counts = -1

    def update_params(
        self,
        loss: torch.Tensor,
        step_kwargs: Optional[Dict] = None,
        zero_kwargs: Optional[Dict] = None,
    ) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        return self.optimizer.defaults

    def get_lr(self) -> Dict[str, List[float]]:
        lr = [group["lr"] for group in self.param_groups]
        return dict(lr=lr)

    def get_momentum(self) -> Dict[str, List[float]]:
        momentum = []
        for group in self.param_groups:
            if "momentum" in group.keys():
                momentum.append(group["momentum"])
            elif "betas" in group.keys():
                momentum.append(group["betas"][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    @contextmanager
    def optim_context(self, model: nn.Module):
        if not self.should_sync() and hasattr(model, "no_sync"):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group["params"])

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            if grad is not None:
                self.message_hub.update_scalar(f"train/{self.grad_name}", float(grad))

    def initialize_count_status(
        self, model: nn.Module, init_counts: int, max_counts: int
    ) -> None:
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                "Resumed iteration number is not divisible by "
                "`_accumulative_counts` in `GradientCumulativeOptimizerHook`, "
                "which means the gradient of some iterations is lost and the "
                "result may be influenced slightly.",
                logger="current",
                level=logging.WARNING,
            )

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                "Gradient accumulative may slightly decrease "
                "performance because the model has BatchNorm layers.",
                logger="current",
                level=logging.WARNING,
            )
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        return (
            self._inner_count % self._accumulative_counts == 0
            or self._inner_count == self._max_counts
        )

    def should_sync(self) -> bool:
        return (self._inner_count + 1) % self._accumulative_counts == 0 or (
            self._inner_count + 1
        ) == self._max_counts

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self._accumulative_counts == 1:
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                "loss_factor should be larger than zero! This error could "
                "happened when initialize_iter_status called with an "
                "error `init_counts` or `max_counts`"
            )

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        return self._inner_count

    def __repr__(self):
        wrapper_info = (
            f"Type: {type(self).__name__}\n"
            f"_accumulative_counts: {self._accumulative_counts}\n"
            "optimizer: \n"
        )
        optimizer_str = repr(self.optimizer) + "\n"
        return wrapper_info + optimizer_str


def is_list_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=list)


class BaseDataElement:
    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        assert isinstance(
            metainfo, dict
        ), f"metainfo should be a ``dict`` but got {type(metainfo)}"
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"
        for k, v in data.items():
            setattr(self, k, v)

    def update(self, instance: "BaseDataElement") -> None:
        assert isinstance(
            instance, BaseDataElement
        ), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> "BaseDataElement":
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        private_keys = {
            "_" + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        return list(self._metainfo_fields)

    def values(self) -> list:
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    def __delattr__(self, item: str):
        if item in ("_metainfo_fields", "_data_fields"):
            raise AttributeError(
                f"{item} has been used as a " "private attribute, which is immutable."
            )
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        assert len(args) < 3, "``pop`` get more than 2 arguments"
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item: str) -> bool:
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        value: Any,
        name: str,
        dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
        field_type: str = "data",
    ) -> None:
        assert field_type in ["metainfo", "data"]
        if dtype is not None:
            assert isinstance(
                value, dtype
            ), f"{value} should be a {dtype} but got {type(value)}"

        if field_type == "metainfo":
            if name in self._data_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of metainfo "
                    f"because {name} is already a data field"
                )
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of data "
                    f"because {name} is already a metainfo field"
                )
            self._data_fields.add(name)
        super().__setattr__(name, value)

    def to(self, *args, **kwargs) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cpu(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cuda(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def npu(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def detach(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def numpy(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        def _addindent(s_: str, num_spaces: int) -> str:
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def dump(obj: Any) -> str:
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)


EnhancedBatchInputs = List[Union[torch.Tensor, List[torch.Tensor]]]
EnhancedBatchDataSamples = List[List[BaseDataElement]]
DATA_BATCH = Union[
    Dict[str, Union[EnhancedBatchInputs, EnhancedBatchDataSamples]], tuple, dict
]
MergedDataSamples = List[BaseDataElement]
CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    def __init__(self, non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")

    def cast_data(self, data: CastData) -> CastData:
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, "_fields"):
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        return self.cast_data(data)

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) -> nn.Module:
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def npu(self, *args, **kwargs) -> nn.Module:
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        self._device = torch.device("cpu")
        return super().cpu()


MODEL_WRAPPERS = Registry("model_wrapper")


def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(is_model_wrapper(model, child) for child in registry.children.values())


class BaseDataElement:
    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        assert isinstance(
            metainfo, dict
        ), f"metainfo should be a ``dict`` but got {type(metainfo)}"
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"
        for k, v in data.items():
            setattr(self, k, v)

    def update(self, instance: "BaseDataElement") -> None:
        assert isinstance(
            instance, BaseDataElement
        ), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> "BaseDataElement":
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        private_keys = {
            "_" + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        return list(self._metainfo_fields)

    def values(self) -> list:
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    def __delattr__(self, item: str):
        if item in ("_metainfo_fields", "_data_fields"):
            raise AttributeError(
                f"{item} has been used as a " "private attribute, which is immutable."
            )
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        assert len(args) < 3, "``pop`` get more than 2 arguments"
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item: str) -> bool:
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        value: Any,
        name: str,
        dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
        field_type: str = "data",
    ) -> None:
        assert field_type in ["metainfo", "data"]
        if dtype is not None:
            assert isinstance(
                value, dtype
            ), f"{value} should be a {dtype} but got {type(value)}"

        if field_type == "metainfo":
            if name in self._data_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of metainfo "
                    f"because {name} is already a data field"
                )
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of data "
                    f"because {name} is already a metainfo field"
                )
            self._data_fields.add(name)
        super().__setattr__(name, value)

    def to(self, *args, **kwargs) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cpu(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cuda(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def npu(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def detach(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def numpy(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> "BaseDataElement":
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        def _addindent(s_: str, num_spaces: int) -> str:
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def dump(obj: Any) -> str:
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)


@MODELS.register_module()
class BaseTTAModel(BaseModel):
    def __init__(
        self,
        module: Union[dict, nn.Module],
        data_preprocessor: Union[dict, nn.Module, None] = None,
    ):
        super().__init__()
        if isinstance(module, nn.Module):
            self.module = module
        elif isinstance(module, dict):
            if data_preprocessor is not None:
                module["data_preprocessor"] = data_preprocessor
            self.module = MODELS.build(module)
        else:
            raise TypeError(
                "The type of module should be a `nn.Module` "
                f"instance or a dict, but got {module}"
            )
        assert hasattr(
            self.module, "test_step"
        ), "Model wrapped by BaseTTAModel must implement `test_step`!"

    @abstractmethod
    def merge_preds(
        self, data_samples_list: EnhancedBatchDataSamples
    ) -> MergedDataSamples:
        pass

    def test_step(self, data):
        data_list: Union[List[dict], List[list]]
        if isinstance(data, dict):
            num_augs = len(data[next(iter(data))])
            data_list = [
                {key: value[idx] for key, value in data.items()}
                for idx in range(num_augs)
            ]
        elif isinstance(data, (tuple, list)):
            num_augs = len(data[0])
            data_list = [[_data[idx] for _data in data] for idx in range(num_augs)]
        else:
            raise TypeError(
                "data given by dataLoader should be a dict, "
                f"tuple or a list, but got {type(data)}"
            )

        predictions = []
        for data in data_list:
            predictions.append(self.module.test_step(data))
        return self.merge_preds(list(zip(*predictions)))

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> Union[Dict[str, torch.Tensor], list]:
        raise NotImplementedError(
            "`BaseTTAModel.forward` will not be called during training or"
            "testing. Please call `test_step` instead. If you want to use"
            "`BaseTTAModel.forward`, please implement this method"
        )


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)


def _load_checkpoint_to_model(
    model, checkpoint, strict=False, logger=None, revise_keys=[(r"^module\.", "")]
):

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})

    state_dict._metadata = metadata

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint(
    model,
    filename,
    map_location=None,
    strict=False,
    logger=None,
    revise_keys=[(r"^module\.", "")],
):
    checkpoint = _load_checkpoint(filename, map_location, logger)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    return _load_checkpoint_to_model(model, checkpoint, strict, logger, revise_keys)


DATASETS = Registry("dataset", scope="yoloworld")


def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = "none",
    device: str = "cpu",
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None
    init_default_scope(config.get("default_scope", "mmdet"))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter("once")
        warnings.warn("checkpoint is None, use COCO classes by default.")
        model.dataset_meta = {"classes": get_classes("coco")}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        checkpoint_meta = checkpoint.get("meta", {})
        if "dataset_meta" in checkpoint_meta:
            model.dataset_meta = {
                k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()
            }
        elif "CLASSES" in checkpoint_meta:
            classes = checkpoint_meta["CLASSES"]
            model.dataset_meta = {"classes": classes}
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "dataset_meta or class names are not saved in the "
                "checkpoint's meta data, use COCO classes by default."
            )
            model.dataset_meta = {"classes": get_classes("coco")}

    if palette != "none":
        model.dataset_meta["palette"] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        test_dataset_cfg["lazy_init"] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get("palette", None)
        if cfg_palette is not None:
            model.dataset_meta["palette"] = cfg_palette
        else:
            if "palette" not in model.dataset_meta:
                warnings.warn(
                    "palette does not exist, random is used by default. "
                    "You can also set the palette to customize."
                )
                model.dataset_meta["palette"] = "random"

    model.cfg = config
    model.eval()
    return model


IndexType = Union[
    str,
    slice,
    int,
    list,
    torch.LongTensor,
    torch.BoolTensor,
    np.ndarray,
]


class InstanceData(BaseDataElement):
    def __setattr__(self, name: str, value: Sized):
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )

        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"

            if len(self) > 0:
                assert len(value) == len(self), (
                    "The length of "
                    f"values {len(value)} is "
                    "not consistent with "
                    "the length of this "
                    ":obj:`InstanceData` "
                    f"{len(self)}"
                )
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> "InstanceData":
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)
        assert isinstance(
            item,
            (
                str,
                slice,
                int,
                torch.LongTensor,
                torch.BoolTensor,
            ),
        )

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError(f"Index {item} out of range!")
            else:
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, (
                "Only support to get the" " values along the first dimension."
            )
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), (
                    "The shape of the "
                    "input(BoolTensor) "
                    f"{len(item)} "
                    "does not match the shape "
                    "of the indexed tensor "
                    "in results_field "
                    f"{len(self)} at "
                    "first dimension."
                )

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (
                    hasattr(v, "__getitem__") and hasattr(v, "cat")
                ):
                    if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f"The type of `{k}` is `{type(v)}`, which has no "
                        "attribute of `cat`, so it does not "
                        "support slice with `bool`"
                    )

        else:
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data

    @staticmethod
    def cat(instances_list: List["InstanceData"]) -> "InstanceData":
        assert all(isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        field_keys_list = [instances.all_keys() for instances in instances_list]
        assert len({len(field_keys) for field_keys in field_keys_list}) == 1 and len(
            set(itertools.chain(*field_keys_list))
        ) == len(field_keys_list[0]), (
            "There are different keys in "
            "`instances_list`, which may "
            "cause the cat operation "
            "to fail. Please make sure all "
            "elements in `instances_list` "
            "have the exact same key."
        )

        new_data = instances_list[0].__class__(metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, "cat"):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f"The type of `{k}` is `{type(v0)}` which has no "
                    "attribute of `cat`"
                )
            new_data[k] = new_values
        return new_data

    def __len__(self) -> int:
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


class PixelData(BaseDataElement):
    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), (
                f"Can not set {type(value)}, only support"
                f" {(torch.Tensor, np.ndarray)}"
            )

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    "The height and width of "
                    f"values {tuple(value.shape[-2:])} is "
                    "not consistent with "
                    "the shape of this "
                    ":obj:`PixelData` "
                    f"{self.shape}"
                )
            assert value.ndim in [
                2,
                3,
            ], f"The dim of value must be 2 or 3, but got {value.ndim}"
            if value.ndim == 2:
                value = value[None]
                warnings.warn(
                    "The shape of value will convert from "
                    f"{value.shape[-2:]} to {value.shape}"
                )
            super().__setattr__(name, value)

    def __getitem__(self, item: Sequence[Union[int, slice]]) -> "PixelData":
        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, "Only support to slice height and width"
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        "The type of element in input must be int or slice, "
                        f"but got {type(single_item)}"
                    )
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(f"Unsupported type {type(item)} for slicing PixelData")
        return new_data

    @property
    def shape(self):
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None


class DetDataSample(BaseDataElement):
    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_gt_panoptic_seg", dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_pred_panoptic_seg", dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, "_gt_sem_seg", dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, "_pred_sem_seg", dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]


class BaseInstanceMasks(metaclass=ABCMeta):
    @abstractmethod
    def rescale(self, scale, interpolation="nearest"):
        pass

    @abstractmethod
    def resize(self, out_shape, interpolation="nearest"):
        pass

    @abstractmethod
    def flip(self, flip_direction="horizontal"):
        pass

    @abstractmethod
    def pad(self, out_shape, pad_val):
        pass

    @abstractmethod
    def crop(self, bbox):
        pass

    @abstractmethod
    def crop_and_resize(
        self, bboxes, out_shape, inds, device, interpolation="bilinear", binarize=True
    ):
        pass

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        pass

    @property
    @abstractmethod
    def areas(self):
        pass

    @abstractmethod
    def to_ndarray(self):
        pass

    @abstractmethod
    def to_tensor(self, dtype, device):
        pass

    @abstractmethod
    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        pass

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        pass

    @abstractmethod
    def rotate(self, out_shape, angle, center=None, scale=1.0, border_value=0):
        pass

    def get_bboxes(self, dst_type="hbb"):
        _, box_type_cls = get_box_type(dst_type)
        return box_type_cls.from_instance_masks(self)

    @classmethod
    @abstractmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        pass


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f"Scale must be a number or tuple of int, but got {type(scale)}"
        )

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }

imread_backend = "cv2"
cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imrescale(
    img: np.ndarray,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imflip(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        return np.flip(img, axis=1)
    elif direction == "vertical":
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def impad(
    img: np.ndarray,
    *,
    shape: Optional[Tuple[int, int]] = None,
    padding: Union[int, tuple, None] = None,
    pad_val: Union[float, List] = 0,
    padding_mode: str = "constant",
) -> np.ndarray:
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(
            "pad_val must be a int or a tuple. " f"But received {type(pad_val)}"
        )

    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(
            "Padding must be a int or a 2, or 4 element tuple."
            f"But received {padding}"
        )

    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val,
    )

    return img


def _get_translate_matrix(
    offset: Union[int, float], direction: str = "horizontal"
) -> np.ndarray:
    if direction == "horizontal":
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == "vertical":
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(
    img: np.ndarray,
    offset: Union[int, float],
    direction: str = "horizontal",
    border_value: Union[int, tuple] = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (
            "Expected the num of elements in tuple equals the channels"
            "of input image. Found {} vs {}".format(len(border_value), channels)
        )
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`.")
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation],
    )
    return translated


def _get_shear_matrix(
    magnitude: Union[int, float], direction: str = "horizontal"
) -> np.ndarray:
    if direction == "horizontal":
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == "vertical":
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(
    img: np.ndarray,
    magnitude: Union[int, float],
    direction: str = "horizontal",
    border_value: Union[int, Tuple[int, int]] = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (
            "Expected the num of elements in tuple equals the channels"
            "of input image. Found {} vs {}".format(len(border_value), channels)
        )
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`")
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation],
    )
    return sheared


cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}


def imrotate(
    img: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    border_value: int = 0,
    interpolation: str = "bilinear",
    auto_bound: bool = False,
    border_mode: str = "constant",
) -> np.ndarray:
    if center is not None and auto_bound:
        raise ValueError("`auto_bound` conflicts with `center`")
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value,
    )
    return rotated


def ensure_rng(rng=None):
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


class BitmapMasks(BaseInstanceMasks):
    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2
            else:
                assert masks.ndim == 3

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation="nearest"):
        if len(self.masks) == 0:
            new_w, new_h = rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack(
                [
                    imrescale(mask, scale, interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation="nearest"):
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack(
                [
                    imresize(mask, out_shape[::-1], interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction="horizontal"):
        assert flip_direction in ("horizontal", "vertical", "diagonal")

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack(
                [imflip(mask, direction=flip_direction) for mask in self.masks]
            )
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack(
                [impad(mask, shape=out_shape, pad_val=pad_val) for mask in self.masks]
            )
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1 : y1 + h, x1 : x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(
        self,
        bboxes,
        out_shape,
        inds,
        device="cpu",
        interpolation="bilinear",
        binarize=True,
    ):
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)
        if num_bbox > 0:
            gt_masks_th = (
                torch.from_numpy(self.masks).index_select(0, inds).to(dtype=rois.dtype)
            )
            targets = roi_align(
                gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, "avg", True
            ).squeeze(1)
            if binarize:
                resized_masks = (targets >= 0.5).cpu().numpy()
            else:
                resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w), dtype=np.uint8)
        else:
            expanded_mask = np.zeros(
                (len(self), expanded_h, expanded_w), dtype=np.uint8
            )
            expanded_mask[
                :, top : top + self.height, left : left + self.width
            ] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            masks = self.masks
            if masks.shape[-2:] != out_shape:
                empty_masks = np.zeros((masks.shape[0], *out_shape), dtype=masks.dtype)
                min_h = min(out_shape[0], masks.shape[1])
                min_w = min(out_shape[1], masks.shape[2])
                empty_masks[:, :min_h, :min_w] = masks[:, :min_h, :min_w]
                masks = empty_masks
            translated_masks = imtranslate(
                masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose((2, 0, 1)).astype(
                self.masks.dtype
            )
        return BitmapMasks(translated_masks, *out_shape)

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(
        self,
        out_shape,
        angle,
        center=None,
        scale=1.0,
        border_value=0,
        interpolation="bilinear",
    ):
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=border_value,
                interpolation=interpolation,
            )
            if rotated_masks.ndim == 2:
                rotated_masks = rotated_masks[:, :, None]
            rotated_masks = rotated_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        return self.masks

    def to_tensor(self, dtype, device):
        return torch.tensor(self.masks, dtype=dtype)

    @classmethod
    def random(cls, num_masks=3, height=32, width=32, dtype=np.uint8, rng=None):
        rng = ensure_rng(rng)
        masks = (rng.rand(num_masks, height, width) > 0.1).astype(dtype)
        self = cls(masks, height=height, width=width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_array = np.concatenate([m.masks for m in masks], axis=0)
        return cls(mask_array, *mask_array.shape[1:])


class PolygonMasks(BaseInstanceMasks):
    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                index = np.where(index)[0].tolist()
            else:
                index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    f"Unsupported input of type {type(index)} for indexing!"
                )
        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        new_w, new_h = rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2] * w_scale
                    p[1::2] = p[1::2] * h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction="horizontal"):
        assert flip_direction in ("horizontal", "vertical", "diagonal")
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if flip_direction == "horizontal":
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == "vertical":
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height, self.width)
        return flipped_masks

    def crop(self, bbox):
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            crop_box = geometry.box(x1, y1, x2, y2).buffer(0.0)
            cropped_masks = []
            initial_settings = np.seterr()
            np.seterr(invalid="ignore")
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)
                    if not p.is_valid:
                        continue
                    cropped = p.intersection(crop_box)
                    if cropped.is_empty:
                        continue
                    if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                        cropped = cropped.geoms
                    else:
                        cropped = [cropped]
                    for poly in cropped:
                        if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                            continue
                        coords = np.asarray(poly.exterior.coords)
                        coords = coords[:-1]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        cropped_poly_per_obj.append(coords.reshape(-1))
                if len(cropped_poly_per_obj) == 0:
                    cropped_poly_per_obj = [np.array([0, 0, 0, 0, 0, 0])]
                cropped_masks.append(cropped_poly_per_obj)
            np.seterr(**initial_settings)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        raise NotImplementedError

    def crop_and_resize(
        self,
        bboxes,
        out_shape,
        inds,
        device=None,
        interpolation="bilinear",
        binarize=True,
    ):
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        if not binarize:
            raise ValueError(
                "Polygons are always binary, " "setting binarize=False is unsupported"
            )

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :]
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1, 1)
            h = np.maximum(y2 - y1, 1)
            h_scale = out_h / max(h, 0.1)
            w_scale = out_w / max(w, 0.1)

            resized_mask = []
            for p in mask:
                p = p.copy()
                p[0::2] = p[0::2] - bbox[0]
                p[1::2] = p[1::2] - bbox[1]
                p[0::2] = p[0::2] * w_scale
                p[1::2] = p[1::2] * h_scale
                resized_mask.append(p)
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=None,
        interpolation=None,
    ):
        assert border_value is None or border_value == 0, (
            "Here border_value is not "
            f"used, and defaultly should be None or 0. got {border_value}."
        )
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if direction == "horizontal":
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == "vertical":
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks = []
            if direction == "horizontal":
                shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(np.float32)
            elif direction == "vertical":
                shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)
                    new_coords = np.matmul(shear_matrix, p)
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, out_shape[0])
                    sheared_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks, *out_shape)
        return sheared_masks

    def rotate(
        self,
        out_shape,
        angle,
        center=None,
        scale=1.0,
        border_value=0,
        interpolation="bilinear",
    ):
        if len(self.masks) == 0:
            rotated_masks = PolygonMasks([], *out_shape)
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for poly_per_obj in self.masks:
                rotated_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    coords = np.stack([p[0::2], p[1::2]], axis=1)
                    coords = np.concatenate(
                        (coords, np.ones((coords.shape[0], 1), coords.dtype)), axis=1
                    )
                    rotated_coords = np.matmul(
                        rotate_matrix[None, :, :], coords[:, :, None]
                    )[..., 0]
                    rotated_coords[:, 0] = np.clip(
                        rotated_coords[:, 0], 0, out_shape[1]
                    )
                    rotated_coords[:, 1] = np.clip(
                        rotated_coords[:, 1], 0, out_shape[0]
                    )
                    rotated_poly.append(rotated_coords.reshape(-1))
                rotated_masks.append(rotated_poly)
            rotated_masks = PolygonMasks(rotated_masks, *out_shape)
        return rotated_masks

    def to_bitmap(self):
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width)
            )
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width), dtype=dtype)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype)

    @classmethod
    def random(
        cls, num_masks=3, height=32, width=32, n_verts=5, dtype=np.float32, rng=None
    ):
        rng = ensure_rng(rng)

        def _gen_polygon(n, irregularity, spikeyness):
            cx, cy = (0.0, 0.0)
            radius = 1

            tau = np.pi * 2

            irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
            spikeyness = np.clip(spikeyness, 1e-9, 1)
            lower = (tau / n) - irregularity
            upper = (tau / n) + irregularity
            angle_steps = rng.uniform(lower, upper, n)

            k = angle_steps.sum() / (2 * np.pi)
            angles = (angle_steps / k).cumsum() + rng.uniform(0, tau)

            low = 0
            high = 2 * radius
            mean = radius
            std = spikeyness
            a = (low - mean) / std
            b = (high - mean) / std
            tnorm = truncnorm(a=a, b=b, loc=mean, scale=std)
            radii = tnorm.rvs(n, random_state=rng)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)
            points = np.hstack([x_pts[:, None], y_pts[:, None]])
            points = points - points.min(axis=0)
            points = points / points.max(axis=0)
            points = points * (rng.rand() * 0.8 + 0.2)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            high = 1 - max_pt
            low = 0 - min_pt
            offset = (rng.rand(2) * (high - low)) + low
            points = points + offset
            return points

        def _order_vertices(verts):
            mlat = verts.T[0].sum() / len(verts)
            mlng = verts.T[1].sum() / len(verts)

            tau = np.pi * 2
            angle = (np.arctan2(mlat - verts.T[0], verts.T[1] - mlng) + tau) % tau
            sortx = angle.argsort()
            verts = verts.take(sortx, axis=0)
            return verts

        masks = []
        for _ in range(num_masks):
            exterior = _order_vertices(_gen_polygon(n_verts, 0.9, 0.9))
            exterior = (exterior * [(width, height)]).astype(dtype)
            masks.append([exterior.ravel()])

        self = cls(masks, height, width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_list = list(itertools.chain(*[m.masks for m in masks]))
        return cls(mask_list, masks[0].height, masks[0].width)


def polygon_to_bitmap(polygons, height, width):
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask


DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]
box_converters: dict = {}


class BaseBoxes(metaclass=ABCMeta):
    box_dim: int = 0

    def __init__(
        self,
        data: Union[Tensor, np.ndarray, Sequence],
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
        clone: bool = True,
    ) -> None:
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError(
                "boxes should be Tensor, ndarray, or Sequence, ",
                f"but got {type(data)}",
            )

        if device is not None or dtype is not None:
            data = data.to(dtype=dtype)
        if clone:
            data = data.clone()
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))

        assert data.dim() >= 2 and data.size(-1) == self.box_dim, (
            "The boxes dimension must >= 2 and the length of the last "
            f"dimension must be {self.box_dim}, but got boxes with "
            f"shape {data.shape}."
        )
        self.tensor = data

    def convert_to(self, dst_type: Union[str, type]) -> "BaseBoxes":
        return convert_box_type(self, dst_type=dst_type)

    def empty_boxes(
        self: T,
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
    ) -> T:
        empty_box = self.tensor.new_zeros(0, self.box_dim, dtype=dtype)
        return type(self)(empty_box, clone=False)

    def fake_boxes(
        self: T,
        sizes: Tuple[int],
        fill: float = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
    ) -> T:
        fake_boxes = self.tensor.new_full(sizes, fill, dtype=dtype)
        return type(self)(fake_boxes, clone=False)

    def __getitem__(self: T, index: IndexType) -> T:
        boxes = self.tensor
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < boxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < boxes.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        boxes = boxes[index]
        if boxes.dim() == 1:
            boxes = boxes.reshape(1, -1)
        return type(self)(boxes, clone=False)

    def __setitem__(self: T, index: IndexType, values: Union[Tensor, T]) -> T:
        assert type(values) is type(
            self
        ), "The value to be set must be the same box type as self"
        values = values.tensor

        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.tensor.dim()
        elif isinstance(index, tuple):
            assert len(index) < self.tensor.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self.tensor[index] = values

    def __len__(self) -> int:
        return self.tensor.size(0)

    def __deepcopy__(self, memo):

        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        return other

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(\n" + str(self.tensor) + ")"

    def new_tensor(self, *args, **kwargs) -> Tensor:
        return self.tensor.new_tensor(*args, **kwargs)

    def new_full(self, *args, **kwargs) -> Tensor:
        return self.tensor.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs) -> Tensor:
        return self.tensor.new_empty(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> Tensor:
        return self.tensor.new_ones(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> Tensor:
        return self.tensor.new_zeros(*args, **kwargs)

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        return self.tensor.size() if dim is None else self.tensor.size(dim)

    def dim(self) -> int:
        return self.tensor.dim()

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def numel(self) -> int:
        return self.tensor.numel()

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def to(self: T, *args, **kwargs) -> T:
        return type(self)(self.tensor.to(*args, **kwargs), clone=False)

    def cpu(self: T) -> T:
        return type(self)(self.tensor.cpu(), clone=False)

    def cuda(self: T, *args, **kwargs) -> T:
        return type(self)(self.tensor.cuda(*args, **kwargs), clone=False)

    def clone(self: T) -> T:
        return type(self)(self.tensor)

    def detach(self: T) -> T:
        return type(self)(self.tensor.detach(), clone=False)

    def view(self: T, *shape: Tuple[int]) -> T:
        return type(self)(self.tensor.view(shape), clone=False)

    def reshape(self: T, *shape: Tuple[int]) -> T:
        return type(self)(self.tensor.reshape(shape), clone=False)

    def expand(self: T, *sizes: Tuple[int]) -> T:
        return type(self)(self.tensor.expand(sizes), clone=False)

    def repeat(self: T, *sizes: Tuple[int]) -> T:
        return type(self)(self.tensor.repeat(sizes), clone=False)

    def transpose(self: T, dim0: int, dim1: int) -> T:
        ndim = self.tensor.dim()
        assert dim0 != -1 and dim0 != ndim - 1
        assert dim1 != -1 and dim1 != ndim - 1
        return type(self)(self.tensor.transpose(dim0, dim1), clone=False)

    def permute(self: T, *dims: Tuple[int]) -> T:
        assert dims[-1] == -1 or dims[-1] == self.tensor.dim() - 1
        return type(self)(self.tensor.permute(dims), clone=False)

    def split(
        self: T, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0
    ) -> List[T]:
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        assert end_dim != -1 and end_dim != self.tensor.dim() - 1
        return type(self)(self.tensor.flatten(start_dim, end_dim), clone=False)

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        boxes = self.tensor.squeeze() if dim is None else self.tensor.squeeze(dim)
        return type(self)(boxes, clone=False)

    def unsqueeze(self: T, dim: int) -> T:
        assert dim != -1 and dim != self.tensor.dim()
        return type(self)(self.tensor.unsqueeze(dim), clone=False)

    @classmethod
    def cat(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError("box_list should not be a empty list.")

        assert dim != -1 and dim != box_list[0].dim() - 1
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.cat(th_box_list, dim=dim), clone=False)

    @classmethod
    def stack(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError("box_list should not be a empty list.")

        assert dim != -1 and dim != box_list[0].dim()
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.stack(th_box_list, dim=dim), clone=False)

    @abstractproperty
    def centers(self) -> Tensor:
        pass

    @abstractproperty
    def areas(self) -> Tensor:
        pass

    @abstractproperty
    def widths(self) -> Tensor:
        pass

    @abstractproperty
    def heights(self) -> Tensor:
        pass

    @abstractmethod
    def flip_(self, img_shape: Tuple[int, int], direction: str = "horizontal") -> None:
        pass

    @abstractmethod
    def translate_(self, distances: Tuple[float, float]) -> None:
        pass

    @abstractmethod
    def clip_(self, img_shape: Tuple[int, int]) -> None:
        pass

    @abstractmethod
    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        pass

    @abstractmethod
    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        pass

    @abstractmethod
    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        pass

    @abstractmethod
    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        pass

    @abstractmethod
    def is_inside(
        self,
        img_shape: Tuple[int, int],
        all_inside: bool = False,
        allowed_border: int = 0,
    ) -> BoolTensor:
        pass

    @abstractmethod
    def find_inside_points(
        self, points: Tensor, is_aligned: bool = False
    ) -> BoolTensor:
        pass

    @abstractstaticmethod
    def overlaps(
        boxes1: "BaseBoxes",
        boxes2: "BaseBoxes",
        mode: str = "iou",
        is_aligned: bool = False,
        eps: float = 1e-6,
    ) -> Tensor:
        pass

    @abstractstaticmethod
    def from_instance_masks(masks: MaskType) -> "BaseBoxes":
        pass


def samplelist_boxtype2tensor(batch_data_samples: SampleList) -> SampleList:
    for data_samples in batch_data_samples:
        if "gt_instances" in data_samples:
            bboxes = data_samples.gt_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if "pred_instances" in data_samples:
            bboxes = data_samples.pred_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if "ignored_instances" in data_samples:
            bboxes = data_samples.ignored_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor


box_types: dict = {}
_box_type_to_name: dict = {}


def get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert (
            type_name in box_types
        ), f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert (
            box_type in _box_type_to_name
        ), f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError(
            "box_type must be a str or class inheriting from "
            f"BaseBoxes, but got {type(box_type)}."
        )
    return type_name, type_cls


BoxType = Union[np.ndarray, Tensor, BaseBoxes]


def convert_box_type(
    boxes: BoxType,
    *,
    src_type: Union[str, type] = None,
    dst_type: Union[str, type] = None,
) -> BoxType:
    assert dst_type is not None
    dst_type_name, dst_type_cls = get_box_type(dst_type)

    is_box_cls = False
    is_numpy = False
    if isinstance(boxes, BaseBoxes):
        src_type_name, _ = get_box_type(type(boxes))
        is_box_cls = True
    elif isinstance(boxes, (Tensor, np.ndarray)):
        assert src_type is not None
        src_type_name, _ = get_box_type(src_type)
        if isinstance(boxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError(
            "boxes must be a instance of BaseBoxes, Tensor or "
            f"ndarray, but get {type(boxes)}."
        )

    if src_type_name == dst_type_name:
        return boxes

    converter_name = src_type_name + "2" + dst_type_name
    assert (
        converter_name in box_converters
    ), "Convert function hasn't been registered in box_converters."
    converter = box_converters[converter_name]

    if is_box_cls:
        boxes = converter(boxes.tensor)
        return dst_type_cls(boxes)
    elif is_numpy:
        boxes = converter(torch.from_numpy(boxes))
        return boxes.numpy()
    else:
        return converter(boxes)


def digit_version(version_str: str, length: int = 4):
    assert "parrots" not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version_str}"
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {"a": -3, "b": -2, "rc": -1}
        val = -4
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(
                    f"unknown prerelease version {version.pre[0]}, "
                    "version checking may go wrong"
                )
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


def stack_batch(
    tensor_list: List[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: Union[int, float] = 0,
) -> torch.Tensor:
    assert isinstance(
        tensor_list, list
    ), f"Expected input type to be list, but got {type(tensor_list)}"
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = (
        torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    )
    padded_sizes = max_sizes - all_sizes
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(non_blocking)
        assert not (
            bgr_to_rgb and rgb_to_bgr
        ), "`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time"
        assert (mean is None) == (
            std is None
        ), "mean and std should be both None or tuple"
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                "`mean` should have 1 or 3 values, to be compatible with "
                f"RGB or gray image, but got {len(mean)} values"
            )
            assert len(std) == 3 or len(std) == 1, (
                "`std` should have 1 or 3 values, to be compatible with RGB "
                f"or gray image, but got {len(std)} values"
            )
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)
        _batch_inputs = data["inputs"]
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                _batch_input = _batch_input.float()
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, (
                            "If the mean has 3 values, the input tensor "
                            "should in shape of (3, H, W), but got the tensor "
                            f"with shape {_batch_input.shape}"
                        )
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            batch_inputs = stack_batch(
                batch_inputs, self.pad_size_divisor, self.pad_value
            )
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(
                _batch_inputs, (0, pad_w, 0, pad_h), "constant", self.pad_value
            )
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict of "
                "list/tuple with inputs and data_samples, "
                f"but got {type(data)}： {data}"
            )
        data["inputs"] = batch_inputs
        data.setdefault("data_samples", None)
        return data


@MODELS.register_module()
class DetDataPreprocessor(ImgDataPreprocessor):
    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
        )
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments]
            )
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data["inputs"], data["data_samples"]

        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo(
                    {"batch_input_shape": batch_input_shape, "pad_shape": pad_shape}
                )

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "data_samples": data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        _batch_inputs = data["inputs"]
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = (
                    int(np.ceil(ori_input.shape[1] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                pad_w = (
                    int(np.ceil(ori_input.shape[2] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                batch_pad_shape.append((pad_h, pad_w))
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            pad_h = (
                int(np.ceil(_batch_inputs.shape[1] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            pad_w = (
                int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict "
                "or a tuple with inputs and data_samples, but got"
                f"{type(data)}： {data}"
            )
        return batch_pad_shape

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        if "masks" in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape, pad_val=self.mask_pad_value
                )

    def pad_gt_sem_seg(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        if "gt_sem_seg" in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode="constant",
                    value=self.seg_pad_value,
                )
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


@MODELS.register_module()
class YOLOWDetDataPreprocessor(DetDataPreprocessor):
    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]
        assert isinstance(data["data_samples"], dict)

        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{"batch_input_shape": inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            "bboxes_labels": data_samples["bboxes_labels"],
            "texts": data_samples["texts"],
            "img_metas": img_metas,
        }
        if "masks" in data_samples:
            data_samples_output["masks"] = data_samples["masks"]
        if "is_detection" in data_samples:
            data_samples_output["is_detection"] = data_samples["is_detection"]

        return {"inputs": inputs, "data_samples": data_samples_output}


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


try:
    import torch_npu

    npu_jit_compile = bool(os.getenv("NPUJITCompile", False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False


def is_npu_available() -> bool:
    return IS_NPU_AVAILABLE


DEVICE = "cpu"
if is_npu_available():
    DEVICE = "npu"
elif is_cuda_available():
    DEVICE = "cuda"


def get_device() -> str:
    return DEVICE


@contextmanager
def autocast(
    device_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
    cache_enabled: Optional[bool] = None,
):
    assert digit_version(TORCH_VERSION) >= digit_version("1.5.0"), (
        "The minimum pytorch version requirements of mmengine is 1.5.0, but "
        f"got {TORCH_VERSION}"
    )

    if digit_version("1.5.0") <= digit_version(TORCH_VERSION) < digit_version("1.10.0"):
        assert device_type == "cuda" or device_type is None, (
            "Pytorch version under 1.10.0 only supports running automatic "
            "mixed training with cuda"
        )
        if dtype is not None or cache_enabled is not None:
            print_log(
                f"{dtype} and {device_type} will not work for "
                "`autocast` since your Pytorch version: "
                f"{TORCH_VERSION} <= 1.10.0",
                logger="current",
                level=logging.WARNING,
            )

        if is_npu_available():
            with torch.npu.amp.autocast(enabled=enabled):
                yield
        elif is_cuda_available():
            with torch.cuda.amp.autocast(enabled=enabled):
                yield
        else:
            if not enabled:
                yield
            else:
                raise RuntimeError(
                    "If pytorch versions is between 1.5.0 and 1.10, "
                    "`autocast` is only available in gpu mode"
                )

    else:
        if cache_enabled is None:
            cache_enabled = torch.is_autocast_cache_enabled()
        device = get_device()
        device_type = device if device_type is None else device_type

        if device_type == "cuda":
            if dtype is None:
                dtype = torch.get_autocast_gpu_dtype()

            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "Current CUDA Device does not support bfloat16. Please "
                    "switch dtype to float16."
                )

        elif device_type == "cpu":
            if dtype is None:
                dtype = torch.bfloat16
            assert (
                dtype == torch.bfloat16
            ), "In CPU autocast, only support `torch.bfloat16` dtype"

        elif device_type == "mlu":
            pass

        elif device_type == "npu":
            pass

        else:
            if enabled is False:
                yield
                return
            else:
                raise ValueError(
                    "User specified autocast device_type must be "
                    f"cuda or cpu, but got {device_type}"
                )

        with torch.autocast(
            enabled=enabled,
            dtype=dtype,
            cache_enabled=cache_enabled,
        ):
            yield


class TimerError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class Timer:
    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        if start:
            self.start()

    @property
    def is_running(self):
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._t_start = ti()
            self._is_running = True
        self._t_last = ti()

    def since_start(self):
        if not self._is_running:
            raise TimerError("timer is not running")
        self._t_last = ti()
        return self._t_last - self._t_start

    def since_last_check(self):
        if not self._is_running:
            raise TimerError("timer is not running")
        dur = ti() - self._t_last
        self._t_last = ti()
        return dur


class ProgressBar:
    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(
                f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:"
            )
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(
                self.bar_width,
                int(self.terminal_width - len(msg)) + 2,
                int(self.terminal_width * 0.6),
            )
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,"
                f" {fps:.1f} tasks/s"
            )
        self.file.flush()


def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if "pipeline" in dataset_cfg:
            return dataset_cfg.pipeline
        elif "dataset" in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        elif "datasets" in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError("Cannot find `pipeline` in `test_dataloader`")

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)


def deprecated_api_warning(name_dict: dict, cls_name: Optional[str] = None) -> Callable:
    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if args:
                arg_names = args_info.args[: len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead",
                            DeprecationWarning,
                        )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f"The expected behavior is to replace "
                            f"the deprecated key `{src_arg_name}` to "
                            f"new key `{dst_arg_name}`, but got them "
                            f"in the arguments at the same time, which "
                            f"is confusing. `{src_arg_name} will be "
                            f"deprecated in the future, please "
                            f"use `{dst_arg_name}` instead."
                        )

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead",
                            DeprecationWarning,
                        )
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def _to_torch(x: ArrayOrTensor, device=None, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    raise TypeError("Expected np.ndarray or torch.Tensor")


def nms(
    boxes: ArrayOrTensor,
    scores: ArrayOrTensor,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0.0,
    max_num: int = -1,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    boxes_t = _to_torch(boxes)
    scores_t = _to_torch(scores)

    assert boxes_t.ndim == 2 and boxes_t.size(1) == 4
    assert scores_t.ndim == 1 and boxes_t.size(0) == scores_t.size(0)
    assert offset in (0, 1)

    device = boxes_t.device
    dtype = boxes_t.dtype
    if score_threshold > 0:
        keep_mask = scores_t > float(score_threshold)
        if keep_mask.sum().item() == 0:
            empty_dets = boxes_t.new_zeros((0, 5))
            empty_inds = torch.empty((0,), dtype=torch.long)
            return (
                empty_dets
                if isinstance(boxes, torch.Tensor)
                else empty_dets.cpu().numpy()
            ), (
                empty_inds
                if isinstance(scores, torch.Tensor)
                else empty_inds.cpu().numpy()
            )
        boxes_t = boxes_t[keep_mask]
        scores_t = scores_t[keep_mask]
        orig_inds = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    else:
        orig_inds = torch.arange(boxes_t.size(0), dtype=torch.long)

    x1 = boxes_t[:, 0]
    y1 = boxes_t[:, 1]
    x2 = boxes_t[:, 2]
    y2 = boxes_t[:, 3]

    areas = (x2 - x1 + offset).clamp(min=0) * (y2 - y1 + offset).clamp(min=0)

    _, order = scores_t.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        inter_w = (xx2 - xx1 + offset).clamp(min=0)
        inter_h = (yy2 - yy1 + offset).clamp(min=0)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter
        iou = torch.zeros_like(inter)
        mask = union > 0
        iou[mask] = inter[mask] / union[mask]

        rem_mask = iou <= iou_threshold
        order = order[1:][rem_mask]

    keep = torch.as_tensor(keep, dtype=torch.long)
    final_inds = orig_inds[keep]

    if max_num > 0 and final_inds.numel() > max_num:
        final_inds = final_inds[:max_num]
        keep = keep[:max_num]

    final_boxes = boxes_t[keep]
    final_scores = scores_t[keep].unsqueeze(1)
    dets = torch.cat([final_boxes, final_scores], dim=1)

    if isinstance(boxes, np.ndarray) or isinstance(scores, np.ndarray):
        return dets.cpu().numpy(), final_inds.cpu().numpy()
    return dets, final_inds


def remove_delete_keys(cfg):
    if isinstance(cfg, dict):
        cfg.pop("_delete_", None)
        for v in cfg.values():
            remove_delete_keys(v)
    elif isinstance(cfg, list):
        for item in cfg:
            remove_delete_keys(item)


def simple_recursive_merge(
    base: Dict[str, Any], *updates: Dict[str, Any]
) -> Dict[str, Any]:
    def clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            kk: copy.deepcopy(vv)
            for kk, vv in d.items()
            if kk not in ("_delete", "_delete_")
        }

    def merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if k not in dst:
                if isinstance(v, dict) and (
                    v.get("_delete") is True or v.get("_delete_") is True
                ):
                    dst[k] = clean(v)
                else:
                    dst[k] = copy.deepcopy(v)
                continue

            if isinstance(v, dict) and (
                v.get("_delete") is True or v.get("_delete_") is True
            ):
                dst[k] = clean(v)
                continue

            if isinstance(dst[k], dict) and isinstance(v, dict):
                merge(dst[k], v)
                continue

            dst[k] = copy.deepcopy(v)

    for upd in updates:
        merge(base, upd)

    return base


def get_s_coco_init_params(base_cfg):
    update_cfg = dict(
        num_classes=80,
        img_scale=(640, 640),
        deepen_factor=0.33,
        widen_factor=0.5,
        strides=[8, 16, 32],
        last_stage_out_channels=1024,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        loss_cls_weight=0.5,
        loss_bbox_weight=7.5,
        loss_dfl_weight=1.5 / 4,
    )
    simple_recursive_merge(base_cfg, update_cfg)
    return base_cfg


def get_s_variant_init_params(base_cfg):
    if base_cfg["variant"].endswith("640"):
        override_cfg = dict(
            num_classes=1203,
            num_training_classes=80,
            text_channels=512,
            neck_embed_channels=[128, 256, base_cfg["last_stage_out_channels"] // 2],
            neck_num_heads=[4, 8, base_cfg["last_stage_out_channels"] // 2 // 32],
            img_scale=(640, 640),
        )
    else:
        override_cfg = dict(
            num_classes=1203,
            num_training_classes=80,
            text_channels=512,
            neck_embed_channels=[128, 256, base_cfg["last_stage_out_channels"] // 2],
            neck_num_heads=[4, 8, base_cfg["last_stage_out_channels"] // 2 // 32],
            img_scale=(1280, 1280),
        )
    simple_recursive_merge(base_cfg, override_cfg)
    return base_cfg


def get_s_coco_module_params(base_cfg):
    model = dict(
        type="YOLODetector",
        data_preprocessor=dict(
            type="YOLOv5DetDataPreprocessor",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True,
        ),
        backbone=dict(
            type="YOLOv8CSPDarknet",
            arch="P5",
            last_stage_out_channels=base_cfg["last_stage_out_channels"],
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
            norm_cfg=base_cfg["norm_cfg"],
            act_cfg=dict(type="SiLU", inplace=True),
        ),
        neck=dict(
            type="YOLOv8PAFPN",
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
            in_channels=[256, 512, base_cfg["last_stage_out_channels"]],
            out_channels=[256, 512, base_cfg["last_stage_out_channels"]],
            num_csp_blocks=3,
            norm_cfg=base_cfg["norm_cfg"],
            act_cfg=dict(type="SiLU", inplace=True),
        ),
        bbox_head=dict(
            type="YOLOv8Head",
            head_module=dict(
                type="YOLOv8HeadModule",
                num_classes=base_cfg["num_classes"],
                in_channels=[256, 512, base_cfg["last_stage_out_channels"]],
                widen_factor=base_cfg["widen_factor"],
                reg_max=16,
                norm_cfg=base_cfg["norm_cfg"],
                act_cfg=dict(type="SiLU", inplace=True),
                featmap_strides=base_cfg["strides"],
            ),
            prior_generator=dict(
                type="MlvlPointGenerator", offset=0.5, strides=base_cfg["strides"]
            ),
            bbox_coder=dict(type="DistancePointBBoxCoder"),
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="none",
                loss_weight=base_cfg["loss_cls_weight"],
            ),
            loss_bbox=dict(
                type="IoULoss",
                iou_mode="ciou",
                bbox_format="xyxy",
                reduction="sum",
                loss_weight=base_cfg["loss_bbox_weight"],
                return_iou=False,
            ),
            loss_dfl=dict(
                type="DistributionFocalLoss",
                reduction="mean",
                loss_weight=base_cfg["loss_dfl_weight"],
            ),
        ),
    )

    simple_recursive_merge(base_cfg, {"model": model})

    base_cfg["test_pipeline"] = [
        dict(type="LoadImageFromFile", backend_args=base_cfg.get("backend_args")),
        dict(type="YOLOv5KeepRatioResize", scale=base_cfg["img_scale"]),
        dict(
            type="LetterResize",
            scale=base_cfg["img_scale"],
            allow_scale_up=False,
            pad_val=dict(img=114),
        ),
        dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "pad_param",
            ),
        ),
    ]

    return base_cfg


def get_s_variant_module_params(base_cfg):
    backbone_copy = copy.deepcopy(base_cfg["model"]["backbone"])

    model = dict(
        type="YOLOWorldDetector",
        mm_neck=True,
        num_train_classes=base_cfg["num_training_classes"],
        num_test_classes=base_cfg["num_classes"],
        data_preprocessor=dict(type="YOLOWDetDataPreprocessor"),
        backbone=dict(
            _delete_=True,
            type="MultiModalYOLOBackbone",
            image_model=backbone_copy,
            text_model=dict(
                type="HuggingCLIPLanguageBackbone",
                model_name="openai/clip-vit-base-patch32",
                frozen_modules=["all"],
            ),
        ),
        neck=dict(
            type="YOLOWorldPAFPN",
            guide_channels=base_cfg["text_channels"],
            embed_channels=base_cfg["neck_embed_channels"],
            num_heads=base_cfg["neck_num_heads"],
            block_cfg=dict(type="MaxSigmoidCSPLayerWithTwoConv"),
        ),
        bbox_head=dict(
            type="YOLOWorldHead",
            head_module=dict(
                type="YOLOWorldHeadModule",
                use_bn_head=True,
                embed_dims=base_cfg["text_channels"],
                num_classes=base_cfg["num_training_classes"],
            ),
        ),
    )

    simple_recursive_merge(base_cfg, {"model": model})

    base_cfg["test_pipeline"] = [
        *base_cfg["test_pipeline"][:-1],
        dict(type="LoadText"),
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "pad_param",
                "texts",
            ),
        ),
    ]
    base_cfg["coco_val_dataset"] = dict(
        _delete_=True,
        type="MultiModalDataset",
        dataset=dict(
            type="YOLOv5LVISV1Dataset",
            data_root="data/coco/",
            test_mode=True,
            ann_file=get_file(
                "https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json"
            ),
            data_prefix=dict(img=""),
            batch_shapes_cfg=None,
        ),
        class_text_path="data/texts/lvis_v1_class_texts.json",
        pipeline=base_cfg["test_pipeline"],
    )

    base_cfg["val_dataloader"] = dict(dataset=base_cfg["coco_val_dataset"])
    base_cfg["test_dataloader"] = base_cfg["val_dataloader"]

    return base_cfg


def get_m_coco_init_params(base_cfg):
    m_coco_init_cfg = dict(
        deepen_factor=0.67,
        widen_factor=0.75,
        last_stage_out_channels=768,
        img_scale=base_cfg["img_scale"],
    )
    simple_recursive_merge(base_cfg, m_coco_init_cfg)
    return base_cfg


def get_m_coco_module_params(base_cfg):
    model = dict(
        backbone=dict(
            last_stage_out_channels=base_cfg["last_stage_out_channels"],
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
        ),
        neck=dict(
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
            in_channels=[256, 512, base_cfg["last_stage_out_channels"]],
            out_channels=[256, 512, base_cfg["last_stage_out_channels"]],
        ),
        bbox_head=dict(
            head_module=dict(
                widen_factor=base_cfg["widen_factor"],
                in_channels=[256, 512, base_cfg["last_stage_out_channels"]],
            )
        ),
    )
    simple_recursive_merge(base_cfg, {"model": model})

    return base_cfg


def get_m_variant_init_params(base_cfg):
    common_override_cfg = dict(
        num_classes=1203,
        num_training_classes=80,
        text_channels=512,
        neck_embed_channels=[128, 256, base_cfg["last_stage_out_channels"] // 2],
        neck_num_heads=[4, 8, base_cfg["last_stage_out_channels"] // 2 // 32],
        text_model_name="openai/clip-vit-base-patch32",
    )
    if base_cfg["variant"].endswith("640"):
        base_cfg["img_scale"] = (640, 640)

    else:
        base_cfg["img_scale"] = (1280, 1280)
    simple_recursive_merge(base_cfg, common_override_cfg)
    return base_cfg


def get_m_variant_module_params(base_cfg):
    return get_s_variant_module_params(base_cfg)


def get_l_coco_init_params(base_cfg):
    l_coco_init_cfg = dict(
        deepen_factor=1.00,
        widen_factor=1.00,
        last_stage_out_channels=512,
    )
    simple_recursive_merge(base_cfg, l_coco_init_cfg)
    return base_cfg


def get_l_coco_module_params(base_cfg):
    return get_m_coco_module_params(base_cfg)


def get_l_variant_init_params(base_cfg):
    return get_m_variant_init_params(base_cfg)


def get_l_variant_module_params(base_cfg):
    return get_s_variant_module_params(base_cfg)


def get_xl_coco_init_params(base_cfg):
    xl_coco_init_cfg = dict(
        deepen_factor=1.00,
        widen_factor=1.25,
    )
    simple_recursive_merge(base_cfg, xl_coco_init_cfg)
    return base_cfg


def get_xl_coco_module_params(base_cfg):
    model = dict(
        backbone=dict(
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
        ),
        neck=dict(
            deepen_factor=base_cfg["deepen_factor"],
            widen_factor=base_cfg["widen_factor"],
        ),
        bbox_head=dict(head_module=dict(widen_factor=base_cfg["widen_factor"])),
    )
    simple_recursive_merge(base_cfg, {"model": model})
    return base_cfg


def get_xl_variant_init_params(base_cfg):
    return get_m_variant_init_params(base_cfg)


def get_xl_variant_module_params(base_cfg):
    return get_s_variant_module_params(base_cfg)


def get_base_cfg(variant="small_640"):
    if hasattr(variant, "value"):
        variant = variant.value
    base_cfg = dict(
        variant=variant,
    )
    if base_cfg["variant"].startswith("small"):
        base_cfg = get_s_coco_init_params(base_cfg)
        base_cfg = get_s_variant_init_params(base_cfg)
        base_cfg = get_s_coco_module_params(base_cfg)
        base_cfg = get_m_variant_module_params(base_cfg)

    if base_cfg["variant"].startswith("medium"):
        base_cfg = get_s_coco_init_params(base_cfg)
        base_cfg = get_m_coco_init_params(base_cfg)
        base_cfg = get_m_variant_init_params(base_cfg)
        base_cfg = get_s_coco_module_params(base_cfg)
        base_cfg = get_m_coco_module_params(base_cfg)
        base_cfg = get_m_variant_module_params(base_cfg)

    if base_cfg["variant"].startswith("large"):
        base_cfg = get_s_coco_init_params(base_cfg)
        base_cfg = get_m_coco_init_params(base_cfg)
        base_cfg = get_l_coco_init_params(base_cfg)
        base_cfg = get_l_variant_init_params(base_cfg)
        base_cfg = get_s_coco_module_params(base_cfg)
        base_cfg = get_m_coco_module_params(base_cfg)
        base_cfg = get_l_coco_module_params(base_cfg)
        base_cfg = get_l_variant_module_params(base_cfg)

    if base_cfg["variant"].startswith("xlarge"):
        base_cfg = get_s_coco_init_params(base_cfg)
        base_cfg = get_m_coco_init_params(base_cfg)
        base_cfg = get_l_coco_init_params(base_cfg)
        base_cfg = get_xl_coco_init_params(base_cfg)
        base_cfg = get_xl_variant_init_params(base_cfg)
        base_cfg = get_s_coco_module_params(base_cfg)
        base_cfg = get_m_coco_module_params(base_cfg)
        base_cfg = get_l_coco_module_params(base_cfg)
        base_cfg = get_xl_coco_module_params(base_cfg)
        base_cfg = get_xl_variant_module_params(base_cfg)

    remove_delete_keys(base_cfg)
    return base_cfg


CLASS_TEXTS = [
    ["aerosol can", "spray can"],
    ["air conditioner"],
    ["airplane", "aeroplane"],
    ["alarm clock"],
    ["alcohol", "alcoholic beverage"],
    ["alligator", "gator"],
    ["almond"],
    ["ambulance"],
    ["amplifier"],
    ["anklet", "ankle bracelet"],
    ["antenna", "aerial", "transmitting aerial"],
    ["apple"],
    ["applesauce"],
    ["apricot"],
    ["apron"],
    ["aquarium", "fish tank"],
    [
        "arctic",
        "arctic type of shoe",
        "galosh",
        "golosh",
        "rubber",
        "rubber type of shoe",
        "gumshoe",
    ],
    ["armband"],
    ["armchair"],
    ["armoire"],
    ["armor", "armour"],
    ["artichoke"],
    ["trash can", "garbage can", "wastebin", "dustbin", "trash barrel", "trash bin"],
    ["ashtray"],
    ["asparagus"],
    ["atomizer", "atomiser", "spray", "sprayer", "nebulizer", "nebuliser"],
    ["avocado"],
    ["award", "accolade"],
    ["awning"],
    ["ax", "axe"],
    ["baboon"],
    ["baby buggy", "baby carriage", "perambulator", "pram", "stroller"],
    ["basketball backboard"],
    ["backpack", "knapsack", "packsack", "rucksack", "haversack"],
    ["handbag", "purse", "pocketbook"],
    ["suitcase", "baggage", "luggage"],
    ["bagel", "beigel"],
    ["bagpipe"],
    ["baguet", "baguette"],
    ["bait", "lure"],
    ["ball"],
    ["ballet skirt", "tutu"],
    ["balloon"],
    ["bamboo"],
    ["banana"],
    ["Band Aid"],
    ["bandage"],
    ["bandanna", "bandana"],
    ["banjo"],
    ["banner", "streamer"],
    ["barbell"],
    ["barge"],
    ["barrel", "cask"],
    ["barrette"],
    ["barrow", "garden cart", "lawn cart", "wheelbarrow"],
    ["baseball base"],
    ["baseball"],
    ["baseball bat"],
    ["baseball cap", "jockey cap", "golf cap"],
    ["baseball glove", "baseball mitt"],
    ["basket", "handbasket"],
    ["basketball"],
    ["bass horn", "sousaphone", "tuba"],
    ["bat", "bat animal"],
    ["bath mat"],
    ["bath towel"],
    ["bathrobe"],
    ["bathtub", "bathing tub"],
    ["batter", "batter food"],
    ["battery"],
    ["beachball"],
    ["bead"],
    ["bean curd", "tofu"],
    ["beanbag"],
    ["beanie", "beany"],
    ["bear"],
    ["bed"],
    ["bedpan"],
    ["bedspread", "bedcover", "bed covering", "counterpane", "spread"],
    ["cow"],
    ["beef", "beef food", "boeuf", "boeuf food"],
    ["beeper", "pager"],
    ["beer bottle"],
    ["beer can"],
    ["beetle"],
    ["bell"],
    ["bell pepper", "capsicum"],
    ["belt"],
    ["belt buckle"],
    ["bench"],
    ["beret"],
    ["bib"],
    ["Bible"],
    ["bicycle", "bike", "bike bicycle"],
    ["visor", "vizor"],
    ["billboard"],
    ["binder", "ring-binder"],
    ["binoculars", "field glasses", "opera glasses"],
    ["bird"],
    ["birdfeeder"],
    ["birdbath"],
    ["birdcage"],
    ["birdhouse"],
    ["birthday cake"],
    ["birthday card"],
    ["pirate flag"],
    ["black sheep"],
    ["blackberry"],
    ["blackboard", "chalkboard"],
    ["blanket"],
    ["blazer", "sport jacket", "sport coat", "sports jacket", "sports coat"],
    ["blender", "liquidizer", "liquidiser"],
    ["blimp"],
    ["blinker", "flasher"],
    ["blouse"],
    ["blueberry"],
    ["gameboard"],
    ["boat", "ship", "ship boat"],
    ["bob", "bobber", "bobfloat"],
    ["bobbin", "spool", "reel"],
    ["bobby pin", "hairgrip"],
    ["boiled egg", "coddled egg"],
    ["bolo tie", "bolo", "bola tie", "bola"],
    ["deadbolt"],
    ["bolt"],
    ["bonnet"],
    ["book"],
    ["bookcase"],
    ["booklet", "brochure", "leaflet", "pamphlet"],
    ["bookmark", "bookmarker"],
    ["boom microphone", "microphone boom"],
    ["boot"],
    ["bottle"],
    ["bottle opener"],
    ["bouquet"],
    ["bow", "bow weapon"],
    ["bow", "bow decorative ribbons"],
    ["bow-tie", "bowtie"],
    ["bowl"],
    ["pipe bowl"],
    ["bowler hat", "bowler", "derby hat", "derby", "plug hat"],
    ["bowling ball"],
    ["box"],
    ["boxing glove"],
    ["suspenders"],
    ["bracelet", "bangle"],
    ["brass plaque"],
    ["brassiere", "bra", "bandeau"],
    ["bread-bin", "breadbox"],
    ["bread"],
    ["breechcloth", "breechclout", "loincloth"],
    ["bridal gown", "wedding gown", "wedding dress"],
    ["briefcase"],
    ["broccoli"],
    ["broach"],
    ["broom"],
    ["brownie"],
    ["brussels sprouts"],
    ["bubble gum"],
    ["bucket", "pail"],
    ["horse buggy"],
    ["horned cow"],
    ["bulldog"],
    ["bulldozer", "dozer"],
    ["bullet train"],
    ["bulletin board", "notice board"],
    ["bulletproof vest"],
    ["bullhorn", "megaphone"],
    ["bun", "roll"],
    ["bunk bed"],
    ["buoy"],
    ["burrito"],
    [
        "bus",
        "bus vehicle",
        "autobus",
        "charabanc",
        "double-decker",
        "motorbus",
        "motorcoach",
    ],
    ["business card"],
    ["butter"],
    ["butterfly"],
    ["button"],
    ["cab", "cab taxi", "taxi", "taxicab"],
    ["cabana"],
    ["cabin car", "caboose"],
    ["cabinet"],
    ["locker", "storage locker"],
    ["cake"],
    ["calculator"],
    ["calendar"],
    ["calf"],
    ["camcorder"],
    ["camel"],
    ["camera"],
    ["camera lens"],
    ["camper", "camper vehicle", "camping bus", "motor home"],
    ["can", "tin can"],
    ["can opener", "tin opener"],
    ["candle", "candlestick"],
    ["candle holder"],
    ["candy bar"],
    ["candy cane"],
    ["walking cane"],
    ["canister", "cannister"],
    ["canoe"],
    ["cantaloup", "cantaloupe"],
    ["canteen"],
    ["cap", "cap headwear"],
    ["bottle cap", "cap", "cap container lid"],
    ["cape"],
    ["cappuccino", "coffee cappuccino"],
    ["car", "car automobile", "auto", "auto automobile", "automobile"],
    [
        "railcar",
        "railcar part of a train",
        "railway car",
        "railway car part of a train",
        "railroad car",
        "railroad car part of a train",
    ],
    ["elevator car"],
    ["car battery", "automobile battery"],
    ["identity card"],
    ["card"],
    ["cardigan"],
    ["cargo ship", "cargo vessel"],
    ["carnation"],
    ["horse carriage"],
    ["carrot"],
    ["tote bag"],
    ["cart"],
    ["carton"],
    ["cash register", "register", "register for cash transactions"],
    ["casserole"],
    ["cassette"],
    ["cast", "plaster cast", "plaster bandage"],
    ["cat"],
    ["cauliflower"],
    [
        "cayenne",
        "cayenne spice",
        "cayenne pepper",
        "cayenne pepper spice",
        "red pepper",
        "red pepper spice",
    ],
    ["CD player"],
    ["celery"],
    [
        "cellular telephone",
        "cellular phone",
        "cellphone",
        "mobile phone",
        "smart phone",
    ],
    [
        "chain mail",
        "ring mail",
        "chain armor",
        "chain armour",
        "ring armor",
        "ring armour",
    ],
    ["chair"],
    ["chaise longue", "chaise", "daybed"],
    ["chalice"],
    ["chandelier"],
    ["chap"],
    ["checkbook", "chequebook"],
    ["checkerboard"],
    ["cherry"],
    ["chessboard"],
    ["chicken", "chicken animal"],
    ["chickpea", "garbanzo"],
    [
        "chili",
        "chili vegetable",
        "chili pepper",
        "chili pepper vegetable",
        "chilli",
        "chilli vegetable",
        "chilly",
        "chilly vegetable",
        "chile",
        "chile vegetable",
    ],
    ["chime", "gong"],
    ["chinaware"],
    ["crisp", "crisp potato chip", "potato chip"],
    ["poker chip"],
    ["chocolate bar"],
    ["chocolate cake"],
    ["chocolate milk"],
    ["chocolate mousse"],
    ["choker", "collar", "neckband"],
    ["chopping board", "cutting board", "chopping block"],
    ["chopstick"],
    ["Christmas tree"],
    ["slide"],
    ["cider", "cyder"],
    ["cigar box"],
    ["cigarette"],
    ["cigarette case", "cigarette pack"],
    ["cistern", "water tank"],
    ["clarinet"],
    ["clasp"],
    ["cleansing agent", "cleanser", "cleaner"],
    ["cleat", "cleat for securing rope"],
    ["clementine"],
    ["clip"],
    ["clipboard"],
    ["clippers", "clippers for plants"],
    ["cloak"],
    ["clock", "timepiece", "timekeeper"],
    ["clock tower"],
    ["clothes hamper", "laundry basket", "clothes basket"],
    ["clothespin", "clothes peg"],
    ["clutch bag"],
    ["coaster"],
    ["coat"],
    ["coat hanger", "clothes hanger", "dress hanger"],
    ["coatrack", "hatrack"],
    ["cock", "rooster"],
    ["cockroach"],
    [
        "cocoa",
        "cocoa beverage",
        "hot chocolate",
        "hot chocolate beverage",
        "drinking chocolate",
    ],
    ["coconut", "cocoanut"],
    ["coffee maker", "coffee machine"],
    ["coffee table", "cocktail table"],
    ["coffeepot"],
    ["coil"],
    ["coin"],
    ["colander", "cullender"],
    ["coleslaw", "slaw"],
    ["coloring material", "colouring material"],
    ["combination lock"],
    ["pacifier", "teething ring"],
    ["comic book"],
    ["compass"],
    ["computer keyboard", "keyboard", "keyboard computer"],
    ["condiment"],
    ["cone", "traffic cone"],
    ["control", "controller"],
    ["convertible", "convertible automobile"],
    ["sofa bed"],
    ["cooker"],
    ["cookie", "cooky", "biscuit", "biscuit cookie"],
    ["cooking utensil"],
    ["cooler", "cooler for food", "ice chest"],
    ["cork", "cork bottle plug", "bottle cork"],
    ["corkboard"],
    ["corkscrew", "bottle screw"],
    ["edible corn", "corn", "maize"],
    ["cornbread"],
    ["cornet", "horn", "trumpet"],
    ["cornice", "valance", "valance board", "pelmet"],
    ["cornmeal"],
    ["corset", "girdle"],
    ["costume"],
    ["cougar", "puma", "catamount", "mountain lion", "panther"],
    ["coverall"],
    ["cowbell"],
    ["cowboy hat", "ten-gallon hat"],
    ["crab", "crab animal"],
    ["crabmeat"],
    ["cracker"],
    ["crape", "crepe", "French pancake"],
    ["crate"],
    ["crayon", "wax crayon"],
    ["cream pitcher"],
    ["crescent roll", "croissant"],
    ["crib", "cot"],
    ["crock pot", "earthenware jar"],
    ["crossbar"],
    ["crouton"],
    ["crow"],
    ["crowbar", "wrecking bar", "pry bar"],
    ["crown"],
    ["crucifix"],
    ["cruise ship", "cruise liner"],
    ["police cruiser", "patrol car", "police car", "squad car"],
    ["crumb"],
    ["crutch"],
    ["cub", "cub animal"],
    ["cube", "square block"],
    ["cucumber", "cuke"],
    ["cufflink"],
    ["cup"],
    ["trophy cup"],
    ["cupboard", "closet"],
    ["cupcake"],
    ["hair curler", "hair roller", "hair crimper"],
    ["curling iron"],
    ["curtain", "drapery"],
    ["cushion"],
    ["cylinder"],
    ["cymbal"],
    ["dagger"],
    ["dalmatian"],
    ["dartboard"],
    ["date", "date fruit"],
    ["deck chair", "beach chair"],
    ["deer", "cervid"],
    ["dental floss", "floss"],
    ["desk"],
    ["detergent"],
    ["diaper"],
    ["diary", "journal"],
    ["die", "dice"],
    ["dinghy", "dory", "rowboat"],
    ["dining table"],
    ["tux", "tuxedo"],
    ["dish"],
    ["dish antenna"],
    ["dishrag", "dishcloth"],
    ["dishtowel", "tea towel"],
    ["dishwasher", "dishwashing machine"],
    ["dishwasher detergent", "dishwashing detergent", "dishwashing liquid", "dishsoap"],
    ["dispenser"],
    ["diving board"],
    ["Dixie cup", "paper cup"],
    ["dog"],
    ["dog collar"],
    ["doll"],
    ["dollar", "dollar bill", "one dollar bill"],
    ["dollhouse", "doll's house"],
    ["dolphin"],
    ["domestic ass", "donkey"],
    ["doorknob", "doorhandle"],
    ["doormat", "welcome mat"],
    ["doughnut", "donut"],
    ["dove"],
    ["dragonfly"],
    ["drawer"],
    ["underdrawers", "boxers", "boxershorts"],
    ["dress", "frock"],
    ["dress hat", "high hat", "opera hat", "silk hat", "top hat"],
    ["dress suit"],
    ["dresser"],
    ["drill"],
    ["drone"],
    ["dropper", "eye dropper"],
    ["drum", "drum musical instrument"],
    ["drumstick"],
    ["duck"],
    ["duckling"],
    ["duct tape"],
    ["duffel bag", "duffle bag", "duffel", "duffle"],
    ["dumbbell"],
    ["dumpster"],
    ["dustpan"],
    ["eagle"],
    ["earphone", "earpiece", "headphone"],
    ["earplug"],
    ["earring"],
    ["easel"],
    ["eclair"],
    ["eel"],
    ["egg", "eggs"],
    ["egg roll", "spring roll"],
    ["egg yolk", "yolk", "yolk egg"],
    ["eggbeater", "eggwhisk"],
    ["eggplant", "aubergine"],
    ["electric chair"],
    ["refrigerator"],
    ["elephant"],
    ["elk", "moose"],
    ["envelope"],
    ["eraser"],
    ["escargot"],
    ["eyepatch"],
    ["falcon"],
    ["fan"],
    ["faucet", "spigot", "tap"],
    ["fedora"],
    ["ferret"],
    ["Ferris wheel"],
    ["ferry", "ferryboat"],
    ["fig", "fig fruit"],
    ["fighter jet", "fighter aircraft", "attack aircraft"],
    ["figurine"],
    ["file cabinet", "filing cabinet"],
    ["file", "file tool"],
    ["fire alarm", "smoke alarm"],
    ["fire engine", "fire truck"],
    ["fire extinguisher", "extinguisher"],
    ["fire hose"],
    ["fireplace"],
    ["fireplug", "fire hydrant", "hydrant"],
    ["first-aid kit"],
    ["fish"],
    ["fish", "fish food"],
    ["fishbowl", "goldfish bowl"],
    ["fishing rod", "fishing pole"],
    ["flag"],
    ["flagpole", "flagstaff"],
    ["flamingo"],
    ["flannel"],
    ["flap"],
    ["flash", "flashbulb"],
    ["flashlight", "torch"],
    ["fleece"],
    ["flip-flop", "flip-flop sandal"],
    ["flipper", "flipper footwear", "fin", "fin footwear"],
    ["flower arrangement", "floral arrangement"],
    ["flute glass", "champagne flute"],
    ["foal"],
    ["folding chair"],
    ["food processor"],
    ["football", "football American"],
    ["football helmet"],
    ["footstool", "footrest"],
    ["fork"],
    ["forklift"],
    ["freight car"],
    ["French toast"],
    ["freshener", "air freshener"],
    ["frisbee"],
    ["frog", "toad", "toad frog"],
    ["fruit juice"],
    ["frying pan", "frypan", "skillet"],
    ["fudge"],
    ["funnel"],
    ["futon"],
    ["gag", "muzzle"],
    ["garbage"],
    ["garbage truck"],
    ["garden hose"],
    ["gargle", "mouthwash"],
    ["gargoyle"],
    ["garlic", "ail"],
    ["gasmask", "respirator", "gas helmet"],
    ["gazelle"],
    ["gelatin", "jelly"],
    ["gemstone"],
    ["generator"],
    ["giant panda", "panda", "panda bear"],
    ["gift wrap"],
    ["ginger", "gingerroot"],
    ["giraffe"],
    ["cincture", "sash", "waistband", "waistcloth"],
    ["glass", "glass drink container", "drinking glass"],
    ["globe"],
    ["glove"],
    ["goat"],
    ["goggles"],
    ["goldfish"],
    ["golf club", "golf-club"],
    ["golfcart"],
    ["gondola", "gondola boat"],
    ["goose"],
    ["gorilla"],
    ["gourd"],
    ["grape"],
    ["grater"],
    ["gravestone", "headstone", "tombstone"],
    ["gravy boat", "gravy holder"],
    ["green bean"],
    ["green onion", "spring onion", "scallion"],
    ["griddle"],
    ["grill", "grille", "grillwork", "radiator grille"],
    ["grits", "hominy grits"],
    ["grizzly", "grizzly bear"],
    ["grocery bag"],
    ["guitar"],
    ["gull", "seagull"],
    ["gun"],
    ["hairbrush"],
    ["hairnet"],
    ["hairpin"],
    ["halter top"],
    ["ham", "jambon", "gammon"],
    ["hamburger", "beefburger", "burger"],
    ["hammer"],
    ["hammock"],
    ["hamper"],
    ["hamster"],
    ["hair dryer"],
    ["hand glass", "hand mirror"],
    ["hand towel", "face towel"],
    ["handcart", "pushcart", "hand truck"],
    ["handcuff"],
    ["handkerchief"],
    ["handle", "grip", "handgrip"],
    ["handsaw", "carpenter's saw"],
    ["hardback book", "hardcover book"],
    [
        "harmonium",
        "organ",
        "organ musical instrument",
        "reed organ",
        "reed organ musical instrument",
    ],
    ["hat"],
    ["hatbox"],
    ["veil"],
    ["headband"],
    ["headboard"],
    ["headlight", "headlamp"],
    ["headscarf"],
    ["headset"],
    ["headstall", "headstall for horses", "headpiece", "headpiece for horses"],
    ["heart"],
    ["heater", "warmer"],
    ["helicopter"],
    ["helmet"],
    ["heron"],
    ["highchair", "feeding chair"],
    ["hinge"],
    ["hippopotamus"],
    ["hockey stick"],
    ["hog", "pig"],
    ["home plate", "home plate baseball", "home base", "home base baseball"],
    ["honey"],
    ["fume hood", "exhaust hood"],
    ["hook"],
    ["hookah", "narghile", "nargileh", "sheesha", "shisha", "water pipe"],
    ["hornet"],
    ["horse"],
    ["hose", "hosepipe"],
    ["hot-air balloon"],
    ["hotplate"],
    ["hot sauce"],
    ["hourglass"],
    ["houseboat"],
    ["hummingbird"],
    ["hummus", "humus", "hommos", "hoummos", "humous"],
    ["polar bear"],
    ["icecream"],
    ["popsicle"],
    ["ice maker"],
    ["ice pack", "ice bag"],
    ["ice skate"],
    ["igniter", "ignitor", "lighter"],
    ["inhaler", "inhalator"],
    ["iPod"],
    ["iron", "iron for clothing", "smoothing iron", "smoothing iron for clothing"],
    ["ironing board"],
    ["jacket"],
    ["jam"],
    ["jar"],
    ["jean", "blue jean", "denim"],
    ["jeep", "landrover"],
    ["jelly bean", "jelly egg"],
    ["jersey", "T-shirt", "tee shirt"],
    ["jet plane", "jet-propelled plane"],
    ["jewel", "gem", "precious stone"],
    ["jewelry", "jewellery"],
    ["joystick"],
    ["jumpsuit"],
    ["kayak"],
    ["keg"],
    ["kennel", "doghouse"],
    ["kettle", "boiler"],
    ["key"],
    ["keycard"],
    ["kilt"],
    ["kimono"],
    ["kitchen sink"],
    ["kitchen table"],
    ["kite"],
    ["kitten", "kitty"],
    ["kiwi fruit"],
    ["knee pad"],
    ["knife"],
    ["knitting needle"],
    ["knob"],
    ["knocker", "knocker on a door", "doorknocker"],
    ["koala", "koala bear"],
    ["lab coat", "laboratory coat"],
    ["ladder"],
    ["ladle"],
    ["ladybug", "ladybeetle", "ladybird beetle"],
    ["lamb", "lamb animal"],
    ["lamb-chop", "lambchop"],
    ["lamp"],
    ["lamppost"],
    ["lampshade"],
    ["lantern"],
    ["lanyard", "laniard"],
    ["laptop computer", "notebook computer"],
    ["lasagna", "lasagne"],
    ["latch"],
    ["lawn mower"],
    ["leather"],
    ["legging", "legging clothing", "leging", "leging clothing", "leg covering"],
    ["Lego", "Lego set"],
    ["legume"],
    ["lemon"],
    ["lemonade"],
    ["lettuce"],
    ["license plate", "numberplate"],
    ["life buoy", "lifesaver", "life belt", "life ring"],
    ["life jacket", "life vest"],
    ["lightbulb"],
    ["lightning rod", "lightning conductor"],
    ["lime"],
    ["limousine"],
    ["lion"],
    ["lip balm"],
    ["liquor", "spirits", "hard liquor", "liqueur", "cordial"],
    ["lizard"],
    ["log"],
    ["lollipop"],
    ["speaker", "speaker stereo equipment"],
    ["loveseat"],
    ["machine gun"],
    ["magazine"],
    ["magnet"],
    ["mail slot"],
    ["mailbox", "mailbox at home", "letter box", "letter box at home"],
    ["mallard"],
    ["mallet"],
    ["mammoth"],
    ["manatee"],
    ["mandarin orange"],
    ["manger", "trough"],
    ["manhole"],
    ["map"],
    ["marker"],
    ["martini"],
    ["mascot"],
    ["mashed potato"],
    ["masher"],
    ["mask", "facemask"],
    ["mast"],
    ["mat", "mat gym equipment", "gym mat"],
    ["matchbox"],
    ["mattress"],
    ["measuring cup"],
    ["measuring stick", "ruler", "ruler measuring stick", "measuring rod"],
    ["meatball"],
    ["medicine"],
    ["melon"],
    ["microphone"],
    ["microscope"],
    ["microwave oven"],
    ["milestone", "milepost"],
    ["milk"],
    ["milk can"],
    ["milkshake"],
    ["minivan"],
    ["mint candy"],
    ["mirror"],
    ["mitten"],
    ["mixer", "mixer kitchen tool", "stand mixer"],
    ["money"],
    ["monitor", "monitor computer equipment"],
    ["monkey"],
    ["motor"],
    ["motor scooter", "scooter"],
    ["motor vehicle", "automotive vehicle"],
    ["motorcycle"],
    ["mound", "mound baseball", "pitcher's mound"],
    ["mouse", "mouse computer equipment", "computer mouse"],
    ["mousepad"],
    ["muffin"],
    ["mug"],
    ["mushroom"],
    ["music stool", "piano stool"],
    ["musical instrument", "instrument", "instrument musical"],
    ["nailfile"],
    ["napkin", "table napkin", "serviette"],
    ["neckerchief"],
    ["necklace"],
    ["necktie", "tie", "tie necktie"],
    ["needle"],
    ["nest"],
    ["newspaper", "paper", "paper newspaper"],
    ["newsstand"],
    ["nightshirt", "nightwear", "sleepwear", "nightclothes"],
    ["nosebag", "nosebag for animals", "feedbag"],
    ["noseband", "noseband for animals", "nosepiece", "nosepiece for animals"],
    ["notebook"],
    ["notepad"],
    ["nut"],
    ["nutcracker"],
    ["oar"],
    ["octopus", "octopus food"],
    ["octopus", "octopus animal"],
    ["oil lamp", "kerosene lamp", "kerosine lamp"],
    ["olive oil"],
    ["omelet", "omelette"],
    ["onion"],
    ["orange", "orange fruit"],
    ["orange juice"],
    ["ostrich"],
    ["ottoman", "pouf", "pouffe", "hassock"],
    ["oven"],
    ["overalls", "overalls clothing"],
    ["owl"],
    ["packet"],
    ["inkpad", "inking pad", "stamp pad"],
    ["pad"],
    ["paddle", "boat paddle"],
    ["padlock"],
    ["paintbrush"],
    ["painting"],
    ["pajamas", "pyjamas"],
    ["palette", "pallet"],
    ["pan", "pan for cooking", "cooking pan"],
    ["pan", "pan metal container"],
    ["pancake"],
    ["pantyhose"],
    ["papaya"],
    ["paper plate"],
    ["paper towel"],
    ["paperback book", "paper-back book", "softback book", "soft-cover book"],
    ["paperweight"],
    ["parachute"],
    ["parakeet", "parrakeet", "parroket", "paraquet", "paroquet", "parroquet"],
    ["parasail", "parasail sports"],
    ["parasol", "sunshade"],
    ["parchment"],
    ["parka", "anorak"],
    ["parking meter"],
    ["parrot"],
    [
        "passenger car",
        "passenger car part of a train",
        "coach",
        "coach part of a train",
    ],
    ["passenger ship"],
    ["passport"],
    ["pastry"],
    ["patty", "patty food"],
    ["pea", "pea food"],
    ["peach"],
    ["peanut butter"],
    ["pear"],
    ["peeler", "peeler tool for fruit and vegetables"],
    ["wooden leg", "pegleg"],
    ["pegboard"],
    ["pelican"],
    ["pen"],
    ["pencil"],
    ["pencil box", "pencil case"],
    ["pencil sharpener"],
    ["pendulum"],
    ["penguin"],
    ["pennant"],
    ["penny", "penny coin"],
    ["pepper", "peppercorn"],
    ["pepper mill", "pepper grinder"],
    ["perfume"],
    ["persimmon"],
    ["person", "baby", "child", "boy", "girl", "man", "woman", "human"],
    ["pet"],
    ["pew", "pew church bench", "church bench"],
    ["phonebook", "telephone book", "telephone directory"],
    [
        "phonograph record",
        "phonograph recording",
        "record",
        "record phonograph recording",
    ],
    ["piano"],
    ["pickle"],
    ["pickup truck"],
    ["pie"],
    ["pigeon"],
    ["piggy bank", "penny bank"],
    ["pillow"],
    ["pin", "pin non jewelry"],
    ["pineapple"],
    ["pinecone"],
    ["ping-pong ball"],
    ["pinwheel"],
    ["tobacco pipe"],
    ["pipe", "piping"],
    ["pistol", "handgun"],
    ["pita", "pita bread", "pocket bread"],
    ["pitcher", "pitcher vessel for liquid", "ewer"],
    ["pitchfork"],
    ["pizza"],
    ["place mat"],
    ["plate"],
    ["platter"],
    ["playpen"],
    ["pliers", "plyers"],
    ["plow", "plow farm equipment", "plough", "plough farm equipment"],
    ["plume"],
    ["pocket watch"],
    ["pocketknife"],
    ["poker", "poker fire stirring tool", "stove poker", "fire hook"],
    ["pole", "post"],
    ["polo shirt", "sport shirt"],
    ["poncho"],
    ["pony"],
    ["pool table", "billiard table", "snooker table"],
    ["pop", "pop soda", "soda", "soda pop", "tonic", "soft drink"],
    ["postbox", "postbox public", "mailbox", "mailbox public"],
    ["postcard", "postal card", "mailing-card"],
    ["poster", "placard"],
    ["pot"],
    ["flowerpot"],
    ["potato"],
    ["potholder"],
    ["pottery", "clayware"],
    ["pouch"],
    ["power shovel", "excavator", "digger"],
    ["prawn", "shrimp"],
    ["pretzel"],
    ["printer", "printing machine"],
    ["projectile", "projectile weapon", "missile"],
    ["projector"],
    ["propeller", "propellor"],
    ["prune"],
    ["pudding"],
    ["puffer", "puffer fish", "pufferfish", "blowfish", "globefish"],
    ["puffin"],
    ["pug-dog"],
    ["pumpkin"],
    ["puncher"],
    ["puppet", "marionette"],
    ["puppy"],
    ["quesadilla"],
    ["quiche"],
    ["quilt", "comforter"],
    ["rabbit"],
    ["race car", "racing car"],
    ["racket", "racquet"],
    ["radar"],
    ["radiator"],
    ["radio receiver", "radio set", "radio", "tuner", "tuner radio"],
    ["radish", "daikon"],
    ["raft"],
    ["rag doll"],
    ["raincoat", "waterproof jacket"],
    ["ram", "ram animal"],
    ["raspberry"],
    ["rat"],
    ["razorblade"],
    ["reamer", "reamer juicer", "juicer", "juice reamer"],
    ["rearview mirror"],
    ["receipt"],
    ["recliner", "reclining chair", "lounger", "lounger chair"],
    ["record player", "phonograph", "phonograph record player", "turntable"],
    ["reflector"],
    ["remote control"],
    ["rhinoceros"],
    ["rib", "rib food"],
    ["rifle"],
    ["ring"],
    ["river boat"],
    ["road map"],
    ["robe"],
    ["rocking chair"],
    ["rodent"],
    ["roller skate"],
    ["Rollerblade"],
    ["rolling pin"],
    ["root beer"],
    ["router", "router computer equipment"],
    ["rubber band", "elastic band"],
    ["runner", "runner carpet"],
    ["plastic bag", "paper bag"],
    ["saddle", "saddle on an animal"],
    ["saddle blanket", "saddlecloth", "horse blanket"],
    ["saddlebag"],
    ["safety pin"],
    ["sail"],
    ["salad"],
    ["salad plate", "salad bowl"],
    ["salami"],
    ["salmon", "salmon fish"],
    ["salmon", "salmon food"],
    ["salsa"],
    ["saltshaker"],
    ["sandal", "sandal type of shoe"],
    ["sandwich"],
    ["satchel"],
    ["saucepan"],
    ["saucer"],
    ["sausage"],
    ["sawhorse", "sawbuck"],
    ["saxophone"],
    ["scale", "scale measuring instrument"],
    ["scarecrow", "strawman"],
    ["scarf"],
    ["school bus"],
    ["scissors"],
    ["scoreboard"],
    ["scraper"],
    ["screwdriver"],
    ["scrubbing brush"],
    ["sculpture"],
    ["seabird", "seafowl"],
    ["seahorse"],
    ["seaplane", "hydroplane"],
    ["seashell"],
    ["sewing machine"],
    ["shaker"],
    ["shampoo"],
    ["shark"],
    ["sharpener"],
    ["Sharpie"],
    ["shaver", "shaver electric", "electric shaver", "electric razor"],
    ["shaving cream", "shaving soap"],
    ["shawl"],
    ["shears"],
    ["sheep"],
    ["shepherd dog", "sheepdog"],
    ["sherbert", "sherbet"],
    ["shield"],
    ["shirt"],
    ["shoe", "sneaker", "sneaker type of shoe", "tennis shoe"],
    ["shopping bag"],
    ["shopping cart"],
    ["short pants", "shorts", "shorts clothing", "trunks", "trunks clothing"],
    ["shot glass"],
    ["shoulder bag"],
    ["shovel"],
    ["shower head"],
    ["shower cap"],
    ["shower curtain"],
    ["shredder", "shredder for paper"],
    ["signboard"],
    ["silo"],
    ["sink"],
    ["skateboard"],
    ["skewer"],
    ["ski"],
    ["ski boot"],
    ["ski parka", "ski jacket"],
    ["ski pole"],
    ["skirt"],
    ["skullcap"],
    ["sled", "sledge", "sleigh"],
    ["sleeping bag"],
    ["sling", "sling bandage", "triangular bandage"],
    ["slipper", "slipper footwear", "carpet slipper", "carpet slipper footwear"],
    ["smoothie"],
    ["snake", "serpent"],
    ["snowboard"],
    ["snowman"],
    ["snowmobile"],
    ["soap"],
    ["soccer ball"],
    ["sock"],
    ["sofa", "couch", "lounge"],
    ["softball"],
    ["solar array", "solar battery", "solar panel"],
    ["sombrero"],
    ["soup"],
    ["soup bowl"],
    ["soupspoon"],
    ["sour cream", "soured cream"],
    ["soya milk", "soybean milk", "soymilk"],
    ["space shuttle"],
    ["sparkler", "sparkler fireworks"],
    ["spatula"],
    ["spear", "lance"],
    ["spectacles", "specs", "eyeglasses", "glasses"],
    ["spice rack"],
    ["spider"],
    ["crawfish", "crayfish"],
    ["sponge"],
    ["spoon"],
    ["sportswear", "athletic wear", "activewear"],
    ["spotlight"],
    ["squid", "squid food", "calamari", "calamary"],
    ["squirrel"],
    ["stagecoach"],
    ["stapler", "stapler stapling machine"],
    ["starfish", "sea star"],
    ["statue", "statue sculpture"],
    ["steak", "steak food"],
    ["steak knife"],
    ["steering wheel"],
    ["stepladder"],
    ["step stool"],
    ["stereo", "stereo sound system"],
    ["stew"],
    ["stirrer"],
    ["stirrup"],
    ["stool"],
    ["stop sign"],
    ["brake light"],
    [
        "stove",
        "kitchen stove",
        "range",
        "range kitchen appliance",
        "kitchen range",
        "cooking stove",
    ],
    ["strainer"],
    ["strap"],
    ["straw", "straw for drinking", "drinking straw"],
    ["strawberry"],
    ["street sign"],
    ["streetlight", "street lamp"],
    ["string cheese"],
    ["stylus"],
    ["subwoofer"],
    ["sugar bowl"],
    ["sugarcane", "sugarcane plant"],
    ["suit", "suit clothing"],
    ["sunflower"],
    ["sunglasses"],
    ["sunhat"],
    ["surfboard"],
    ["sushi"],
    ["mop"],
    ["sweat pants"],
    ["sweatband"],
    ["sweater"],
    ["sweatshirt"],
    ["sweet potato"],
    [
        "swimsuit",
        "swimwear",
        "bathing suit",
        "swimming costume",
        "bathing costume",
        "swimming trunks",
        "bathing trunks",
    ],
    ["sword"],
    ["syringe"],
    ["Tabasco sauce"],
    ["table-tennis table", "ping-pong table"],
    ["table"],
    ["table lamp"],
    ["tablecloth"],
    ["tachometer"],
    ["taco"],
    ["tag"],
    ["taillight", "rear light"],
    ["tambourine"],
    ["army tank", "armored combat vehicle", "armoured combat vehicle"],
    ["tank", "tank storage vessel", "storage tank"],
    ["tank top", "tank top clothing"],
    ["tape", "tape sticky cloth or paper"],
    ["tape measure", "measuring tape"],
    ["tapestry"],
    ["tarp"],
    ["tartan", "plaid"],
    ["tassel"],
    ["tea bag"],
    ["teacup"],
    ["teakettle"],
    ["teapot"],
    ["teddy bear"],
    ["telephone", "phone", "telephone set"],
    ["telephone booth", "phone booth", "call box", "telephone box", "telephone kiosk"],
    ["telephone pole", "telegraph pole", "telegraph post"],
    ["telephoto lens", "zoom lens"],
    ["television camera", "tv camera"],
    ["television set", "tv", "tv set"],
    ["tennis ball"],
    ["tennis racket"],
    ["tequila"],
    ["thermometer"],
    ["thermos bottle"],
    ["thermostat"],
    ["thimble"],
    ["thread", "yarn"],
    ["thumbtack", "drawing pin", "pushpin"],
    ["tiara"],
    ["tiger"],
    ["tights", "tights clothing", "leotards"],
    ["timer", "stopwatch"],
    ["tinfoil"],
    ["tinsel"],
    ["tissue paper"],
    ["toast", "toast food"],
    ["toaster"],
    ["toaster oven"],
    ["toilet"],
    ["toilet tissue", "toilet paper", "bathroom tissue"],
    ["tomato"],
    ["tongs"],
    ["toolbox"],
    ["toothbrush"],
    ["toothpaste"],
    ["toothpick"],
    ["cover"],
    ["tortilla"],
    ["tow truck"],
    ["towel"],
    ["towel rack", "towel rail", "towel bar"],
    ["toy"],
    ["tractor", "tractor farm equipment"],
    ["traffic light"],
    ["dirt bike"],
    [
        "trailer truck",
        "tractor trailer",
        "trucking rig",
        "articulated lorry",
        "semi truck",
    ],
    ["train", "train railroad vehicle", "railroad train"],
    ["trampoline"],
    ["tray"],
    ["trench coat"],
    ["triangle", "triangle musical instrument"],
    ["tricycle"],
    ["tripod"],
    ["trousers", "pants", "pants clothing"],
    ["truck"],
    ["truffle", "truffle chocolate", "chocolate truffle"],
    ["trunk"],
    ["vat"],
    ["turban"],
    ["turkey", "turkey food"],
    ["turnip"],
    ["turtle"],
    ["turtleneck", "turtleneck clothing", "polo-neck"],
    ["typewriter"],
    ["umbrella"],
    ["underwear", "underclothes", "underclothing", "underpants"],
    ["unicycle"],
    ["urinal"],
    ["urn"],
    ["vacuum cleaner"],
    ["vase"],
    ["vending machine"],
    ["vent", "blowhole", "air vent"],
    ["vest", "waistcoat"],
    ["videotape"],
    ["vinegar"],
    ["violin", "fiddle"],
    ["vodka"],
    ["volleyball"],
    ["vulture"],
    ["waffle"],
    ["waffle iron"],
    ["wagon"],
    ["wagon wheel"],
    ["walking stick"],
    ["wall clock"],
    [
        "wall socket",
        "wall plug",
        "electric outlet",
        "electrical outlet",
        "outlet",
        "electric receptacle",
    ],
    ["wallet", "billfold"],
    ["walrus"],
    ["wardrobe"],
    ["washbasin", "basin", "basin for washing", "washbowl", "washstand", "handbasin"],
    ["automatic washer", "washing machine"],
    ["watch", "wristwatch"],
    ["water bottle"],
    ["water cooler"],
    ["water faucet", "water tap", "tap", "tap water faucet"],
    ["water heater", "hot-water heater"],
    ["water jug"],
    ["water gun", "squirt gun"],
    ["water scooter", "sea scooter", "jet ski"],
    ["water ski"],
    ["water tower"],
    ["watering can"],
    ["watermelon"],
    ["weathervane", "vane", "vane weathervane", "wind vane"],
    ["webcam"],
    ["wedding cake", "bridecake"],
    ["wedding ring", "wedding band"],
    ["wet suit"],
    ["wheel"],
    ["wheelchair"],
    ["whipped cream"],
    ["whistle"],
    ["wig"],
    ["wind chime"],
    ["windmill"],
    ["window box", "window box for plants"],
    ["windshield wiper", "windscreen wiper", "wiper", "wiper for windshield or screen"],
    ["windsock", "air sock", "air-sleeve", "wind sleeve", "wind cone"],
    ["wine bottle"],
    ["wine bucket", "wine cooler"],
    ["wineglass"],
    ["blinder", "blinder for horses"],
    ["wok"],
    ["wolf"],
    ["wooden spoon"],
    ["wreath"],
    ["wrench", "spanner"],
    ["wristband"],
    ["wristlet", "wrist band"],
    ["yacht"],
    ["yogurt", "yoghurt", "yoghourt"],
    ["yoke", "yoke animal equipment"],
    ["zebra"],
    ["zucchini", "courgette"],
]
