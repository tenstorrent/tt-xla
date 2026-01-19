# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Consolidated imports and dependencies
import ast
import fnmatch
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import urllib.parse
import zipfile
from collections import namedtuple
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from hashlib import sha256
from os.path import basename, isdir, isfile, join
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile, is_zipfile

import requests
import torch
from filelock import FileLock
from huggingface_hub import HfApi, HfFolder, snapshot_download
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    hf_raise_for_status,
)
from requests.exceptions import HTTPError
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.utils import http_user_agent, is_remote_url

__version__ = "1.2.0"
# Define regex patterns locally to avoid circular imports
NEWLINE_WHITESPACE_RE = re.compile(r"\n\s*\n")
SPACE_RE = re.compile(r"\s")
PUNCTUATION = re.compile(
    r"""["''\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?''"";/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]"""
)


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend(
                [i + offset for i in range(token_len)]
                + [-1] * (max_token_len - token_len)
            )
            seq_masks.extend(
                [1.0 / token_len] * token_len + [0.0] * (max_token_len - token_len)
            )
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


# Dataset classes
instance_fields = [
    "paragraph_index",
    "wordpieces",
    "wordpiece_labels",
    "wordpiece_ends",
    "piece_idxs",
    "attention_masks",
    "token_type_idxs",
    "wordpiece_num",
]

batch_fields = [
    "paragraph_index",
    "wordpieces",
    "wordpiece_labels",
    "wordpiece_ends",
    "piece_idxs",
    "attention_masks",
    "token_type_idxs",
    "wordpiece_num",
]

Instance = namedtuple("Instance", field_names=instance_fields)
Batch = namedtuple("Batch", field_names=batch_fields)


class TokenizeDatasetLive(Dataset):
    def __init__(self, config, raw_text, max_input_length=512):
        self.config = config
        self.max_input_length = max_input_length
        self.treebank_name = config.treebank_name
        self.raw_text = raw_text
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        self.data = charlevel_format_to_wordpiece_format(
            wordpiece_splitter=self.config.wordpiece_splitter,
            max_input_length=self.max_input_length,
            plaintext=self.raw_text,
            treebank_name=self.config.treebank_name,
        )

    def numberize(self, wordpiece_splitter):
        data = []
        for inst in self.data:
            wordpieces = inst["wordpieces"]
            wordpiece_labels = inst["wordpiece_labels"]
            wordpiece_ends = inst["wordpiece_ends"]
            paragraph_index = inst["paragraph_index"]
            piece_idxs = wordpiece_splitter.encode(
                wordpieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True,
            )
            assert len(piece_idxs) <= self.max_input_length

            pad_num = self.max_input_length - len(piece_idxs)
            attn_masks = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            token_type_idxs = [
                -100 if piece_id >= len(wordpieces) else wordpiece_labels[piece_id]
                for piece_id in range(len(piece_idxs) - 2)
            ]

            instance = Instance(
                paragraph_index=paragraph_index,
                wordpieces=wordpieces,
                wordpiece_labels=wordpiece_labels,
                wordpiece_ends=wordpiece_ends,
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                token_type_idxs=token_type_idxs,
                wordpiece_num=len(wordpieces),
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_paragraph_index = []
        batch_wordpieces = []
        batch_wordpiece_labels = []
        batch_wordpiece_ends = []
        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_type_idxs = []
        batch_wordpiece_num = []

        for inst in batch:
            batch_paragraph_index.append(inst.paragraph_index)
            batch_wordpieces.append(inst.wordpieces)
            batch_wordpiece_labels.append(inst.wordpiece_labels)
            batch_wordpiece_ends.append(inst.wordpiece_ends)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_masks)
            batch_token_type_idxs.append(inst.token_type_idxs)
            batch_wordpiece_num.append(inst.wordpiece_num)

        batch_piece_idxs = torch.tensor(
            batch_piece_idxs, dtype=torch.long, device=self.config.device
        )
        batch_attention_masks = torch.tensor(
            batch_attention_masks, dtype=torch.long, device=self.config.device
        )
        batch_token_type_idxs = torch.tensor(
            batch_token_type_idxs, dtype=torch.long, device=self.config.device
        )
        batch_wordpiece_num = torch.tensor(
            batch_wordpiece_num, dtype=torch.long, device=self.config.device
        )

        return Batch(
            paragraph_index=batch_paragraph_index,
            wordpieces=batch_wordpieces,
            wordpiece_labels=batch_wordpiece_labels,
            wordpiece_ends=batch_wordpiece_ends,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_type_idxs=batch_token_type_idxs,
            wordpiece_num=batch_wordpiece_num,
        )


# Utility functions
def is_string(input):
    if type(input) == str and len(input.strip()) > 0:
        return True
    return False


def normalize_input(input):
    tmp = input.lstrip()
    lstrip_offset = len(input) - len(input.lstrip())
    return tmp, lstrip_offset


def get_start_char_idx(substring, text):
    start_char_idx = text.index(substring)
    text = text[start_char_idx + len(substring) :]
    return text, start_char_idx


def split_to_substrings(sent_text):
    tokens_by_space = sent_text.split()
    substrings = []
    for token in tokens_by_space:
        if len(PUNCTUATION.findall(token)) > 0:
            tmp = ""
            for char in token:
                if PUNCTUATION.match(char):
                    if tmp != "":
                        substrings.append(tmp)
                        tmp = ""
                    substrings.append(char)
                else:
                    tmp += char
            if tmp != "":
                substrings.append(tmp)
        else:
            substrings.append(token)
    return substrings


def get_startchar(word, text):
    start_char_idx = 0
    for k in range(len(text)):
        if len(text[k].strip()) > 0:
            start_char_idx = k
            break
    text = text[start_char_idx + len(word) :]
    return text, start_char_idx


def get_character_locations(string_units, text):
    tmp_text = deepcopy(text)
    offset = 0
    end_positions = []
    for str_unit in string_units:
        tmp_text, start_position = get_startchar(str_unit, tmp_text)
        start_position += offset
        end_position = start_position + len(str_unit) - 1
        end_positions.append(end_position)
        offset = start_position + len(str_unit)
    return end_positions


def get_mapping_wp_character_to_or_character(
    wordpiece_splitter, wp_single_string, or_single_string
):
    wp_char_to_or_char = {}
    converted_text = ""
    for char_id, char in enumerate(or_single_string):
        converted_chars = "".join(
            [
                c if not c.startswith("▁") else c[1:]
                for c in wordpiece_splitter.tokenize(char)
                if c != "▁"
            ]
        )
        for converted_c in converted_chars:
            c_id = len(converted_text)
            wp_char_to_or_char[c_id] = char_id
            converted_text += converted_c
    return wp_char_to_or_char


def wordpiece_tokenize_from_raw_text(
    wordpiece_splitter,
    sent_text,
    sent_labels,
    sent_position_in_paragraph,
    treebank_name,
):
    if "Chinese" in treebank_name or "Japanese" in treebank_name:
        pseudo_tokens = [c for c in sent_text]  # characters as pseudo tokens
    else:
        if treebank_name == "UD_Urdu-UDTB":
            sent_text = sent_text.replace("۔", ".")
        elif treebank_name == "UD_Uyghur-UDT":
            sent_text = sent_text.replace("-", "،")
        pseudo_tokens = split_to_substrings(sent_text)
    end_pids = set()
    group_pieces = [wordpiece_splitter.tokenize(t) for t in pseudo_tokens]
    flat_wordpieces = []
    for group in group_pieces:
        if len(group) > 0:
            for p in group:
                if p != "▁":
                    pid = len(flat_wordpieces)
                    flat_wordpieces.append((p, pid))
            end_pids.add(len(flat_wordpieces) - 1)

    single_original_string = "".join([c.strip() for c in sent_text])
    original_characters = [c for c in single_original_string]
    character_locations = get_character_locations(original_characters, sent_text)
    single_wordpiece_string = "".join(
        [p if not p.startswith("▁") else p.lstrip("▁") for p, pid in flat_wordpieces]
    )
    wp_character_2_or_character = get_mapping_wp_character_to_or_character(
        wordpiece_splitter, single_wordpiece_string, single_original_string
    )

    flat_wordpiece_labels = []
    flat_wordpiece_ends = []
    offset = 0
    for wordpiece, _ in flat_wordpieces:
        if wordpiece.startswith("▁"):
            str_form = wordpiece[1:]
        else:
            str_form = wordpiece
        end_char = offset + len(str_form) - 1
        ori_char = wp_character_2_or_character[end_char]
        location_in_sentence = character_locations[ori_char]
        wp_label = int(sent_labels[location_in_sentence])
        wp_end = sent_position_in_paragraph + location_in_sentence
        flat_wordpiece_labels.append(wp_label)
        flat_wordpiece_ends.append(wp_end)
        offset = end_char + 1

    return flat_wordpieces, flat_wordpiece_labels, flat_wordpiece_ends, end_pids


def split_to_sentences(paragraph_text, charlabels):
    sent_text = ""
    sent_labels = ""
    sentences = []
    start = 0

    for k in range(len(charlabels)):
        sent_text += paragraph_text[k]
        sent_labels += charlabels[k]

        if charlabels[k] == "2" or charlabels[k] == "4":
            end = k
            sentences.append((deepcopy(sent_text), deepcopy(sent_labels), start, end))
            start = end + 1
            sent_text = ""
            sent_labels = ""

    if len(sentences) > 0:
        if not (len(sent_text) == 0 and len(sent_labels) == 0):
            sentences.append(
                (
                    deepcopy(sent_text),
                    deepcopy(sent_labels),
                    start,
                    len(paragraph_text) - 1,
                )
            )
    else:
        sentences = [(paragraph_text, charlabels, 0, len(paragraph_text) - 1)]
    return sentences


def split_to_subsequences(
    wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids, max_input_length
):
    subsequences = []
    subseq = [[], [], []]

    for wp_wpid, wl, we in zip(wordpieces, wordpiece_labels, wordpiece_ends):
        wp, wpid = wp_wpid
        subseq[0].append((wp, wpid))
        subseq[1].append(wl)
        subseq[2].append(we)
        if wpid in end_piece_ids and len(subseq[0]) >= max_input_length - 10:
            subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))
            subseq = [[], [], []]

    if len(subseq[0]) > 0:
        subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))
    return subsequences


def charlevel_format_to_wordpiece_format(
    wordpiece_splitter,
    max_input_length,
    plaintext,
    treebank_name,
    char_labels_output_fpath=None,
):
    if char_labels_output_fpath is not None:
        with open(char_labels_output_fpath) as f:
            corpus_labels = "".join(f.readlines()).rstrip()
    else:
        corpus_labels = "\n\n".join(
            ["0" * len(pt.rstrip()) for pt in NEWLINE_WHITESPACE_RE.split(plaintext)]
        )

    data = [
        {"text": pt.rstrip(), "charlabels": pc}
        for pt, pc in zip(
            NEWLINE_WHITESPACE_RE.split(plaintext),
            NEWLINE_WHITESPACE_RE.split(corpus_labels),
        )
        if len(pt.rstrip()) > 0
    ]

    wordpiece_examples = []
    kept_tokens = 0
    total_tokens = 0
    for paragraph_index, paragraph in enumerate(data):
        paragraph_text = paragraph["text"]
        paragraph_labels = paragraph["charlabels"]
        sentences = split_to_sentences(paragraph_text, paragraph_labels)
        tmp_examples = []
        for sent in sentences:
            sent_text, sent_labels, sent_start, sent_end = sent
            (
                wordpieces,
                wordpiece_labels,
                wordpiece_ends,
                end_piece_ids,
            ) = wordpiece_tokenize_from_raw_text(
                wordpiece_splitter, sent_text, sent_labels, sent_start, treebank_name
            )
            kept_tokens += len([x for x in wordpiece_labels if x != 0])
            total_tokens += len([x for x in sent_labels if x != "0"])
            if len(wordpieces) <= max_input_length - 2:
                tmp_examples.append(
                    (wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids)
                )
            else:
                subsequences = split_to_subsequences(
                    wordpieces,
                    wordpiece_labels,
                    wordpiece_ends,
                    end_piece_ids,
                    max_input_length,
                )
                for subseq in subsequences:
                    tmp_examples.append(subseq)
        new_example = [[], [], []]
        for example in tmp_examples:
            if len(new_example[0]) + len(example[0]) > max_input_length - 2:
                num_extra_wordpieces = min(
                    max_input_length - 2 - len(new_example[0]), len(example[0])
                )
                end_piece_ids = example[-1]
                takeout_position = 0
                for tmp_id in range(num_extra_wordpieces):
                    wp, wpid = example[0][tmp_id]
                    if wpid in end_piece_ids:
                        takeout_position = tmp_id + 1
                num_extra_wordpieces = takeout_position
                new_example[0] += deepcopy(example[0][:num_extra_wordpieces])
                new_example[1] += deepcopy(example[1][:num_extra_wordpieces])
                new_example[2] += deepcopy(example[2][:num_extra_wordpieces])
                wordpiece_examples.append(
                    (
                        [wp for wp, wpid in new_example[0]],
                        new_example[1],
                        new_example[2],
                        paragraph_index,
                    )
                )
                new_example = [[], [], []]

            new_example[0] += deepcopy(example[0])
            new_example[1] += deepcopy(example[1])
            new_example[2] += deepcopy(example[2])
        if len(new_example[0]) > 0:
            wordpiece_examples.append(
                (
                    [wp for wp, wpid in new_example[0]],
                    new_example[1],
                    new_example[2],
                    paragraph_index,
                )
            )

    final_examples = []
    for wp_example in wordpiece_examples:
        wordpieces, wordpiece_labels, wordpiece_ends, paragraph_index = wp_example
        final_examples.append(
            {
                "wordpieces": wordpieces,
                "wordpiece_labels": wordpiece_labels,
                "wordpiece_ends": wordpiece_ends,
                "paragraph_index": paragraph_index,
            }
        )

    return final_examples


def normalize_token(treebank_name, token, ud_eval=True):
    token = SPACE_RE.sub(" ", token.lstrip())
    if ud_eval:
        if (
            "chinese" in treebank_name.lower()
            or "korean" in treebank_name.lower()
            or "japanese" in treebank_name.lower()
        ):
            token = token.replace(" ", "")
    return token


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def unzip(dir, filename):
    with zipfile.ZipFile(os.path.join(dir, filename)) as f:
        f.extractall(dir)
    os.remove(os.path.join(dir, filename))


def download(language, saved_model_version, embedding_name):
    """
    Args:
        language: Language code for the model
        saved_model_version: Version of the saved model
        embedding_name: Name of the embedding model

    Returns:
        str: Path to the cache directory where files were downloaded
    """
    from pathlib import Path
    import shutil

    # Construct the URL
    url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/{saved_model_version}/{embedding_name}/{language}.zip"

    # Use get_file to handle download and cache directory logic automatically
    zip_file_path = get_file(url)

    # Extract cache directory from the downloaded file path
    zip_path = Path(zip_file_path)
    cache_base = zip_path.parent.parent  # Go up from url_cache to main cache dir

    # Create trankit directory structure
    trankit_cache = cache_base / "trankit"
    lang_dir = trankit_cache / embedding_name / language
    downloaded_marker = lang_dir / f"{language}.downloaded"

    # Only process if not already done
    if not downloaded_marker.exists():
        print(f"Setting up {language} models...")

        # Ensure directory exists
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Copy zip file to trankit location
        target_zip = lang_dir / f"{language}.zip"
        shutil.copy2(zip_file_path, target_zip)

        # Unzip
        unzip(str(lang_dir), f"{language}.zip")

        # Mark as completed
        with open(downloaded_marker, "w") as f:
            f.write("")

        print(f"Successfully set up models for {language}")

    return str(trankit_cache)


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    path_is_url = path.startswith(("http://", "https://"))

    if path_is_url:
        # Create a hash from the URL to ensure uniqueness and prevent collisions
        url_hash = hashlib.md5(path.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(path).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        rel_path = Path("url_cache")
        cache_dir_fallback = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        rel_path = Path("models/tt-ci-models-private") / rel_dir
        cache_dir_fallback = Path.home() / ".cache/lfcache" / rel_dir

    # Determine the base cache directory based on environment variables
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / rel_path
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_path
    else:
        cache_dir = cache_dir_fallback

    file_path = cache_dir / file_name

    # Support case where shared cache is read only and file not found. Can read files from it, but
    # fall back to home dir cache for storing downloaded files. Common w/ CI cache shared w/ users.
    cache_dir_rdonly = not os.access(cache_dir, os.W_OK)
    if not file_path.exists() and cache_dir_rdonly and cache_dir != cache_dir_fallback:
        print(
            f"Warning: {cache_dir} is read-only, using {cache_dir_fallback} for {path}"
        )
        cache_dir = cache_dir_fallback
        file_path = cache_dir / file_name

    cache_dir.mkdir(parents=True, exist_ok=True)

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if path_is_url:
            try:
                print(f"Downloading file from URL {path} to {file_path}")
                response = requests.get(path, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {path}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            print(f"Downloading file from path {path} to {cache_dir}/{file_name}")
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


class AdapterSetup(ContextManager):
    """
    Represents an adapter setup of a model including active adapters and active heads. This class is intended to be
    used as a context manager using the ``with`` statement. The setup defined by the ``AdapterSetup`` context will
    override static adapter setups defined in a model (i.e. setups specified via ``active_adapters``).

    Example::

        with AdapterSetup(Stack("a", "b")):
            # will use the adapter stack "a" and "b" outputs = model(**inputs)

    Note that the context manager is thread-local, i.e. it can be used with different setups in a multi-threaded
    environment.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    def __init__(self, adapter_setup, head_setup=None, ignore_empty: bool = False):
        # Import here to avoid circular import
        from .adapter_utils import parse_composition, parse_heads_from_composition

        self.adapter_setup = parse_composition(adapter_setup)
        if head_setup:
            self.head_setup = head_setup
        else:
            self.head_setup = parse_heads_from_composition(self.adapter_setup)
        self._empty = (
            ignore_empty and self.adapter_setup is None and self.head_setup is None
        )

    def __enter__(self):
        if not self._empty:
            AdapterSetup.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        if not self._empty:
            AdapterSetup.get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        contexts = cls.get_contexts()
        if not contexts:
            return None
        return contexts[-1]

    @classmethod
    def get_context_adapter_setup(cls):
        context = cls.get_context()
        if context:
            return context.adapter_setup
        return None

    @classmethod
    def get_context_head_setup(cls):
        context = cls.get_context()
        if context:
            return context.head_setup
        return None


class ForwardContext(ContextManager):
    """
    Holds context information during a forward pass through a model. This class should be used via the
    ``ForwardContext.wrap()`` method.

    Note that the context is thread-local.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    context_args = {
        "output_adapter_gating_scores",
        "output_adapter_fusion_attentions",
        "adapter_input_parallelized",
        "task_ids",
    }
    context_attributes = {
        "adapter_gating_scores",
        "adapter_fusion_attentions",
    }
    # Additional used attributes not exposed to the user
    # - prompt_tokens_length: length of the prompt tokens

    def __init__(self, model, *args, **kwargs):
        # If the model has a method ``forward_context()``, use it to create the context.
        for arg_name in self.context_args:
            setattr(self, arg_name, kwargs.pop(arg_name, None))
        if hasattr(model, "forward_context"):
            model.forward_context(self, *args, **kwargs)

    def __enter__(self):
        ForwardContext.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        ForwardContext.get_contexts().pop()

    def _call_forward(self, model, f, *args, **kwargs):
        """
        Calls the forward function of the model with the given arguments and keyword arguments.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in self.context_args}
        results = f(model, *args, **kwargs)

        # append output attributes
        if isinstance(results, tuple):
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results = results + (dict(getattr(self, attr)),)
        else:
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results[attr] = dict(getattr(self, attr))

        return results

    @classmethod
    def add_context_args_in_signature(cls, f):
        old_signature = inspect.signature(f)
        params = list(old_signature.parameters.values())
        # search if a VAR_POSITIONAL or VAR_KEYWORD is present
        # if yes insert step parameter before it, else insert it in last position
        param_types = [param.kind for param in params]
        i = min(
            [
                (
                    param_types.index(param_type)
                    if param_type in param_types
                    else float("inf")
                )
                for param_type in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            + [len(params)]
        )
        for name in cls.context_args:
            new_param = inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            )
            if new_param not in params:
                params.insert(i, new_param)
            # we can now build the signature for the wrapper function
        new_signature = old_signature.replace(parameters=params)
        return new_signature

    @classmethod
    def wrap_base(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a base model class.
        Unlike ``wrap()``, this method does not create a new context if the is an existing one.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if (
                self.adapters_config is not None
                and ForwardContext.get_context() is None
            ):
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        contexts = cls.get_contexts()
        if not contexts:
            return None
        return contexts[-1]


CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
SAFE_WEIGHTS_NAME = "adapter.safetensors"
HEAD_CONFIG_NAME = "head_config.json"
HEAD_WEIGHTS_NAME = "pytorch_model_head.bin"
SAFE_HEAD_WEIGHTS_NAME = "model_head.safetensors"
ADAPTERFUSION_CONFIG_NAME = "adapter_fusion_config.json"
ADAPTERFUSION_WEIGHTS_NAME = "pytorch_model_adapter_fusion.bin"
SAFE_ADAPTERFUSION_WEIGHTS_NAME = "model_adapter_fusion.safetensors"
EMBEDDING_FILE = "embedding.pt"
TOKENIZER_PATH = "tokenizer"
SETUP_CONFIG_NAME = "adapter_setup.json"
INTERFACE_CONFIG_NAME = "adapter_interface.json"

ADAPTER_HUB_URL = "https://raw.githubusercontent.com/Adapter-Hub/Hub/master/dist/v2/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "index/{}.json"
ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + "architectures.json"
ADAPTER_HUB_ALL_FILE = ADAPTER_HUB_URL + "all.json"
ADAPTER_HUB_ADAPTER_ENTRY_JSON = ADAPTER_HUB_URL + "adapters/{}/{}.json"

# the download cache
torch_cache_home = os.getenv(
    "TORCH_HOME",
    os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "torch"),
)
ADAPTER_CACHE = join(torch_cache_home, "adapters")

# these keys are ignored when calculating the config hash
ADAPTER_CONFIG_HASH_IGNORE = []

# old: new
ACTIVATION_RENAME = {
    "gelu": "gelu_new",
    "gelu_orig": "gelu",
}
# HACK: To keep config hashs consistent with v2, remove default values of keys introduced in v3 from hash computation
ADAPTER_CONFIG_HASH_IGNORE_DEFAULT = {
    "phm_layer": True,
    "phm_dim": 4,
    "factorized_phm_W": True,
    "shared_W_phm": False,
    "shared_phm_rule": True,
    "factorized_phm_rule": False,
    "phm_c_init": "normal",
    "phm_init_range": 0.0001,
    "learn_phm": True,
    "hypercomplex_nonlinearity": "glorot-uniform",
    "phm_rank": 1,
    "phm_bias": True,
    "init_weights": "bert",
    "scaling": 1.0,
}
ADAPTER_CONFIG_STRING_PATTERN = re.compile(
    r"^(?P<name>[^\[\]\|\n]+)(?:\[(?P<kvs>.*)\])?$"
)


class AdapterType(str, Enum):
    """Models all currently available model adapter types."""

    text_task = "text_task"
    text_lang = "text_lang"

    @classmethod
    def has(cls, value):
        return value in cls.__members__.values()

    def __repr__(self):
        return self.value


@dataclass
class AdapterInfo:
    """
    Holds information about an adapter publicly available on the Hub. Returned by
    :func:`list_adapters()`.

    Args:
        source (str): The source repository of this adapter. Always 'hf' for adapters available on HF Model Hub.
        adapter_id (str): The unique identifier of this adapter.
        model_name (str, optional): The identifier of the model this adapter was trained for.
        task (str, optional): The task this adapter was trained for.
        subtask (str, optional): The subtask or dataset this adapter was trained on.
        username (str, optional): The username of author(s) of this adapter.
        adapter_config (dict, optional): The configuration dictionary of this adapter.
    """

    source: str
    adapter_id: str
    model_name: Optional[str] = None
    task: Optional[str] = None
    subtask: Optional[str] = None
    username: Optional[str] = None
    adapter_config: Optional[dict] = None
    sha1_checksum: Optional[str] = None


def _minimize_dict(d):
    if isinstance(d, Mapping):
        return {k: _minimize_dict(v) for (k, v) in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config, length=16, ignore_params=[]):
    """
    Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    minimized_config = _minimize_dict(
        {
            k: v
            for (k, v) in config.items()
            if k not in ADAPTER_CONFIG_HASH_IGNORE + ignore_params
        }
    )
    # ensure hash is kept consistent to previous versions
    for name, default in ADAPTER_CONFIG_HASH_IGNORE_DEFAULT.items():
        if minimized_config.get(name, None) == default:
            del minimized_config[name]
    dict_str = json.dumps(minimized_config, sort_keys=True)
    h = hashlib.sha1()
    h.update(dict_str.encode(encoding="utf-8"))
    return h.hexdigest()[:length]


def inherit_doc(cls):
    for name, func in vars(cls).items():
        if isinstance(func, Callable) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, "__doc__", None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls


def multigetattr(o: object, name: str, default=None) -> Optional[object]:
    if not name:
        return default
    for n in name.split("."):
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return default
    return o


def multihasattr(o: object, name: str) -> bool:
    if not name:
        return False
    parts = name.split(".")
    for n in parts:
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return False
    return True


def multisetattr(o: object, name: str, value: object):
    parts = name.split(".")
    for n in parts[:-1]:
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return
    setattr(o, parts[-1], value)


def urljoin(*args):
    return "/".join([s.strip("/") for s in args])


def remote_file_exists(url):
    r = requests.head(url)
    return r.status_code == 200


# Copied from here: https://github.com/huggingface/huggingface_hub/blob/v0.25.0/src/huggingface_hub/file_download.py#L266
def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """Generate a local filename from a url.

    Convert `url` into a hashed filename in a reproducible way. If `etag` is
    specified, append its hash to the url's, delimited by a period. If the url
    ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)

    Args:
        url (`str`):
            The address to the file.
        etag (`str`, *optional*):
            The ETag of the file.

    Returns:
        The generated filename.
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


# Copied from last version of this method in HF codebase:
# https://github.com/huggingface/transformers/blob/9129fd0377e4d46cb2d0ea28dc1eb91a15f65b77/src/transformers/utils/hub.py#L460
def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = ADAPTER_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not found."
            )
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(
                url,
                headers=headers,
                allow_redirects=False,
                proxies=proxies,
                timeout=etag_timeout,
            )
            hf_raise_for_status(r)
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ProxyError,
            RepositoryNotFoundError,
            EntryNotFoundError,
            RevisionNotFoundError,
        ):
            # Actually raise for those subclasses of ConnectionError
            # Also raise the custom errors coming from a non existing repo/branch/file as they are caught later on.
            raise
        except (
            HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(
                    os.listdir(cache_dir), filename.split(".")[0] + ".*"
                )
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    fname = url.split("/")[-1]
                    raise EntryNotFoundError(
                        f"Cannot find the requested file ({fname}) in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(
                f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}"
            )

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def download_cached(
    url,
    checksum=None,
    checksum_algo="sha1",
    cache_dir=None,
    force_extract=False,
    **kwargs,
):
    """
    This method downloads a file and caches it.

    For more information on why this is needed, refer to the explanation in this Pull Request: https://github.com/adapter-hub/adapters/pull/750
    """
    if isinstance(url, Path):
        url = str(url)

    if is_remote_url(url):
        output_path = get_from_cache(url, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError("Unable to parse '{}' as a URL".format(url))

    if not output_path:
        return None

    # if checksum is given, verify it
    if checksum and checksum_algo:
        h = hashlib.new(checksum_algo)
        with open(output_path, "rb") as f:
            h.update(f.read())
        calculated_checksum = h.hexdigest()
        if calculated_checksum != checksum.lower():
            raise EnvironmentError(
                "Failed to verify checksum of '{}'".format(output_path)
            )

    if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
        return output_path

    # Path where we extract compressed archives
    # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    if (
        os.path.isdir(output_path_extracted)
        and os.listdir(output_path_extracted)
        and not force_extract
    ):
        return output_path_extracted

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, "r") as zip_file:
                # we want to extract all files into a flat folder structure (i.e. no subfolders)
                for file in zip_file.namelist():
                    # check if we have a valid file
                    if basename(file):
                        file_data = zip_file.read(file)
                        with open(
                            join(output_path_extracted, basename(file)), "wb"
                        ) as f:
                            f.write(file_data)
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError(
                "Archive format of {} could not be identified".format(output_path)
            )

    return output_path_extracted


def parse_adapter_config_string(config_string: str) -> List[Tuple[str, dict]]:
    """
    Parses an adapter configuration string into a list of tuples. Each tuple constists of an adapter config identifier
    and dictionary.
    """
    # First split by "|" into individual adapter configs
    config_string_chunks = config_string.split("|")
    # Now match each adapter config against the regex
    adapter_configs = []
    for config_string_chunk in config_string_chunks:
        match = re.match(ADAPTER_CONFIG_STRING_PATTERN, config_string_chunk.strip())
        if not match or not match.group("name"):
            raise ValueError(
                f"Invalid adapter config string format: '{config_string_chunk}'."
            )
        name = match.group("name")
        if match.group("kvs"):
            kvs = match.group("kvs")
            # Replace "=" with ":" in key-value pairs for valid Python dict
            kvs = re.sub(r"(\w+)=", r"'\1':", kvs)
        else:
            kvs = ""
        # Now evaluate key-value pairs as Python dict
        try:
            config_kwargs = ast.literal_eval("{" + kvs + "}")
        except Exception:
            raise ValueError(f"Invalid adapter configguration '{kvs}' in '{name}'.")
        adapter_configs.append((name, config_kwargs))

    return adapter_configs


def resolve_adapter_config(config: Union[dict, str], local_map=None, **kwargs) -> dict:
    """
    Resolves a given adapter configuration specifier to a full configuration dictionary.

    Args:
        config (Union[dict, str]): The configuration to resolve. Can be either:

            - a dictionary: returned without further action
            - an identifier string available in local_map
            - the path to a file containing a full adapter configuration

    Returns:
        dict: The resolved adapter configuration dictionary.
    """
    # already a dict, so we don't have to do anything
    if isinstance(config, Mapping):
        return config
    # first, look in local map
    if local_map and config in local_map:
        return local_map[config]
    # load from file system if it's a local file
    if isfile(config):
        with open(config, "r") as f:
            loaded_config = json.load(f)
            # search for nested config if the loaded dict has the form of a config saved with an adapter module
            if "config" in loaded_config:
                return loaded_config["config"]
            else:
                return loaded_config
    # parse the config string
    config_pairs = parse_adapter_config_string(config)
    if len(config_pairs) > 0:
        full_configs = []
        for name, config_kwargs in config_pairs:
            # first, look in local map
            if local_map and name in local_map:
                config_obj = local_map[name]
                full_configs.append(config_obj.replace(**config_kwargs))
            else:
                raise ValueError(
                    "Could not identify '{}' as a valid adapter configuration.".format(
                        name
                    )
                )
        # Case 1: only one config, return it directly
        if len(full_configs) == 1:
            return full_configs[0]
        # Case 2: multiple configs, return a config union
        elif len(full_configs) > 1:
            return {"architecture": "union", "configs": full_configs}

    raise ValueError(
        "Could not identify '{}' as a valid adapter configuration.".format(config)
    )


def _split_identifier(identifier):
    task, subtask, org_name = None, None, None
    identifier = identifier.split("@")
    if len(identifier) > 1:
        org_name = identifier[1]
    identifier = identifier[0].split("/")
    if len(identifier) > 1:
        subtask = identifier[1]
    task = identifier[0]
    return task, subtask, org_name


def _dict_extract(d, primary_key, secondary_key=None):
    for k, v in d.items():
        if k == primary_key:
            if secondary_key:
                if secondary_key in v.keys():
                    yield v[secondary_key]
            else:
                for k, v in v.items():
                    yield v
        elif secondary_key is None:
            for k, v in v.items():
                if k == primary_key:
                    yield v


def find_in_index(
    identifier: str,
    model_name: str,
    adapter_config: Optional[dict] = None,
    strict: bool = False,
    index_file: str = None,
) -> Optional[str]:
    identifier = identifier.strip()
    # identifiers of form "@<org>/<file>" are unique and can be retrieved directly
    match = re.match(r"@(\S+)\/(\S+)", identifier)
    if match:
        return ADAPTER_HUB_ADAPTER_ENTRY_JSON.format(match.group(1), match.group(2))

    if not index_file:
        index_file = download_cached(ADAPTER_HUB_INDEX_FILE.format(model_name))
    if not index_file:
        raise EnvironmentError(
            "Unable to load adapter hub index file. The file might be temporarily unavailable."
        )
    with open(index_file, "r") as f:
        adapter_index = json.load(f)
    # split into <task>/<subtask>@<org>
    task, subtask, org = _split_identifier(identifier)
    # find all entries for this task and subtask
    entries = list(_dict_extract(adapter_index, task, subtask))
    if not entries:
        # we found no matching entry
        return None
    elif len(entries) == 1:
        index_entry = entries[0]
    else:
        # there are multiple possible options for this identifier
        raise ValueError(
            "Found multiple possible adapters matching '{}'.".format(identifier)
        )
    # go on with searching a matching adapter_config hash in the task entry
    if adapter_config:
        config_hash = get_adapter_config_hash(adapter_config)
        if config_hash in index_entry:
            # now match the org if given
            hub_entry = _get_matching_version(index_entry[config_hash], org)
            if hub_entry:
                logger.info("Found matching adapter at: {}".format(hub_entry))
            return hub_entry
    # if we're here, no matching config is available or no config was given
    if not adapter_config or not strict:
        if "default" in index_entry:
            logger.info(
                "No exactly matching adapter config found for this specifier, falling back to default."
            )
            return index_entry["default"]
        # there's only one possible config and we allow matches with different configs
        elif len(index_entry) == 1:
            logger.info(
                "Only one configuration available for this adapter, using default."
            )
            config_entry = list(index_entry.values())[0]
            return _get_matching_version(config_entry, org)
    raise ValueError(
        "No adapter '{}' found for the current model or configuration.".format(
            identifier
        )
    )


def _get_matching_version(config_entry, org):
    if org:
        return config_entry["versions"].get(org, None)
    elif len(config_entry["versions"]) == 1:
        return list(config_entry["versions"].values())[0]
    elif "default" in config_entry:
        return config_entry["default"]
    else:
        raise ValueError(
            "Multiple adapters with this name are available for this config."
        )


def pull_from_hub(
    specifier: str,
    model_name: str,
    adapter_config: Optional[Union[dict, str]] = None,
    version: str = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Redirects loading from the archived Hub repository to HuggingFace Model Hub.

    Args:
        specifier (str): A string specifying the adapter to be loaded.
        model_name (str): The identifier of the pre-trained model for which to load an adapter.
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.
        strict (bool, optional):
            If set to True, only allow adapters exactly matching the given config to be loaded. Defaults to False.

    Returns:
        str: The local path to which the adapter has been downloaded.
    """
    if not model_name:
        raise ValueError(
            "Unable to resolve adapter without the name of a model. Please specify model_name."
        )
    # resolve config if it's an identifier
    if adapter_config:
        adapter_config = resolve_adapter_config(adapter_config)
    # search the correct entry in the index
    hub_entry_url = find_in_index(
        specifier, model_name, adapter_config=adapter_config, strict=strict
    )
    if not hub_entry_url:
        raise EnvironmentError(
            "No adapter with name '{}' was found in the adapter index.".format(
                specifier
            )
        )

    hf_hub_specifier = "AdapterHub/" + os.path.basename(hub_entry_url).split(".")[0]
    logger.warning(
        "Automatic redirect to HF Model Hub repo '{}'. Please switch to the new ID to remove this warning.".format(
            hf_hub_specifier
        )
    )
    return pull_from_hf_model_hub(hf_hub_specifier, version=version, **kwargs)


def pull_from_hf_model_hub(specifier: str, version: str = None, **kwargs) -> str:
    download_path = snapshot_download(
        specifier,
        revision=version,
        cache_dir=kwargs.pop("cache_dir", None),
        library_name="adapters",
        library_version=__version__,
    )
    return download_path


def resolve_adapter_path(
    adapter_name_or_path,
    model_name: str = None,
    adapter_config: Union[dict, str] = None,
    version: str = None,
    do_exists_check: bool = True,
    **kwargs,
) -> str:
    """
    Resolves the path to a pre-trained adapter module. Note: If attempting to resolve an adapter from the Hub,
    adapter_config and model_name must be present.

    Args:
        adapter_name_or_path (str): Can be either:

            - the path to a folder in the file system containing the adapter configuration and weights
            - an url pointing to a zip folder containing the adapter configuration and weights
            - a specifier matching a pre-trained adapter uploaded to Adapter-Hub
        model_name (str, optional): The identifier of the pre-trained model for which to load an adapter.
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.

    Returns:
        str: The local path from where the adapter module can be loaded.
    """
    # url of a folder containing pretrained adapters -> try to load from this url
    if is_remote_url(adapter_name_or_path):
        resolved_folder = download_cached(adapter_name_or_path, **kwargs)
        if not resolved_folder:
            raise EnvironmentError(
                "Unable to load file from {}. The file might be unavailable.".format(
                    resolved_folder
                )
            )
        return resolved_folder
    # path to a local folder saved using save()
    elif isdir(adapter_name_or_path):
        if (
            not do_exists_check
            or (
                isfile(join(adapter_name_or_path, WEIGHTS_NAME))
                or isfile(join(adapter_name_or_path, SAFE_WEIGHTS_NAME))
            )
            and isfile(join(adapter_name_or_path, CONFIG_NAME))
        ):
            return adapter_name_or_path
        else:
            raise EnvironmentError(
                "No file {} or no file {} found in directory {}".format(
                    WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path
                )
            )
    else:
        try:
            logger.info("Attempting to load adapter from HF Model Hub...")
            return pull_from_hf_model_hub(
                adapter_name_or_path, version=version, **kwargs
            )
        except (EnvironmentError, ValueError) as ex:
            logger.info(ex)
            logger.info("Attempting to redirect from archived Hub repo...")
            try:
                return pull_from_hub(
                    adapter_name_or_path,
                    model_name,
                    adapter_config=adapter_config,
                    version=version,
                    redirect_to_hf_hub=True,
                    **kwargs,
                )
            except Exception as ex:
                logger.info(ex)
                raise EnvironmentError(
                    "Unable to load adapter {} from any source. Please check the name of the adapter or the source.".format(
                        adapter_name_or_path
                    )
                )


def list_adapters(model_name: str = None) -> List[AdapterInfo]:
    """
    Retrieves a list of all publicly available adapters on AdapterHub.ml or on huggingface.co.

    Args:
        model_name (str, optional): If specified, only returns adapters trained for the model with this identifier.
    """
    adapters = []
    if "fetch_config" in inspect.signature(HfApi.list_models).parameters:
        kwargs = {"full": True, "fetch_config": True}
    else:
        logger.warning(
            "Using old version of huggingface-hub package for fetching. Please upgrade to latest version for"
            " accurate results."
        )
        kwargs = {"full": True}
    all_hf_adapters_data = HfApi().list_models(filter="adapters", **kwargs)
    for model_info in all_hf_adapters_data:
        adapter_info = AdapterInfo(
            source="hf",
            adapter_id=model_info.modelId,
            model_name=model_info.config.get("adapters", {}).get("model_name")
            if model_info.config
            else None,
            username=model_info.modelId.split("/")[0],
            sha1_checksum=model_info.sha,
        )
        adapters.append(adapter_info)

    if model_name is not None:
        adapters = [adapter for adapter in adapters if adapter.model_name == model_name]
    return adapters


def get_adapter_info(adapter_id: str) -> Optional[AdapterInfo]:
    """
    Retrieves information about a specific adapter.

    Args:
        adapter_id (str): The identifier of the adapter to retrieve.

    Returns:
        AdapterInfo: The adapter information or None if the adapter was not found.
    """
    try:
        model_info = HfApi().model_info(adapter_id)
        return AdapterInfo(
            source="hf",
            adapter_id=model_info.modelId,
            model_name=(
                model_info.config.get("adapter_transformers", {}).get("model_name")
                if model_info.config
                else None
            ),
            username=model_info.modelId.split("/")[0],
            sha1_checksum=model_info.sha,
        )
    except requests.exceptions.HTTPError:
        return None


def prefix_attention_mask(
    attention_mask, dim: Union[int, List[int]] = 3, prefix_value: int = 0
):
    """
    Adds a prefix to an attention mask. The length of the prefix is determined by the `prefix_attention_mask_length`
    attribute in the ForwardContext.

    Args:
        attention_mask:
            The attention mask to add the prefix to.
        dim (int):
            The dimension along which to concatenate the prefix_attention_mask. Defaults to 3.
        prefix_value (int):
            The value to use for the prefix_attention_mask. Defaults to 0, however some models, e.g. DistilBert, use
            different values. BERT like models invert their extended_attention_mask, hence they use 0 as value for not
            masked tokens. This inversion is usually done in the forward method of the model in 2 different ways:
            1) by calling self.invert_attention_mask, as BERT does 2) by doing the inversion manually, e.g. ALBERT
            does: `extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min`
    """

    forward_context = ForwardContext.get_context()

    if (
        attention_mask is not None
        and forward_context is not None
        and getattr(forward_context, "prompt_tokens_length", None) is not None
    ):
        if isinstance(dim, int):
            dim = [dim]
        for d in dim:
            # Create a tensor of ones with the desired shape
            ones_shape = list(attention_mask.shape)
            ones_shape[d] = forward_context.prompt_tokens_length

            prefix_attention_mask = torch.full(
                ones_shape,
                prefix_value,
                dtype=attention_mask.dtype,
            ).to(attention_mask.device)

            # Concatenate the prefix_attention_mask along the specified dimension
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=d)

    return attention_mask


def patch_forward(module: torch.nn.Module):
    # HF Accelerate's `add_hook_to_module()` replaces the module forward method with a wrapper
    # and stores the original forward method in `_old_forward`. For this to work with Adapters' post-hook wrapping,
    # we need to explicitly set to potentially overriden forward methods on adapter init.
    # The `add_hook_to_module()` method is e.g. used for `device_map="auto"` in the `PreTrainedModel.from_pretrained()` method.
    if hasattr(module, "_old_forward"):
        module._old_forward = module.__class__.forward.__get__(module, module.__class__)


def fix_seed(seed: Optional[int] = None):
    """
    Helper function to fix the torch seed on cpu and gpu for initializing adapters with the same weights.
    Is only executed if the config provides a respective seed.
    """
    if seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# Copied from https://github.com/huggingface/peft/blob/main/src/peft/utils/integrations.py.
def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            f"Input weight should be of type nn.Parameter, got {type(weight)} instead"
        )

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    import bitsandbytes as bnb

    if cls_name == "Params4bit":
        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state)

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()
