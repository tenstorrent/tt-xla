# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sentencizer model loader implementation for token classification.
"""

# Standard library imports
import os
import json
import gc
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Optional
from .src.models import get_sample_input
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer
from .src.adapter_utils import AdapterLoader

# TT-Forge imports
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel

# Import from src module
from .src import (
    MasterConfig,
    Multilingual_Embedding,
    TokenizerClassifier,
    Sentencizer,
    TokenizeDatasetLive,
    download,
    lang2treebank,
    tbname2max_input_length,
    tbname2tokbatchsize,
    NEWLINE_WHITESPACE_RE,
    get_start_char_idx,
    normalize_token,
    ID,
    TEXT,
    SENTENCES,
    LANG,
    DSPAN,
)


warnings.filterwarnings("ignore", category=UserWarning)


class ModelVariant(StrEnum):
    """Available Sentencizer model variants for token classification."""

    XLM_ROBERTA_BASE = "xlm-roberta-base"
    XLM_ROBERTA_LARGE = "xlm-roberta-large"


class ModelLoader(ForgeModel):
    """Sentencizer model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.XLM_ROBERTA_BASE: LLMModelConfig(
            pretrained_model_name="xlm-roberta-base",
            max_length=512,
        ),
        ModelVariant.XLM_ROBERTA_LARGE: LLMModelConfig(
            pretrained_model_name="xlm-roberta-large",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.XLM_ROBERTA_BASE

    def __init__(
        self, lang: Optional[str] = None, variant: Optional[ModelVariant] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        if lang is None:
            print("Using default english language sample Doc")
            self.lang = "english"
        else:
            self.lang = lang

        supported_langs = [
            lang
            for lang in list(lang2treebank.keys())
            if not lang.startswith("customized")
        ]
        assert (
            self.lang in supported_langs
        ), f"{self.lang} has not been supported. Currently supported languages: {supported_langs}"

        if variant is None:
            variant = self.DEFAULT_VARIANT

        supported_variants = [
            ModelVariant.XLM_ROBERTA_BASE,
            ModelVariant.XLM_ROBERTA_LARGE,
        ]
        assert (
            variant in supported_variants
        ), f"{variant} has not been supported.\nSupported embeddings: {[v.value for v in supported_variants]}"

        self.master_config = MasterConfig()

        self.master_config.embedding_name = variant.value
        self._ud_eval = False

        self.master_config.device = torch.device("cpu")
        self._tokbatchsize = 2

        self.master_config.wordpiece_splitter = XLMRobertaTokenizer.from_pretrained(
            self.master_config.embedding_name
        )
        self._config = self.master_config
        self._config.active_adapter = "None"
        self._config.max_input_length = tbname2max_input_length.get(
            lang2treebank[self.lang], 400
        )  # this is for tokenizer only

        self._config.training = False
        self.added_langs = [self.lang]
        assert (
            self.lang in lang2treebank
        ), f"{self.lang} has not been supported. Currently supported languages: {list(lang2treebank.keys())}"

        # download saved model for initial language
        cache_dir = download(
            language=self.lang,
            saved_model_version="v1.0.0",
            embedding_name=self.master_config.embedding_name,
        )
        self._cache_dir = cache_dir
        self.master_config._cache_dir = self._cache_dir
        if not os.path.exists(self.master_config._cache_dir):
            os.makedirs(self.master_config._cache_dir, exist_ok=True)

        # # load ALL vocabs
        # self._load_vocabs()
        self.batch = None
        self._config.vocabs = {}
        self._config.itos = defaultdict(dict)
        for lang in self.added_langs:
            treebank_name = lang2treebank[lang]
            with open(
                os.path.join(
                    self._config._cache_dir,
                    self.master_config.embedding_name,
                    f"{lang}/{lang}.vocabs.json",
                )
            ) as f:
                vocabs = json.load(f)
                self._config.vocabs[treebank_name] = vocabs
        self._embedding_layers = Multilingual_Embedding(self._config)
        self._embedding_layers.eval()
        self._adapter_loader = AdapterLoader(self._embedding_layers.xlmr, "text_task")

        # tokenizers
        self._tokenizer = {}
        self._tokenizer[lang] = TokenizerClassifier(
            self._config, treebank_name=lang2treebank[lang]
        )
        self._tokenizer[lang].to(self._config.device)

        self._tokenizer[lang].eval()

        # load and hold the pretrained weights
        self._embedding_weights = self._embedding_layers.state_dict()
        self._config.active_lang = lang
        self.active_lang = lang
        self._config.active_adapter = "None"
        self._config.treebank_name = lang2treebank[lang]
        self._config.max_input_length = tbname2max_input_length.get(
            lang2treebank[lang], 400
        )  # this is for tokenizer only
        print("=" * 50)
        print(f"Active language: {self._config.active_lang}")
        print("=" * 50)
        pretrained_weights = self._tokenizer[
            self._config.active_lang
        ].pretrained_tokenizer_weights
        self._adapter_loader.load_from_state_dict(
            pretrained_weights, "tokenizer", load_as="embedding", start_prefix="xlmr."
        )
        # save information of active adapter
        self._config.active_adapter = "tokenizer"

    @classmethod
    def _get_model_info(cls, variant: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Sentencizer",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.XLM_ROBERTA_BASE
            else ModelGroup.GENERALITY,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load Sentencizer model for token classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Sentencizer model instance.
        """
        model = Sentencizer(
            self._embedding_layers, self._tokenizer[self._config.active_lang]
        )
        return model

    def load_inputs(self, in_doc: Optional[str] = None, dtype_override=None):
        """Prepare sample input for Sentencizer token classification.

        Args:
            in_doc: Optional input document string. If not provided, the model will use the sample text.
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if (self.lang is not None) and (in_doc is None):
            self.sample_text = get_sample_input(self.lang)
        elif (self.lang is not None) and (in_doc is not None):
            self.sample_text = in_doc
        else:
            self.sample_text = get_sample_input(self.lang)
        eval_batch_size = tbname2tokbatchsize.get(
            lang2treebank[self.active_lang], self._tokbatchsize
        )
        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(
            config,
            self.sample_text,
            max_input_length=tbname2max_input_length.get(
                lang2treebank[self.active_lang], 400
            ),
        )
        test_set.numberize(config.wordpiece_splitter)

        self.batch = [
            batch
            for batch in DataLoader(
                test_set,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=test_set.collate_fn,
            )
        ]
        inputs = (self.batch[0].piece_idxs, self.batch[0].attention_masks)

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for token classification.

        Args:
            co_out: Model output
            framework_model: Framework model with config (needed for id2label mapping)
        """

        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        wp_pred_labels = co_out.data.cpu().numpy().tolist()

        for i in range(len(wp_pred_labels)):
            wordpiece_pred_labels.append(
                wp_pred_labels[i][: len(self.batch[0].wordpiece_ends[i])]
            )

        wordpiece_ends.extend(self.batch[0].wordpiece_ends)
        paragraph_indexes.extend(self.batch[0].paragraph_index)
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(
            wordpiece_pred_labels, wordpiece_ends, paragraph_indexes
        ):
            para_id_to_wp_pred_labels[p_index].extend(
                [
                    (pred, char_position)
                    for pred, char_position in zip(wp_pred_ls, wp_es)
                ]
            )
        corpus_text = self.sample_text
        paragraphs = [
            pt.rstrip()
            for pt in NEWLINE_WHITESPACE_RE.split(corpus_text)
            if len(pt.rstrip()) > 0
        ]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        cloned_raw_text = deepcopy(corpus_text)
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            cloned_raw_text, start_char_idx = get_start_char_idx(
                para_text, cloned_raw_text
            )
            start_char_idx += global_offset
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)

        ###########################
        sentences = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ""
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(
                        self._config.treebank_name, current_tok, ud_eval=self._ud_eval
                    )
                    assert "\t" not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ""
                        continue
                    additional_info = {
                        DSPAN: (
                            para_start + local_position - len(tok),
                            para_start + local_position,
                        )
                    }
                    current_sent += [(tok, wp_p, additional_info)]
                    current_tok = ""
                    if wp_p == 2 or wp_p == 4:
                        sent_span = (
                            current_sent[0][2][DSPAN][0],
                            current_sent[-1][2][DSPAN][1],
                        )
                        sentences.append(
                            {
                                ID: len(sentences) + 1,
                                TEXT: corpus_text[sent_span[0] : sent_span[1]],
                                DSPAN: (sent_span[0], sent_span[1]),
                            }
                        )
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(
                    self._config.treebank_name, current_tok, ud_eval=self._ud_eval
                )
                assert "\t" not in tok, tok
                if len(tok) > 0:
                    additional_info = {
                        DSPAN: (
                            para_start + local_position - len(tok),
                            para_start + local_position,
                        )
                    }
                    current_sent += [(tok, 2, additional_info)]

            if len(current_sent):
                sent_span = (
                    current_sent[0][2][DSPAN][0],
                    current_sent[-1][2][DSPAN][1],
                )
                sentences.append(
                    {
                        ID: len(sentences) + 1,
                        TEXT: corpus_text[sent_span[0] : sent_span[1]],
                        DSPAN: (sent_span[0], sent_span[1]),
                    }
                )

        # Memory cleanup
        gc.collect()

        print(f"Context: {self.sample_text}")
        print(f"Answer: {sentences}")
        return {TEXT: self.sample_text, SENTENCES: sentences, LANG: self.active_lang}
