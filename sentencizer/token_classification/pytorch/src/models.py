# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import warnings

import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from . import adapter_utils as adapters
from .adapter_utils import AdapterConfig
from .utils import get_file

# Import utility function and regex patterns from utils
from .utils import word_lens_to_idxs_fast, PUNCTUATION, NEWLINE_WHITESPACE_RE, SPACE_RE

warnings.filterwarnings("ignore", category=UserWarning)

# Constants and mappings
supported_embeddings = ["xlm-roberta-base", "xlm-roberta-large"]
saved_model_version = "v1.0.0"

# Language mappings
code2lang = {
    "af": "afrikaans",
    "ar": "arabic",
    "hy": "armenian",
    "eu": "basque",
    "be": "belarusian",
    "bg": "bulgarian",
    "ca": "catalan",
    "zh": "chinese",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "de": "german",
    "el": "greek",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "ja": "japanese",
    "kk": "kazakh",
    "ko": "korean",
    "ku": "kurmanji",
    "la": "latin",
    "lv": "latvian",
    "lt": "lithuanian",
    "mr": "marathi",
    "nn": "norwegian-nynorsk",
    "nb": "norwegian-bokmaal",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "sk": "slovak",
    "sl": "slovenian",
    "es": "spanish",
    "sv": "swedish",
    "ta": "tamil",
    "te": "telugu",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "ug": "uyghur",
    "vi": "vietnamese",
}

lang2code = {v: k for k, v in code2lang.items()}

extra_lang2code = {
    "afrikaans": "af",
    "ancient-greek-perseus": "el",
    "ancient-greek": "el",
    "arabic": "ar",
    "armenian": "hy",
    "basque": "eu",
    "belarusian": "be",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "traditional-chinese": "zh",
    "classical-chinese": "zh",
    "croatian": "hr",
    "czech-cac": "cs",
    "czech-cltt": "cs",
    "czech-fictree": "cs",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "dutch-lassysmall": "nl",
    "english": "en",
    "english-gum": "en",
    "english-lines": "en",
    "english-partut": "en",
    "estonian": "et",
    "estonian-ewt": "et",
    "finnish-ftb": "fi",
    "finnish": "fi",
    "french": "fr",
    "french-partut": "fr",
    "french-sequoia": "fr",
    "french-spoken": "fr",
    "galician": "gl",
    "galician-treegal": "gl",
    "german": "de",
    "german-hdt": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "italian-partut": "it",
    "italian-postwita": "it",
    "italian-twittiro": "it",
    "italian-vit": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "korean-kaist": "ko",
    "kurmanji": "ku",
    "latin": "la",
    "latin-perseus": "la",
    "latin-proiel": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "lithuanian-hse": "lt",
    "marathi": "mr",
    "norwegian-nynorsk": "nn",
    "norwegian-nynorsklia": "nn",
    "norwegian-bokmaal": "nb",
    "old-french": "fr",
    "old-russian": "ru",
    "persian": "fa",
    "polish-lfg": "pl",
    "polish": "pl",
    "portuguese": "pt",
    "portuguese-gsd": "pt",
    "romanian-nonstandard": "ro",
    "romanian": "ro",
    "russian-gsd": "ru",
    "russian": "ru",
    "russian-taiga": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "slovenian-sst": "sl",
    "spanish": "es",
    "spanish-gsd": "es",
    "swedish-lines": "sv",
    "swedish": "sv",
    "tamil": "ta",
    "telugu": "te",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uyghur": "ug",
    "vietnamese": "vi",
    "vietnamese-vtb": "vi",
}

lang2treebank = {
    "afrikaans": "UD_Afrikaans-AfriBooms",
    "ancient-greek-perseus": "UD_Ancient_Greek-Perseus",
    "ancient-greek": "UD_Ancient_Greek-PROIEL",
    "arabic": "UD_Arabic-PADT",
    "armenian": "UD_Armenian-ArmTDP",
    "basque": "UD_Basque-BDT",
    "belarusian": "UD_Belarusian-HSE",
    "bulgarian": "UD_Bulgarian-BTB",
    "catalan": "UD_Catalan-AnCora",
    "chinese": "UD_Simplified_Chinese-GSDSimp",
    "traditional-chinese": "UD_Chinese-GSD",
    "classical-chinese": "UD_Classical_Chinese-Kyoto",
    "croatian": "UD_Croatian-SET",
    "czech-cac": "UD_Czech-CAC",
    "czech-cltt": "UD_Czech-CLTT",
    "czech-fictree": "UD_Czech-FicTree",
    "czech": "UD_Czech-PDT",
    "danish": "UD_Danish-DDT",
    "dutch": "UD_Dutch-Alpino",
    "dutch-lassysmall": "UD_Dutch-LassySmall",
    "english": "UD_English-EWT",
    "english-gum": "UD_English-GUM",
    "english-lines": "UD_English-LinES",
    "english-partut": "UD_English-ParTUT",
    "estonian": "UD_Estonian-EDT",
    "estonian-ewt": "UD_Estonian-EWT",
    "finnish-ftb": "UD_Finnish-FTB",
    "finnish": "UD_Finnish-TDT",
    "french": "UD_French-GSD",
    "french-partut": "UD_French-ParTUT",
    "french-sequoia": "UD_French-Sequoia",
    "french-spoken": "UD_French-Spoken",
    "galician": "UD_Galician-CTG",
    "galician-treegal": "UD_Galician-TreeGal",
    "german": "UD_German-GSD",
    "german-hdt": "UD_German-HDT",
    "greek": "UD_Greek-GDT",
    "hebrew": "UD_Hebrew-HTB",
    "hindi": "UD_Hindi-HDTB",
    "hungarian": "UD_Hungarian-Szeged",
    "indonesian": "UD_Indonesian-GSD",
    "irish": "UD_Irish-IDT",
    "italian": "UD_Italian-ISDT",
    "italian-partut": "UD_Italian-ParTUT",
    "italian-postwita": "UD_Italian-PoSTWITA",
    "italian-twittiro": "UD_Italian-TWITTIRO",
    "italian-vit": "UD_Italian-VIT",
    "japanese": "UD_Japanese-GSD",
    "kazakh": "UD_Kazakh-KTB",
    "korean": "UD_Korean-GSD",
    "korean-kaist": "UD_Korean-Kaist",
    "kurmanji": "UD_Kurmanji-MG",
    "latin": "UD_Latin-ITTB",
    "latin-perseus": "UD_Latin-Perseus",
    "latin-proiel": "UD_Latin-PROIEL",
    "latvian": "UD_Latvian-LVTB",
    "lithuanian": "UD_Lithuanian-ALKSNIS",
    "lithuanian-hse": "UD_Lithuanian-HSE",
    "marathi": "UD_Marathi-UFAL",
    "norwegian-nynorsk": "UD_Norwegian_Nynorsk-Nynorsk",
    "norwegian-nynorsklia": "UD_Norwegian_Nynorsk-NynorskLIA",
    "norwegian-bokmaal": "UD_Norwegian-Bokmaal",
    "old-french": "UD_Old_French-SRCMF",
    "old-russian": "UD_Old_Russian-TOROT",
    "persian": "UD_Persian-Seraji",
    "polish-lfg": "UD_Polish-LFG",
    "polish": "UD_Polish-PDB",
    "portuguese": "UD_Portuguese-Bosque",
    "portuguese-gsd": "UD_Portuguese-GSD",
    "romanian-nonstandard": "UD_Romanian-Nonstandard",
    "romanian": "UD_Romanian-RRT",
    "russian-gsd": "UD_Russian-GSD",
    "russian": "UD_Russian-SynTagRus",
    "russian-taiga": "UD_Russian-Taiga",
    "scottish-gaelic": "UD_Scottish_Gaelic-ARCOSG",
    "serbian": "UD_Serbian-SET",
    "slovak": "UD_Slovak-SNK",
    "slovenian": "UD_Slovenian-SSJ",
    "slovenian-sst": "UD_Slovenian-SST",
    "spanish": "UD_Spanish-AnCora",
    "spanish-gsd": "UD_Spanish-GSD",
    "swedish-lines": "UD_Swedish-LinES",
    "swedish": "UD_Swedish-Talbanken",
    "tamil": "UD_Tamil-TTB",
    "telugu": "UD_Telugu-MTG",
    "turkish": "UD_Turkish-IMST",
    "ukrainian": "UD_Ukrainian-IU",
    "urdu": "UD_Urdu-UDTB",
    "uyghur": "UD_Uyghur-UDT",
    "vietnamese": "UD_Vietnamese-VLSP",
    "vietnamese-vtb": "UD_Vietnamese-VTB",
    "customized": "UD_Customized",
    "customized-mwt": "UD_Customized-MWT",
    "customized-ner": "UD_Customized-NER",
    "customized-mwt-ner": "UD_Customized-MWT-NER",
}

supported_langs = [
    lang for lang in list(lang2treebank.keys()) if not lang.startswith("customized")
]
treebank2lang = {v: k for k, v in lang2treebank.items()}

tbname2max_input_length = {
    "UD_Ancient_Greek-PROIEL": 350,
    "UD_Belarusian-HSE": 450,
    "UD_Chinese-GSD": 450,
    "UD_Czech-CLTT": 512,
    "UD_Dutch-LassySmall": 512,
    "UD_English-LinES": 450,
    "UD_French-GSD": 512,
    "UD_French-Sequoia": 512,
    "UD_Irish-IDT": 512,
    "UD_Italian-ParTUT": 512,
    "UD_Italian-TWITTIRO": 512,
    "UD_Japanese-GSD": 512,
    "UD_Latin-PROIEL": 512,
    "UD_Marathi-UFAL": 512,
    "UD_Norwegian_Nynorsk-NynorskLIA": 512,
    "UD_Norwegian-Bokmaal": 512,
    "UD_Old_Russian-TOROT": 512,
    "UD_Persian-Seraji": 512,
    "UD_Polish-LFG": 512,
    "UD_Romanian-RRT": 512,
}

tbname2tokbatchsize = {}
tbname2tagbatchsize = {"UD_Belarusian-HSE": 16}

# Constants for output format
ID = "id"
TEXT = "text"
SENTENCES = "sentences"
TOKENS = "tokens"
LANG = "lang"
DSPAN = "dspan"
SSPAN = "sspan"
MISC = "misc"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
EXPANDED = "expanded"

# Regular expressions (imported from utils.py to avoid circular imports)
NUMERIC_RE = re.compile(r"^([\d]+[,\.]*)+$")
WHITESPACE_RE = re.compile(r"\s")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class MasterConfig:
    def __init__(self):
        self.embedding_name = "xlm-roberta-base"
        self.embedding_dropout = 0.3
        self.hidden_num = 300
        self.linear_dropout = 0.1
        self.linear_bias = 1
        self.linear_activation = "relu"
        self.adapter_learning_rate = 1e-4
        self.learning_rate = 1e-3
        self.adapter_weight_decay = 1e-4
        self.weight_decay = 1e-3
        self.grad_clipping = 4.5
        self.working_dir = os.path.dirname(os.path.realpath(__file__))
        self.lowercase = False


# Neural network classes
class Linears(nn.Module):
    def __init__(self, dimensions, activation="relu", dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                for i in range(len(dimensions) - 1)
            ]
        )
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class Base_Model(nn.Module):
    def __init__(self, config, task_name):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.xlmr_dim = 768 if config.embedding_name == "xlm-roberta-base" else 1024
        self.xlmr = XLMRobertaModel.from_pretrained(
            config.embedding_name,
            cache_dir=os.path.join(config._cache_dir, config.embedding_name),
            output_hidden_states=True,
        )
        adapters.init(self.xlmr)

        self.xlmr_dropout = nn.Dropout(p=config.embedding_dropout)
        task_config = AdapterConfig.load(
            "pfeiffer",
            reduction_factor=6 if config.embedding_name == "xlm-roberta-base" else 4,
        )
        self.xlmr.add_adapter(task_name, config=task_config)
        self.xlmr.set_active_adapters([task_name])

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]
        wordpiece_reprs = self.xlmr_dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = (
            piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        )
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1, idxs) * masks
        xlmr_outputs = xlmr_outputs.view(
            batch_size, token_num, token_len, self.xlmr_dim
        )
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError


class Multilingual_Embedding(Base_Model):
    def __init__(self, config, model_name="embedding"):
        super(Multilingual_Embedding, self).__init__(config, task_name=model_name)

    def get_tokenizer_inputs(self, piece_idxs, attention_masks):
        wordpiece_reprs = self.encode(
            piece_idxs=piece_idxs, attention_masks=attention_masks
        )
        return wordpiece_reprs

    def get_tagger_inputs(self, piece_idxs, attention_masks, word_lens):
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs=piece_idxs, attention_masks=attention_masks, word_lens=word_lens
        )
        return word_reprs, cls_reprs


class TokenizerClassifier(nn.Module):
    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == "xlm-roberta-base" else 1024
        self.tokenizer_ffn = nn.Linear(self.xlmr_dim, 5)

        if not config.training:
            language = treebank2lang[treebank_name]
            self.pretrained_tokenizer_weights = torch.load(
                os.path.join(
                    self.config._cache_dir,
                    self.config.embedding_name,
                    language,
                    "{}.tokenizer.mdl".format(language),
                ),
                map_location=self.config.device,
            )["adapters"]
            self.initialized_weights = self.state_dict()

            for name, value in self.pretrained_tokenizer_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value

            self.load_state_dict(self.initialized_weights)
            print("Loading tokenizer for {}".format(language))

    def forward(self, wordpiece_reprs):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        predicted_wordpiece_labels = torch.argmax(wordpiece_scores, dim=2)
        return predicted_wordpiece_labels


class Sentencizer(nn.Module):
    def __init__(self, embedding_layers, tokenizer_classifier):
        super().__init__()
        self.embedding_layers = embedding_layers
        self.tokenizer_classifier = tokenizer_classifier

    def forward(self, piece_idxs, attention_masks):
        wordpiece_reprs = self.embedding_layers.get_tokenizer_inputs(
            piece_idxs, attention_masks
        )
        return self.tokenizer_classifier(wordpiece_reprs)


def get_sample_input(language):
    """Download sample input for a specific language from GitHub."""
    # Base GitHub URL for the sample inputs
    base_url = "https://raw.githubusercontent.com/nlp-uoregon/trankit/master/trankit/tests/sample_inputs/"

    # Construct filename by adding .txt extension
    filename = f"{language}.txt"

    # Construct the full GitHub URL
    file_url = base_url + filename

    # Download the file content from GitHub
    file_path = get_file(file_url)
    if file_path and file_path.exists():
        # Read the content from the downloaded file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content
    else:
        print(f"Warning: File not found for {filename}")
        return ""
