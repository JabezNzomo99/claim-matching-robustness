#!/usr/bin/python
# Copyright (c) Lydia Nishimwe.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Implemenation of CustomTokenizer class for using LASER, RoLASER and c-RoLASER models
# From the paper "Making Sentence Embeddings Robust to User-Generated Content" by Nishimwe et al., 2024

import os
from transformers import RobertaTokenizer, BertTokenizer
import sentencepiece as spm

assert os.environ.get("LASER"), "Please set the enviornment variable LASER"
LASER = os.environ["LASER"]
SPM_MODEL_PATH = LASER + "/laser2/laser2.spm"
assert os.path.isfile(SPM_MODEL_PATH), f"SPM model {SPM_MODEL_PATH} not found."

SUPPORTED_TOKENIZERS = ["char", "roberta", "spm"]


class CustomTokenizer:
    def __init__(self, tokenizer_name="spm"):
        self.tokenizer_name = tokenizer_name
        if tokenizer_name == "char":
            self.tokenizer_model = BertTokenizer.from_pretrained(
                "bert-base-cased"
            ).basic_tokenizer
        elif tokenizer_name == "roberta":
            self.tokenizer_model = RobertaTokenizer.from_pretrained("roberta-base")
        elif tokenizer_name == "spm":
            self.tokenizer_model = spm.SentencePieceProcessor()
            self.tokenizer_model.load(SPM_MODEL_PATH)
        else:
            raise ValueError(
                f"The tokenizer {tokenizer_name} is unknown. Expected values are {SUPPORTED_TOKENIZERS}."
            )

    def tokenize(self, line):
        if self.tokenizer_name == "char":
            tokens = self.tokenizer_model.tokenize(line.strip())
            characters = [" ".join(token) for token in tokens]
            return " _EOW ".join(characters) + " _EOW"
        if self.tokenizer_name == "roberta":
            tokens = self.tokenizer_model.tokenize(line.strip())
            return " ".join(tokens)
        # SPM
        tokens = self.tokenizer_model.encode_as_pieces(line.strip())
        return " ".join(tokens)
