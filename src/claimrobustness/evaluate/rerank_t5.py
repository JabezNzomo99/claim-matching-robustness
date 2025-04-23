import os
import csv
import torch
import sys
import argparse
import pandas as pd
import numpy as np
from math import ceil, exp
import subprocess
from typing import List
import time
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    "castorini/monot5-small-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-small-msmarco-100k": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
}


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class MonoT5Reranker:
    name: str = "MonoT5"
    prompt_template: str = "Query: {query} Document: {text} Relevant:"

    def __init__(
        self,
        device,
        batch_size=32,
        silent=False,
        model_name_or_path="castorini/monot5-base-msmarco-10k",
        token_false=None,
        token_true=True,
        torch_compile=False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.silent = silent
        model_args = {}
        # if self.fp16:
        #     model_args["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, **model_args
        )
        self.torch_compile = torch_compile
        if torch_compile:
            self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path,
            self.tokenizer,
            token_false,
            token_true,
        )
        print(f"Max tokens is {self.tokenizer.model_max_length}")

    def get_prediction_tokens(
        self, model_name_or_path, tokenizer, token_false=None, token_true=None
    ):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                print("Loading from base....")
                # raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                #         the checkpoint {model_name_or_path} and you did not provide any.")
                returned = self.get_prediction_tokens(
                    "castorini/monot5-base-msmarco", self.tokenizer
                )
                print(returned)
                return returned
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        max_count = 0
        for batch in tqdm(
            chunks(pairs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / self.batch_size),
            bar_format="{l_bar}{bar}{r_bar}\n",
        ):
            prompts = [
                self.prompt_template.format(query=query, text=text)
                for (query, text) in batch
            ]
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                pad_to_multiple_of=(8 if self.torch_compile else None),
            ).to(self.device)
            if tokens.input_ids.shape[1] == self.tokenizer.model_max_length:
                max_count += 1
            # print(tokens["input_ids"].shape)
            output = self.model.generate(
                **tokens,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_scores = output.scores[0]
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores += batch_scores[:, 1].exp().tolist()
        print(f"Max count is {max_count}/{len(list(chunks(pairs, self.batch_size)))}")
        return scores
