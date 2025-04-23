import torch
from math import ceil, exp
import subprocess
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BGEReranker:
    name: str = "BGE-Reranker"

    def __init__(
        self,
        device,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
        max_length: int = 512,
        silent: bool = False,
    ):

        self.batch_size = batch_size
        self.max_length = max_length
        self.silent = silent
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.model.eval()
        self.model.to(device)

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
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            batch_scores = (
                self.model(**tokens, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            print(batch_scores)
            scores += batch_scores.tolist()
        print(f"Max count is {max_count}/{len(list(chunks(pairs, self.batch_size)))}")
        return scores
