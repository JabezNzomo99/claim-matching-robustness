# import torch
# from math import ceil
# from typing import List
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm.auto import tqdm


# def split_into_paragraphs(text: str, max_tokens: int = 512, tokenizer=None):
#     """
#     Splits `text` by double newline as a simple paragraph segmentation.
#     A more sophisticated approach would use `tokenizer` to ensure
#     each paragraph does not exceed `max_tokens`.
#     """
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     return paragraphs


# def chunks(lst, n):
#     """
#     Yields successive n-sized chunks from lst.
#     """
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]


# class BGELLMReranker:
#     name: str = "BGE-Reranker"

#     def __init__(
#         self,
#         device,
#         model_name_or_path: str = "BAAI/bge-reranker-v2-gemma",
#         batch_size: int = 32,
#         max_length: int = 3072,
#         fp16: bool = True,
#         silent: bool = False,
#         prompt: str = None,
#     ):

#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.silent = silent
#         self.device = device
#         self.fp16 = fp16
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         model_args = {}
#         if self.fp16:
#             model_args["torch_dtype"] = torch.bfloat16
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path, **model_args
#         )
#         self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
#         self.fp16 = fp16

#         self.model.eval()
#         self.model.to(device)
#         self.prompt = prompt

#     @staticmethod
#     def get_inputs(
#         flat_pairs,  # list of (query, paragraph) pairs
#         tokenizer,
#         prompt=None,
#         max_length=3072,
#     ):
#         """
#         Prepares model inputs for the given (query, paragraph) pairs, appended with the prompt.
#         """
#         if prompt is None:
#             prompt = (
#                 "Given a claim A and a fact check B, predict 'Yes' if the fact check "
#                 "helps to address and verify the truth or falsity of the claim (including negated claims), "
#                 "or 'No' if it does not."
#             )
#         sep = "\n"
#         prompt_inputs = tokenizer(
#             prompt, return_tensors=None, add_special_tokens=False
#         )["input_ids"]
#         sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)[
#             "input_ids"
#         ]

#         inputs = []
#         for query, paragraph in flat_pairs:
#             query_inputs = tokenizer(
#                 f"A: {query}",
#                 return_tensors=None,
#                 add_special_tokens=False,
#                 max_length=max_length * 3 // 4,
#                 truncation=True,
#             )
#             paragraph_inputs = tokenizer(
#                 f"B: {paragraph}",
#                 return_tensors=None,
#                 add_special_tokens=False,
#                 max_length=max_length,
#                 truncation=True,
#             )
#             item = tokenizer.prepare_for_model(
#                 [tokenizer.bos_token_id] + query_inputs["input_ids"],
#                 sep_inputs + paragraph_inputs["input_ids"],
#                 truncation="only_second",
#                 max_length=max_length,
#                 padding=False,
#                 return_attention_mask=False,
#                 return_token_type_ids=False,
#                 add_special_tokens=False,
#             )
#             # Append the final prompt after the paragraph
#             item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
#             item["attention_mask"] = [1] * len(item["input_ids"])
#             inputs.append(item)

#         return tokenizer.pad(
#             inputs,
#             padding="max_length",
#             max_length=max_length + len(sep_inputs) + len(prompt_inputs),
#             pad_to_multiple_of=8,
#             return_tensors="pt",
#         )

#     def rescore(self, pairs: List[List[str]]):
#         """
#         Splits each passage (B) into paragraphs, encodes them all in a batched manner,
#         and aggregates paragraph-level scores via max-pooling (or your chosen method).
#         """
#         # Step 1: Flatten out the pairs so that each passage is split by paragraphs.
#         # We'll keep track of which pair (index) these paragraphs came from,
#         # so we can combine scores afterwards.
#         flat_pairs = []
#         pair_to_paragraph_indices = (
#             []
#         )  # For each pair, which row indices in flat_pairs belong to it

#         for idx, (query, passage) in enumerate(pairs):
#             paragraphs = split_into_paragraphs(passage)
#             current_indices = []
#             for paragraph in paragraphs:
#                 flat_pairs.append((query, paragraph))
#                 current_indices.append(len(flat_pairs) - 1)
#             pair_to_paragraph_indices.append(current_indices)

#         # Step 2: We will rescore the entire flattened list in mini-batches
#         # and store the per-paragraph scores.
#         paragraph_scores = [None] * len(
#             flat_pairs
#         )  # placeholder for each paragraph's score

#         for batch_indices in tqdm(
#             list(chunks(range(len(flat_pairs)), self.batch_size)),
#             disable=self.silent,
#             desc="Rescoring (paragraph-level)",
#             total=ceil(len(flat_pairs) / self.batch_size),
#             bar_format="{l_bar}{bar}{r_bar}\n",
#         ):
#             sub_flat_pairs = [flat_pairs[i] for i in batch_indices]
#             with torch.no_grad():
#                 inputs = self.get_inputs(
#                     sub_flat_pairs, self.tokenizer, self.prompt, self.max_length
#                 )
#                 inputs = inputs.to(self.device)
#                 logits = self.model(**inputs, return_dict=True).logits
#                 # Get the logit for "Yes" token at the last position
#                 # shape: [batch_size, vocab_size]
#                 last_token_logits = logits[:, -1, :]
#                 # We only pick out the dimension corresponding to the "Yes" token
#                 yes_scores = last_token_logits[:, self.yes_loc]

#             for i, idx_in_batch in enumerate(batch_indices):
#                 paragraph_scores[idx_in_batch] = yes_scores[i].item()

#         # Step 3: Combine paragraph-level scores back into pair-level scores
#         # We'll use max() across all paragraphs for each passage, but you can
#         # also try average or sum.
#         final_scores = []
#         for indices in pair_to_paragraph_indices:
#             # gather the scores for these paragraphs
#             paragraph_level = [paragraph_scores[i] for i in indices]
#             # for example, take max across paragraphs
#             final_scores.append(
#                 max(paragraph_level) if paragraph_level else float("-inf")
#             )

#         return final_scores
import torch
from math import ceil, exp
import subprocess
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BGELLMReranker:
    name: str = "BGE-Reranker"

    def __init__(
        self,
        device,
        model_name_or_path: str = "BAAI/bge-reranker-v2-gemma",
        batch_size: int = 16,
        max_length: int = 3072,
        fp16: bool = True,
        silent: bool = False,
        prompt: str = None,
    ):

        self.batch_size = batch_size
        self.max_length = max_length
        self.silent = silent
        self.device = device
        self.fp16 = fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_args = {}
        if self.fp16:
            model_args["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_args
        )
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.fp16 = fp16

        self.model.eval()
        self.model.to(device)
        self.prompt = prompt

    @staticmethod
    def get_inputs(pairs, tokenizer, prompt=None, max_length=3072):
        # Changed to
        if prompt is None:
            prompt = "Given a claim A and a fact check B, predict 'Yes' if the fact check helps to address and verify the truth or falsity of the claim (including negated claims), or 'No' if it does not."
        sep = "\n"
        prompt_inputs = tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)
        return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

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
            with torch.no_grad():
                inputs = self.get_inputs(
                    batch, self.tokenizer, self.prompt, self.max_length
                )
                inputs = inputs.to(self.device)
                batch_scores = (
                    self.model(**inputs, return_dict=True)
                    .logits[:, -1, self.yes_loc]
                    .view(
                        -1,
                    )
                    .float()
                )
                scores += batch_scores.tolist()
        print(f"Max count is {max_count}/{len(list(chunks(pairs, self.batch_size)))}")
        return scores
