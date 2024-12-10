import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from functools import partial
import numpy as np
import argparse
import configparser
from tqdm import tqdm
from claimrobustness import utils, defaults
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import dataloaders
import jsonlines


def get_most_recent_checkpoint_filename(path):
    max_step_number = max([int(ckpt.split("-")[-1]) for ckpt in os.listdir(path)])
    return f"checkpoint-{max_step_number}"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config.ini lies")
    parser.add_argument("--candidate-selection", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="specify ckpt instead of most recent",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=None, help="overwrite number of candidates"
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Load Model
    model_str = config["rerank-evaluation"].get("model_string")
    ckpt_str = os.path.join(
        args.experiment_path, config["rerank-evaluation"].get("save_dir")
    )
    ckpt_filename = get_most_recent_checkpoint_filename(ckpt_str)
    if args.checkpoint:
        ckpt_filename = f"checkpoint-{args.checkpoint}"

    if not args.raw:
        ckpt_str = os.path.join(ckpt_str, ckpt_filename)
    else:
        ckpt_str = model_str
    print(f"Loading model from {ckpt_str}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(ckpt_str)

    # Setup Data
    if args.n_candidates is None:
        N_CANDIDATES = config["rerank-evaluation"].getint("n_candidates", 5)
    else:
        N_CANDIDATES = args.n_candidates
    print(f"Reranking top {N_CANDIDATES}")

    # Load the dataset name
    dataset = config["data"].get("dataset")

    embedding_models = config["rerank-evaluation"].get("embedding_models").split(",")
    print(f"Running evaluation on the following models: {embedding_models}")

    # Load the original baseline and edited baseline
    original_baseline_path = os.path.join(
        args.experiment_path,
        dataset,
        config["evaluation"].get("original_baseline_path"),
    )
    edited_baseline_path = os.path.join(
        args.experiment_path, dataset, config["evaluation"].get("edited_baseline_path")
    )

    # Load the original worstcase and edited worstcase
    original_worstcase_path = os.path.join(
        args.experiment_path,
        dataset,
        config["evaluation"].get("original_worstcase_path"),
    )
    edited_worstcase_path = os.path.join(
        args.experiment_path, dataset, config["evaluation"].get("edited_worstcase_path")
    )

    def load_evaluation_data(path: str) -> pd.DataFrame:
        return pd.read_csv(
            path, names=["query_id", "query"], skiprows=[0]
        ).drop_duplicates()

    # Load the datasets
    data = utils.load_data(dataset=dataset)
    targets = data["targets"]
    _, test_qrels = data["test"]

    original_baseline = load_evaluation_data(original_baseline_path)
    edited_baseline = load_evaluation_data(edited_baseline_path)
    original_worstcase = load_evaluation_data(original_worstcase_path)
    edited_worstcase = load_evaluation_data(edited_worstcase_path)

    evaluation_data_dict = {
        "original_baseline": original_baseline,
        "edited_baseline": edited_baseline,
        "original_worstcase": original_worstcase,
        "edited_worstcase": edited_worstcase,
    }

    # eval fns
    def get_idx(connections, claims, tweets):
        run_tweets = tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(targets, on="target_id", how="inner")
        run_tweets = run_tweets[["query", "target"]].reset_index()
        claim_idx = [
            targets.target.to_list().index(t_claim)
            for t_claim in run_tweets.target.to_list()
        ]
        return run_tweets, claim_idx

    def avg_prec(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return (is_rel / np.arange(1, n + 1)).sum()

    def recall(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return is_rel.sum()

    def mean_avg_prec(golds, rankings, n):
        avg_precs = [avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    def mean_recall(golds, rankings, n):
        avg_precs = [recall(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    def get_negative_ranks(ranks, gold):
        return [r for r in ranks if r != gold]

    def get_negative_ranks_arr(ranks, claim_idx):
        n_ranks = [get_negative_ranks(r, g) for r, g in zip(ranks, claim_idx)]
        return np.array(n_ranks)

    # Create a results directory if not exists
    # Save the file output
    save_dir = f"{args.experiment_path}/results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    max_length = config["rerank-evaluation"].getint("max_length")
    if dataset == "clef2021-checkthat-task2a--english":
        query_prefix = defaults.TASK_2A_EN_QUERY_PREFIX
        target_prefix = defaults.TASK_2A_EN_TARGET_PREFIX

    def _tokenize(text, tokenizer, max_length, prefix=""):
        text = tokenizer.bos_token + prefix + text + tokenizer.eos_token
        token_params = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        return tokenizer(text, **token_params)

    tweet_tokenize = partial(
        _tokenize, tokenizer=tokenizer, max_length=max_length, prefix=query_prefix
    )
    claim_tokenize = partial(
        _tokenize, tokenizer=tokenizer, max_length=max_length, prefix=target_prefix
    )

    device = torch.device("cuda")
    print(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
    model.to(device)
    model.eval()
    batch_size = config["rerank-evaluation"].getint("batch_size", 1)

    def get_reranks(dataset):
        reranks = []
        with torch.no_grad():
            labels, input_ids, attention_mask, position_ids = [], [], [], []
            for i in tqdm(range(len(dataset))):
                item = dataset[i]

                labels.append(
                    torch.tensor(
                        np.tile(item["labels"][np.newaxis], (N_CANDIDATES, 1)),
                        device=device,
                    )
                )
                input_ids.append(
                    torch.tensor(np.stack(item["input_ids"], 0), device=device)
                )
                attention_mask.append(
                    torch.tensor(np.stack(item["attention_mask"], 0), device=device)
                )
                position_ids.append(
                    torch.tensor(
                        np.tile(item["position_ids"][np.newaxis], (N_CANDIDATES, 1)),
                        device=device,
                    )
                )

                if i == len(dataset) - 1 or not (i + 1) % batch_size:
                    labels = torch.cat(labels, 0)
                    input_ids = torch.cat(input_ids, 0)
                    attention_mask = torch.cat(attention_mask, 0)
                    position_ids = torch.cat(position_ids, 0)

                    inpt_dict = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "position_ids": position_ids,
                    }
                    outpt = model(**inpt_dict)

                    lm_logits = outpt.logits.to(torch.float32)
                    # Shift so that tokens < n predict n
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss.reshape((-1, N_CANDIDATES, input_ids.shape[1] - 1)).sum(
                        axis=-1
                    )
                    reranks.extend(loss.cpu().numpy().argsort().tolist())

                    labels, input_ids, attention_mask, position_ids = [], [], [], []

                else:
                    continue
        return reranks

    results_file_path = os.path.join(save_dir, "after_reranking_results.jsonl")
    # Load existing results to check for already evaluated models
    evaluated_models = set()
    if os.path.exists(results_file_path):
        with jsonlines.open(results_file_path, mode="r") as reader:
            for obj in reader:
                for model in embedding_models:
                    if model in obj:
                        evaluated_models.add(model)
    ptn = "test"
    with jsonlines.open(results_file_path, mode="a") as writer:
        for embedding_model_path in tqdm(embedding_models):
            # Check if the model has already been evaluated
            if embedding_model_path in evaluated_models:
                print(f"Skipping evaluation for model: {embedding_model_path}")
                continue

            print(f"Running evaluation on model: {embedding_model_path}")
            embedding_save_dir = os.path.join(save_dir, embedding_model_path)
            embedding_model_results = {}
            for key, test_queries in tqdm(evaluation_data_dict.items()):
                print(f"Running evaluation on {key}")
                test_ranks_path = os.path.join(
                    embedding_save_dir, f"{key}_ranks_{ptn}.npy"
                )
                test_ranks = np.load(test_ranks_path)
                map_results = {}
                map_recall_results = {}
                _, claim_idx = get_idx(test_qrels, targets, test_queries)
                k_values = list(filter(lambda k: k <= N_CANDIDATES, [1, 5, 10, 20]))
                test_dataset = dataloaders.ExtendedAutoRegressiveEvalDataset(
                    tweet_encode_fn=tweet_tokenize,
                    claim_encode_fn=claim_tokenize,
                    claims=targets,
                    tweets=test_queries,
                    connections=test_qrels,
                    ranks=test_ranks,
                    n_candidates=N_CANDIDATES,
                )
                test_reranks = get_reranks(test_dataset)

                ranks = np.array(
                    [ids[rerank] for ids, rerank in zip(test_ranks, test_reranks)]
                )

                map_results[ptn] = []
                for n in k_values:
                    map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

                map_recall_results[ptn] = []
                for n in k_values:
                    map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))

                results = {
                    "map_results": map_results,
                    "map_recall_results": map_recall_results,
                }
                embedding_model_results[key] = results
            all_embedding_model_results = {
                embedding_model_path: embedding_model_results
            }
            writer.write(all_embedding_model_results)


if __name__ == "__main__":
    run()
