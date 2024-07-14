import configparser
import argparse
import os
from claimrobustness import utils
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import jsonlines
import json


def run():
    # Load the config file from parameters
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("--save-embs", action="store_true")
    parser.add_argument("--save-ranks", action="store_true")
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip generating baseline edits"
    )
    parser.add_argument(
        "--no-worstcase", action="store_true", help="Skip generating worstcase edits"
    )

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    embedding_models = config["evaluation"].get("embedding_models").split(",")
    print(f"Running evaluation on the following models: {embedding_models}")

    # Load the dataset name
    dataset = config["data"].get("dataset")

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

    device = utils.get_device()
    results_file_path = os.path.join(save_dir, "before_reranking_results.jsonl")
    with jsonlines.open(results_file_path, mode="a") as writer:
        for embedding_model_path in tqdm(embedding_models):
            print(f"Running evaluation on model: {embedding_model_path}")
            embedding_save_dir = os.path.join(save_dir, embedding_model_path)
            if not os.path.exists(embedding_save_dir):
                os.makedirs(embedding_save_dir)

            model = SentenceTransformer(embedding_model_path)
            # Check if cache exists
            embs = model.encode(
                targets.target.tolist(),
                prompt="Represent the evidence for retrieval:",
                show_progress_bar=True,
                device=device,
            )
            embedding_model_results = {}
            for key, test_queries in tqdm(evaluation_data_dict.items()):
                print(f"Running evaluation on {key}")
                map_results = {}
                map_recall_results = {}
                all_tweet_embs = {}
                ptn = "test"
                run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
                tweet_embs = model.encode(
                    run_tweets["query"].to_list(),
                    prompt="Represent the fact for retrieving supporting evidence:",
                    show_progress_bar=True,
                    device=device,
                )
                all_tweet_embs[ptn] = tweet_embs
                scores = tweet_embs @ embs.T
                ranks = [score.argsort()[::-1] for score in scores]
                if args.save_ranks:
                    np.save(
                        os.path.join(embedding_save_dir, f"{key}_ranks_{ptn}.npy"),
                        np.array(ranks),
                    )
                    # np.save(
                    #     os.path.join(save_dir, f"{key}_ranks_{ptn}_negatives.npy"),
                    #     get_negative_ranks_arr(ranks, claim_idx),
                    # )

                map_results[ptn] = []
                for n in [1, 5, 10, 20]:
                    map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

                map_recall_results[ptn] = []
                for n in [1, 5, 10, 20]:
                    map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))

                if args.save_embs:
                    np.save(
                        os.path.join(embedding_save_dir, f"{key}_tweet_embs.npy"),
                        all_tweet_embs,
                    )
                    np.save(os.path.join(embedding_save_dir, f"claim_embs.npy"), embs)

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
