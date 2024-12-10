import configparser
import argparse
import os
from claimrobustness import utils
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
import jsonlines
from tqdm import tqdm


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("--save-ranks", action="store_true")

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

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

    sw_nltk = stopwords.words("english")
    porter = PorterStemmer()

    def preprocess(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^\w\s]", "", sentence)
        return [porter.stem(word) for word in sentence.split() if word not in sw_nltk]

    corpus = utils.get_bm25_preprocess_fn(dataset)(targets)
    tokenized_corpus = [preprocess(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    evaluation_data_dict = {
        "original_baseline": original_baseline,
        "edited_baseline": edited_baseline,
        "original_worstcase": original_worstcase,
        "edited_worstcase": edited_worstcase,
    }

    def get_bm25_rankings(qrels, targets, queries):
        run_queries = queries.merge(
            qrels, left_on="query_id", right_on="query_id", how="inner"
        )
        run_queries = run_queries.merge(
            targets, left_on="target_id", right_on="target_id", how="inner"
        )
        run_queries = run_queries[["query", "target"]]
        target_idx = [
            targets.target.to_list().index(t_claim)
            for t_claim in run_queries.target.to_list()
        ]

        queries = run_queries["query"].to_list()
        tokenized_queries = [preprocess(query) for query in queries]
        doc_scores = [bm25.get_scores(query) for query in tokenized_queries]
        doc_ranks = [score.argsort()[::-1] for score in doc_scores]
        label_ranks = [
            list(doc_rank).index(idx) for idx, doc_rank in zip(target_idx, doc_ranks)
        ]

        return doc_scores, doc_ranks, target_idx, label_ranks

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

    results_file_path = os.path.join(save_dir, "before_reranking_results.jsonl")
    save_dir = os.path.join(save_dir, "bm25")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with jsonlines.open(results_file_path, mode="a") as writer:
        bm25_results = {}
        for key, test_queries in tqdm(evaluation_data_dict.items()):
            print(f"Running evaluation on {key}")
            map_results = {}
            map_recall_results = {}
            ptn = "test"
            doc_scores, doc_ranks, target_idx, label_ranks = get_bm25_rankings(
                test_qrels, targets, test_queries
            )
            map_results[ptn] = []
            for n in [1, 5, 10, 20]:
                map_results[ptn].append(mean_avg_prec(target_idx, doc_ranks, n))

            map_recall_results[ptn] = []
            for n in [1, 5, 10, 20]:
                map_recall_results[ptn].append(mean_recall(target_idx, doc_ranks, n))

            if args.save_ranks:
                np.save(
                    os.path.join(save_dir, f"{key}_ranks_{ptn}.npy"),
                    doc_ranks,
                )

            results = {
                "map_results": map_results,
                "map_recall_results": map_recall_results,
            }
            bm25_results[key] = results
        writer.write({"bm25": bm25_results})


if __name__ == "__main__":
    run()
