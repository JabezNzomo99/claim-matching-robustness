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
import logging
import pickle


# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

# Create a logger
logger = logging.getLogger(__name__)

experiments_dict = {"dialect": "experiments/dialect/gpt4o/"}

dialect_folder_mapping = {
    "african_american_english": "aae",
    "jamaican_patois": "patois",
    "pidgin": "pidgin",
    "singlish": "singlish",
}


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path where config lies")
    parser.add_argument("--save-ranks", action="store_true")

    # Parse the arguments
    # Parse the arguments
    args = parser.parse_args()

    # Load the dataset name
    dataset = args.dataset

    for experiment, experiment_path in experiments_dict.items():
        logger.info(f"Running BM25 evaluation on {experiment} on {dataset} dataset")
        config = configparser.ConfigParser()
        config.read(os.path.join(experiment_path, "config.ini"))

        data_directory = config["evaluation"].get("data_directory")

        # Load the original and edited dialects for each
        for dialect, dialect_folder in dialect_folder_mapping.items():
            logger.info(f"Running evaluation on dialect: {dialect}")

            cache_dir = os.path.join(data_directory, experiment, dataset)
            # Create directory if it does not exists
            os.makedirs(cache_dir, exist_ok=True)

            # Load the original baseline and edited baseline
            original_baseline_path = os.path.join(
                experiment_path,
                dataset,
                dialect_folder,
                "orig_baseline_dialect.tsv",
            )
            edited_baseline_path = os.path.join(
                experiment_path,
                dataset,
                dialect_folder,
                "edited_baseline_dialect.tsv",
            )

            def load_evaluation_data(path: str) -> pd.DataFrame:
                return pd.read_csv(
                    path, names=["query_id", "query"], skiprows=[0], sep="\t"
                ).drop_duplicates()

            # Load the datasets
            data = utils.load_data(dataset=dataset)
            targets = data["targets"]
            _, test_qrels = data["test"]

            logger.info("Loading the evaluation data")
            original_baseline = load_evaluation_data(original_baseline_path)
            edited_baseline = load_evaluation_data(edited_baseline_path)

            print("original_baseline", original_baseline.shape)
            print("edited_baseline", edited_baseline.shape)

            evaluation_data_dict = {
                "original_baseline": original_baseline,
                "edited_baseline": edited_baseline,
            }

            sw_nltk = stopwords.words("english")
            porter = PorterStemmer()

            def preprocess(sentence):
                sentence = sentence.lower()
                sentence = re.sub(r"[^\w\s]", "", sentence)
                return [
                    porter.stem(word)
                    for word in sentence.split()
                    if word not in sw_nltk
                ]

            # BM25 cache path
            bm25_cache_dir = os.path.join(
                data_directory, "bm25", dataset
            )  # Directory for caching
            bm25_cache_path = os.path.join(bm25_cache_dir, "bm25_model.pkl")

            # Ensure the cache directory exists
            os.makedirs(bm25_cache_dir, exist_ok=True)

            cache_dir = os.path.join(data_directory, experiment, dataset, dialect)
            # Create directory if it does not exist
            os.makedirs(cache_dir, exist_ok=True)

            # Check for cached BM25 model
            if os.path.exists(bm25_cache_path):
                print("Loading cached BM25 model...")
                with open(bm25_cache_path, "rb") as cache_file:
                    bm25 = pickle.load(cache_file)
            else:
                print("Initializing and caching BM25 model...")
                corpus = utils.get_bm25_preprocess_fn(dataset)(targets)
                tokenized_corpus = [preprocess(doc) for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)

                # Cache the BM25 model
                with open(bm25_cache_path, "wb") as cache_file:
                    pickle.dump(bm25, cache_file)

            evaluation_data_dict = {
                "original_baseline": original_baseline,
                "edited_baseline": edited_baseline,
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
                    list(doc_rank).index(idx)
                    for idx, doc_rank in zip(target_idx, doc_ranks)
                ]

                return doc_scores, doc_ranks, target_idx, label_ranks

            def avg_prec(gold, rankings, n):
                is_rel = (np.array(rankings)[:n] == gold).astype(float)
                return (is_rel / np.arange(1, n + 1)).sum()

            def recall(gold, rankings, n):
                is_rel = (np.array(rankings)[:n] == gold).astype(float)
                return is_rel.sum()

            def mean_avg_prec(golds, rankings, n):
                avg_precs = [
                    avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)
                ]
                return np.array(avg_precs).mean()

            def mean_recall(golds, rankings, n):
                avg_precs = [
                    recall(gold, rlist, n) for gold, rlist in zip(golds, rankings)
                ]
                return np.array(avg_precs).mean()

            def get_negative_ranks(ranks, gold):
                return [r for r in ranks if r != gold]

            def get_negative_ranks_arr(ranks, claim_idx):
                n_ranks = [get_negative_ranks(r, g) for r, g in zip(ranks, claim_idx)]
                return np.array(n_ranks)

            # Create a results directory if not exists
            # Save the file output
            save_dir = f"{experiment_path}/{dataset}/results"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            results_file_path = os.path.join(
                save_dir, "before_reranking_dialect_results_all.jsonl"
            )

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
                    for n in [1, 5, 10, 20, 50, 100]:
                        map_results[ptn].append(
                            {f"map@{n}": mean_avg_prec(target_idx, doc_ranks, n)}
                        )

                    map_recall_results[ptn] = []
                    for n in [1, 5, 10, 20, 50, 100]:
                        map_recall_results[ptn].append(
                            {f"recall@{n}": mean_recall(target_idx, doc_ranks, n)}
                        )

                    if args.save_ranks:
                        # Create bm25 directory if not exists
                        bm25_dir = os.path.join(cache_dir, "bm25")
                        os.makedirs(bm25_dir, exist_ok=True)
                        np.save(
                            os.path.join(bm25_dir, f"{key}_ranks_{ptn}.npy"),
                            doc_ranks,
                        )

                    results = {
                        "map_results": map_results,
                        "map_recall_results": map_recall_results,
                    }
                    bm25_results[key] = results
                writer.write({"dialect": dialect, "bm25": bm25_results})


if __name__ == "__main__":
    run()
