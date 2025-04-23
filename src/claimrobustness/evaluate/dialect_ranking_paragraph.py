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
import torch
import logging


# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

# Create a logger
logger = logging.getLogger(__name__)

# Define path to dialect confifg
experiments_dict = {
    "dialect": "experiments/dialect/gpt4o/",
}

dialect_folder_mapping = {
    "african_american_english": "aae",
    "jamaican_patois": "patois",
    "pidgin": "pidgin",
    "singlish": "singlish",
}


def split_into_paragraphs(text, max_tokens=512, tokenizer=None):
    # Placeholder splitting strategy:
    # Splits by double newline, then trims.
    # You may want a smarter strategy that uses `tokenizer` to ensure <= max_tokens.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def run():
    # Load the config file from parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("--save-embs", action="store_true")
    parser.add_argument("--save-ranks", action="store_true")
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip generating baseline edits"
    )
    parser.add_argument(
        "--no-worstcase", action="store_true", help="Skip generating worstcase edits"
    )

    args = parser.parse_args()

    # Load the dataset name
    dataset = args.dataset

    for experiment, experiment_path in experiments_dict.items():
        config = configparser.ConfigParser()
        config.read(os.path.join(experiment_path, "config.ini"))

        embedding_models = config["evaluation"].get("embedding_models").split(",")
        logging.info(f"Running evaluation on the following models: {embedding_models}")

        data_directory = config["evaluation"].get("data_directory")

        # Load the original and edited dialects for each
        for dialect, dialect_folder in dialect_folder_mapping.items():
            logger.info(f"Running evaluation on dialect: {dialect}")

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

            original_baseline = load_evaluation_data(original_baseline_path)
            edited_baseline = load_evaluation_data(edited_baseline_path)

            # Print the shape of the data
            print(f"Original Baseline Shape: {original_baseline.shape}")
            print(f"Edited Baseline Shape: {edited_baseline.shape}")

            evaluation_data_dict = {
                "original_baseline": original_baseline,
                "edited_baseline": edited_baseline,
            }

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

            # Split targets into paragraphs
            paragraph_texts = []
            paragraph_to_target_map = []
            for i, full_article in enumerate(targets.target.tolist()):
                paragraphs = split_into_paragraphs(full_article)
                for p in paragraphs:
                    paragraph_texts.append(p)
                    paragraph_to_target_map.append(i)

            # Print length of paragraph_texts
            print(f"Length of Articles: {len(targets.target.tolist())}")
            print(f"Length of paragraph_texts: {len(paragraph_texts)}")

            # Create a results directory if not exists
            # Save the file output
            results_directory = f"{experiment_path}/{dataset}/results"
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            cache_dir = os.path.join(data_directory, experiment, dataset, dialect)
            # Create directory if it does not exist
            os.makedirs(cache_dir, exist_ok=True)

            all_claims_dir = os.path.join(data_directory, "embeddings", dataset)

            results_file_path = os.path.join(
                results_directory, "before_rerank_dialect_results_all.jsonl"
            )

            device = utils.get_device()

            # Load existing results to check for already evaluated dialect and model
            evaluated_dialect_model_pairs = set()

            if os.path.exists(results_file_path):
                with jsonlines.open(results_file_path, mode="r") as reader:
                    for obj in reader:
                        if "dialect" in obj:
                            dialect_retrieved = obj["dialect"]
                            if dialect_retrieved == dialect:
                                for model in embedding_models:
                                    if model in obj:
                                        evaluated_dialect_model_pairs.add(
                                            (dialect, model)
                                        )

            with jsonlines.open(results_file_path, mode="a") as writer:
                for embedding_model_path in tqdm(embedding_models):
                    if (dialect, embedding_model_path) in evaluated_dialect_model_pairs:
                        logger.info(
                            f"Skipping evaluation for dialect: {dialect} and model: {embedding_model_path}"
                        )
                        continue

                    print(f"Running evaluation on model: {embedding_model_path}")
                    logger.info(f"Running evaluation on model: {embedding_model_path}")
                    cache_embedding_dir = os.path.join(
                        all_claims_dir, embedding_model_path
                    )
                    os.makedirs(cache_embedding_dir, exist_ok=True)
                    cached_emb_path = os.path.join(
                        cache_embedding_dir, f"claim_embs.npy"
                    )
                    # Add special support for NVEmbed
                    if embedding_model_path == "nvidia/NV-Embed-v2":
                        model = SentenceTransformer(
                            embedding_model_path, trust_remote_code=True, device=device
                        )
                        model.max_seq_length = 32768
                        model.tokenizer.padding_side = "right"
                        # Add support for finetuned models
                    elif (
                        embedding_model_path
                        in utils.finetuned_models_path_mapping.keys()
                    ):
                        model = SentenceTransformer(
                            utils.finetuned_models_path_mapping[embedding_model_path],
                            # device=device,
                        )
                    else:
                        model = SentenceTransformer(embedding_model_path)

                    # pool = model.start_multi_process_pool()

                    # Check if embeddings are already cached
                    if os.path.exists(cached_emb_path):
                        logger.info(f"Loading embeddings from {cached_emb_path}")
                        paragraph_embs = np.load(cached_emb_path)
                    else:
                        logger.info(
                            f"Embeddings not found in cache. Generating embeddings"
                        )
                        paragraph_embs = model.encode_multi_process(
                            (
                                utils.add_eos(paragraph_texts, model=model)
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else paragraph_texts
                            ),
                            prompt=(
                                None
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else "Represent the evidence for retrieval:"
                            ),
                            # device=device,
                            # pool=pool,
                            show_progress_bar=True,
                            normalize_embeddings=(
                                True
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else False
                            ),
                            batch_size=(
                                4
                                if embedding_model_path
                                in [
                                    "nvidia/NV-Embed-v2",
                                    "Salesforce/SFR-Embedding-Mistral",
                                ]
                                else 32
                            ),
                        )

                    embedding_model_results = {}
                    for key, test_queries in tqdm(evaluation_data_dict.items()):
                        print(f"Running evaluation on {key}")
                        map_results = {}
                        map_recall_results = {}
                        all_tweet_embs = {}
                        ptn = "test"
                        run_tweets, claim_idx = get_idx(
                            test_qrels, targets, test_queries
                        )

                        tweet_embs = model.encode(
                            (
                                utils.add_eos(
                                    run_tweets["query"].to_list(), model=model
                                )
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else run_tweets["query"].tolist()
                            ),
                            prompt=(
                                utils.nv_embed_query_prefix
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else "Represent the tweet for retrieving supporting evidence:"
                            ),
                            # pool=pool,
                            device=device,
                            show_progress_bar=True,
                        )
                        all_tweet_embs[ptn] = tweet_embs

                        # Compute similarities (tweet vs paragraph)
                        scores = tweet_embs @ paragraph_embs.T
                        paragraph_ranks = [score.argsort()[::-1] for score in scores]

                        # Convert paragraph-level ranks to target-level ranks
                        target_ranks_list = []
                        for ranks_for_query in paragraph_ranks:
                            # For each target, find the best (lowest number) paragraph rank
                            target_best_rank = {}
                            for rank_pos, p_idx in enumerate(ranks_for_query):
                                t_idx = paragraph_to_target_map[p_idx]
                                # Keep the best (lowest) rank for each target
                                if t_idx not in target_best_rank:
                                    target_best_rank[t_idx] = rank_pos
                            # Now we have a dict: target -> best paragraph rank
                            # Sort targets by their best paragraph rank
                            sorted_targets = sorted(
                                target_best_rank.keys(),
                                key=lambda x: target_best_rank[x],
                            )
                            target_ranks_list.append(sorted_targets)

                        if args.save_ranks:
                            embedding_save_dir = os.path.join(
                                cache_dir, embedding_model_path
                            )
                            # Create directory if it does not exist
                            if not os.path.exists(embedding_save_dir):
                                os.makedirs(embedding_save_dir)
                            np.save(
                                os.path.join(
                                    embedding_save_dir, f"{key}_ranks_{ptn}.npy"
                                ),
                                np.array(target_ranks_list),
                            )
                            # Log info that we have saved the ranks
                            logger.info(
                                f"Saved ranks for {embedding_model_path} and {key} to {embedding_save_dir}"
                            )

                        # Now use the original evaluation metrics on target_ranks_list
                        map_results[ptn] = []
                        for n in [1, 5, 10, 20]:
                            map_results[ptn].append(
                                mean_avg_prec(claim_idx, target_ranks_list, n)
                            )

                        map_recall_results[ptn] = []
                        for n in [1, 5, 10, 20]:
                            map_recall_results[ptn].append(
                                mean_recall(claim_idx, target_ranks_list, n)
                            )

                        if args.save_embs:
                            np.save(
                                os.path.join(
                                    embedding_save_dir, f"{key}_tweet_embs.npy"
                                ),
                                all_tweet_embs,
                            )
                            np.save(
                                os.path.join(embedding_save_dir, f"paragraph_embs.npy"),
                                paragraph_embs,
                            )
                            np.save(
                                os.path.join(
                                    embedding_save_dir, f"paragraph_to_target_map.npy"
                                ),
                                paragraph_to_target_map,
                            )

                        if args.save_embs:
                            # np.save(
                            #     os.path.join(embedding_save_dir, f"{key}_tweet_embs.npy"),
                            #     all_tweet_embs,
                            # )
                            np.save(cached_emb_path, paragraph_embs)

                        results = {
                            "map_results": map_results,
                            "map_recall_results": map_recall_results,
                        }
                        embedding_model_results[key] = results

                    all_embedding_model_results = {
                        "dialect": dialect,
                        embedding_model_path: embedding_model_results,
                    }
                    writer.write(all_embedding_model_results)
                    # model.stop_multi_process_pool(pool)
                    del model
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
