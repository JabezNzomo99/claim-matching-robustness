"""
Create a dataset for the verifier model
Retrieve dataset
Create positives
Create hard negatives using closest sentence embedding prediction apart from gold claim title
Save output dataset
"""

import argparse
from claimrobustness import utils
import configparser
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def run():
    # Parse the arguments needed
    parser = argparse.ArgumentParser(
        description="Create a dataset for the verifier model"
    )
    parser.add_argument(
        "experiment_path",
        help="Path where config lies",
        type=str,
    )

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    dataset = config["data"].get("dataset")
    model_name = config["model"].get("model_name")

    # Load the dataset
    data = utils.load_data(dataset=dataset)
    train_queries, dev_queries = data["queries"]
    train_qrels, dev_qrels = data["qrels"]
    test_queries, test_qrels = data["test"]
    targets = data["targets"]

    # print("Shape of train_queries: ", train_queries.shape)
    # print("Shape of dev_queries: ", dev_queries.shape)
    # print("Shape of train_qrels: ", train_qrels.shape)
    # print("Shape of dev_qrels: ", dev_qrels.shape)

    # Run
    device = utils.get_device()
    # Get all the embeddings of all claims
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        targets["target"].tolist(),
        show_progress_bar=True,
        device=device,
    )

    def get_idx(connections: pd.DataFrame, targets: pd.DataFrame, tweets: pd.DataFrame):
        run_tweets = tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(targets, on="target_id", how="inner")
        run_tweets = run_tweets[["query_id", "query", "target"]].reset_index()
        claim_idx = [
            targets.target.to_list().index(t_claim)
            for t_claim in run_tweets.target.to_list()
        ]
        return run_tweets, claim_idx

    def get_negative_ranks(ranks, gold):
        return [r for r in ranks if r != gold]

    def get_negative_ranks_arr(ranks, claim_idx):
        n_ranks = [get_negative_ranks(r, g) for r, g in zip(ranks, claim_idx)]
        return np.array(n_ranks)

    partitions = ["train", "dev", "test"]
    for ptn in partitions:
        if ptn == "train":
            run_tweets, claim_idx = get_idx(train_qrels, targets, train_queries)
            print(f"Train: {run_tweets.shape}")
        elif ptn == "dev":
            run_tweets, claim_idx = get_idx(dev_qrels, targets, dev_queries)
            print(f"dev: {run_tweets.shape}")
        elif ptn == "test":
            run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
            print(f"test: {run_tweets.shape}")

        print(run_tweets.shape)
        tweet_embeds = model.encode(
            run_tweets["query"].tolist(), show_progress_bar=True, device=device
        )
        scores = tweet_embeds @ embs.T
        ranks = [scores.argsort()[::-1] for scores in scores]
        negative_ranks = get_negative_ranks_arr(ranks, claim_idx)

        # Get only the top 1 for each get negative ranks and use it as hard negative
        top_negative_rank_ids = [doc_rank[0] for doc_rank in negative_ranks]
        claim_titles = [
            targets["title"].to_list()[rank_id] for rank_id in top_negative_rank_ids
        ]
        run_tweets["vclaim_negative"] = claim_titles

        # Create an empty list to store the training dataset
        tmp_data = []

        # Iterate over each row in the DataFrame
        for _, row in run_tweets.iterrows():
            query = row["query"]
            target = row["target"]
            vclaim_negative = row["vclaim_negative"]

            # Create a positive pair
            positive_pair = {"query": query, "claim": target, "label": 1}
            tmp_data.append(positive_pair)

            # Create a negative pair
            negative_pair = {"query": query, "claim": vclaim_negative, "label": 0}
            tmp_data.append(negative_pair)

        # Create a new DataFrame from the training dataset
        tmp_data = pd.DataFrame(tmp_data)

        # Save the output
        tmp_data.to_csv(
            os.path.join(args.experiment_path, f"{ptn}_verifier_dataset.csv"),
            index=False,
        )


if __name__ == "__main__":
    run()
