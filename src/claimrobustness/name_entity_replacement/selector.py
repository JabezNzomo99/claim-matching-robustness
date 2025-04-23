# Takes in a file containing the scores and selects a sample with the lowest score
# Creates a file with query id and query
import argparse
import configparser
import os
from claimrobustness import utils
import json
import pandas as pd
import random


def run():
    parser = argparse.ArgumentParser(
        description="Script to select instances from the LLM that have been modified"
    )
    parser.add_argument(
        "experiment_path",
        help="Path where config lies",
        type=str,
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset",
        type=str,
    )
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
    dataset = args.dataset
    dataset_dir = f"{args.experiment_path}/{dataset}"

    def parse_rewritten_tweets(text):
        """
        Parses a given string of rewritten tweets into a list of individual tweets.

        Args:
            text (str): The input string containing rewritten tweets.

        Returns:
            list: A list of individual rewritten tweets.
        """
        # Split the text by lines and filter out any empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Extract tweets after the colon ": " in lines that start with "Rewritten Tweet"
        tweets = [
            line.split(": ", 1)[1]
            for line in lines
            if line.startswith("Rewritten Tweet") and ": " in line
        ]

        return tweets

    def select_queries(budget: str):
        # Define the verified jsonl paths
        verified_jsonl_path = os.path.join(
            dataset_dir, f"{budget}_named_entity_replacements_verified.jsonl"
        )
        verified_df = pd.read_json(verified_jsonl_path, lines=True)
        original_claims = []
        rewritten_claims = []
        for idx, row in verified_df.iterrows():
            rewrites = parse_rewritten_tweets(row["rewrites"])
            verified_labels = json.loads(row["verification"])["labels"]
            # Get indices where the label is 1
            verified_idx = [
                idx for idx, label in enumerate(verified_labels) if label == 1
            ]
            # Proceed only if there are verified indices with label 1
            if verified_idx:
                # Randomly sample an index
                sampled_idx = random.choice(verified_idx)
                selected_rewrite = rewrites[sampled_idx]
                # Add the original claim and rewritten claim to their respective lists
                orig_json = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                }
                rewrite_json = {
                    "query_id": row["query_id"],
                    "query": selected_rewrite,
                }
                original_claims.append(orig_json)
                rewritten_claims.append(rewrite_json)

        # Save the original claims
        pd.DataFrame(original_claims).to_csv(
            os.path.join(dataset_dir, f"orig_{budget}_named_entity_replacements.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

        # Save the rewritten claims
        pd.DataFrame(rewritten_claims).to_csv(
            os.path.join(dataset_dir, f"edited_{budget}_named_entity_replacements.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

    if not args.no_baseline:
        select_queries("baseline")

    if not args.no_worstcase:
        select_queries("worstcase")


if __name__ == "__main__":
    run()
