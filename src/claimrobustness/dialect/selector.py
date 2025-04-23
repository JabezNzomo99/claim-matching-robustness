# Takes in a file containing the scores and selects a sample with the lowest score
# Creates a file with query id and query
import argparse
import configparser
import os
from claimrobustness import utils
import json
import pandas as pd
import random
from tqdm import tqdm


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

    dialect_idx_mapping = {
        "aae": 0,
        "pidgin": 1,
        "singlish": 2,
        "patois": 3,
    }

    def select_queries():
        # Load the jsonl file containing the verified labels
        verified_jsonl_path = os.path.join(
            dataset_dir, f"dialect_rewrites_verified.jsonl"
        )
        verified_df = pd.read_json(verified_jsonl_path, lines=True)

        for dialect, idx in tqdm(dialect_idx_mapping.items()):
            print(f"Processing dialect: {dialect}")
            # Create dialect subdirectory
            dialect_dir = os.path.join(dataset_dir, dialect)
            os.makedirs(dialect_dir, exist_ok=True)
            original_claims = []
            rewritten_dialect_claims = []

            for _, row in tqdm(verified_df.iterrows()):
                try:
                    rewrites = parse_rewritten_tweets(row["rewrites"])
                    verified_labels = list(json.loads(row["verification"])["labels"])
                    # For this specific dialect, check whether the label is 1
                    if len(verified_labels) > 0 and verified_labels[idx] == 1:
                        orig_json = {
                            "query_id": row["query_id"],
                            "query": row["query"],
                        }
                        rewrite_json = {
                            "query_id": row["query_id"],
                            "query": rewrites[idx],
                        }
                        original_claims.append(orig_json)
                        rewritten_dialect_claims.append(rewrite_json)
                except Exception as e:
                    continue

            # Save the original claims
            pd.DataFrame(original_claims).to_csv(
                os.path.join(dialect_dir, f"orig_baseline_dialect.tsv"),
                index=False,
                header=["query_id", "query"],
                sep="\t",
            )

            # Save the rewritten claims
            pd.DataFrame(rewritten_dialect_claims).to_csv(
                os.path.join(dialect_dir, f"edited_baseline_dialect.tsv"),
                index=False,
                header=["query_id", "query"],
                sep="\t",
            )

    select_queries()


if __name__ == "__main__":
    run()
