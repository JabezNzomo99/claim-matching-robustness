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

    def select_queries():
        # Load the jsonl file containing the verified labels
        verified_jsonl_path = os.path.join(dataset_dir, f"llm_typos_verified.jsonl")
        verified_df = pd.read_json(verified_jsonl_path, lines=True)
        original_baseline_claims = []
        rewritten_baseline_claims = []
        orig_worstcase_claims = []
        rewritten_worstcase_claims = []

        for idx, row in tqdm(verified_df.iterrows()):
            rewrites = parse_rewritten_tweets(row["rewrites"])
            verified_labels = json.loads(row["verification"])["labels"]
            # Get indices where the label is 1
            verified_idx = [
                idx for idx, label in enumerate(verified_labels) if label == 1
            ]
            # Proceed only if there are verified indices with label 1
            if verified_idx:
                if len(verified_idx) == 1:
                    selected_idx = verified_idx[0]
                    orig_json = {
                        "query_id": row["query_id"],
                        "query": row["query"],
                    }
                    rewrite_json = {
                        "query_id": row["query_id"],
                        "query": rewrites[selected_idx],
                    }
                    original_baseline_claims.append(orig_json)
                    rewritten_baseline_claims.append(rewrite_json)
                else:
                    # Select all the verified rewrites and calculate rouge score between the original and rewritten claims
                    # Assign the original and rewritten claims to the baseline or worstcase based on the rouge score
                    valid_rewrites = [rewrites[idx] for idx in verified_idx]
                    original_claim = row["query"]
                    edit_distances = [
                        utils.calculate_normalised_levenshtein_dist(
                            sentence1=original_claim, sentence2=rewrite
                        )
                        for rewrite in valid_rewrites
                    ]
                    baseline_idx = edit_distances.index(min(edit_distances))
                    worstcase_idx = edit_distances.index(max(edit_distances))
                    orig_json = {
                        "query_id": row["query_id"],
                        "query": row["query"],
                    }
                    baseline_rewrite_json = {
                        "query_id": row["query_id"],
                        "query": rewrites[baseline_idx],
                    }
                    worstcase_rewrite_json = {
                        "query_id": row["query_id"],
                        "query": rewrites[worstcase_idx],
                    }
                    original_baseline_claims.append(orig_json)
                    rewritten_baseline_claims.append(baseline_rewrite_json)
                    orig_worstcase_claims.append(orig_json)
                    rewritten_worstcase_claims.append(worstcase_rewrite_json)

        # Save the original claims
        pd.DataFrame(original_baseline_claims).to_csv(
            os.path.join(dataset_dir, f"orig_baseline_typos.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

        # Save the rewritten claims
        pd.DataFrame(rewritten_baseline_claims).to_csv(
            os.path.join(dataset_dir, f"edited_baseline_typos.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

        # Save the original worstcase claims
        pd.DataFrame(orig_worstcase_claims).to_csv(
            os.path.join(dataset_dir, f"orig_worstcase_typos.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

        # Save the rewritten worstcase claims
        pd.DataFrame(rewritten_worstcase_claims).to_csv(
            os.path.join(dataset_dir, f"edited_worstcase_typos.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

    select_queries()


if __name__ == "__main__":
    run()
