# Takes in a file containing the scores and selects a sample with the lowest score
# Creates a file with query id and query
import argparse
import configparser
import os
from claimrobustness import utils
import json
import pandas as pd
import random
import re


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

    def parse_claims(markdown_json_string):
        """
        Parses a JSON string formatted with Markdown-style backticks and returns the list of claims.

        Args:
            markdown_json_string (str): A string containing JSON wrapped in Markdown backticks.

        Returns:
            list: A list of claims from the JSON or an empty list if no claims are found.
        """
        try:
            # Remove Markdown formatting (backticks and optional language labels)
            cleaned_json_string = re.sub(
                r"```(?:json)?\n", "", markdown_json_string.strip()
            ).strip("`")

            # Parse the cleaned JSON string
            parsed_data = json.loads(cleaned_json_string)

            # Return the list of claims
            return parsed_data.get("negated_claims", [])
        except (json.JSONDecodeError, AttributeError) as e:
            # Handle errors gracefully and return an empty list
            print(f"Error parsing JSON: {e}")
            return []

    def select_queries(budget: str):
        # Define the verified jsonl paths
        verified_jsonl_path = os.path.join(
            dataset_dir, f"{budget}_negation_verified.jsonl"
        )
        verified_df = pd.read_json(verified_jsonl_path, lines=True)
        original_claims = []
        rewritten_claims = []
        for idx, row in verified_df.iterrows():
            rewrites = parse_claims(row["rewrites"])
            verified_labels = json.loads(row["verification"])["labels"]
            # Get indices where the label is 1
            verified_idx = [
                idx for idx, label in enumerate(verified_labels) if label == 1
            ]
            # Proceed only if there are verified indices with label 1
            if verified_idx:
                # Randomly sample an index
                sampled_idx = random.choice(verified_idx)
                # print(row["rewrites"])
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
            os.path.join(dataset_dir, f"orig_{budget}_negation_queries.tsv"),
            index=False,
            header=["query_id", "query"],
            sep="\t",
        )

        # Save the rewritten claims
        pd.DataFrame(rewritten_claims).to_csv(
            os.path.join(dataset_dir, f"edited_{budget}_negation_queries.tsv"),
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
