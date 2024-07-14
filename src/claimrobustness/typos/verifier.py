# Code to evaluate generations of typos
# Typos may not need to be passed through a verifier
import argparse
import configparser
import os
from claimrobustness import utils, defaults
import json
import pandas as pd
from itertools import product
import random


def run():
    parser = argparse.ArgumentParser(
        description="Create a dataset for the verifier model"
    )
    parser.add_argument(
        "experiment_path",
        help="Path where config lies",
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
    dataset = config["data"].get("dataset")

    def select_replacements(process: str):
        # Load the dataset
        dataset_dir = f"{args.experiment_path}/{dataset}"
        dataset_path = os.path.join(dataset_dir, f"{process}_typos.csv")
        df = utils.load_verifier_data(dataset_path)

        verified_list = []

        for _, row in df.iterrows():
            try:
                input_claim = row["query"]
                replaceable_entities = json.loads(row["replaced_token"])["typo_tokens"]
                replacement_dict = {}
                for entity in replaceable_entities:
                    token_to_misspell = entity["token"]
                    replacement_typo = random.choice(entity["typos"])
                    replacement_dict[token_to_misspell] = replacement_typo

                # Replace tokens in the input_claim
                edited_claim = input_claim
                for token, typo in replacement_dict.items():
                    edited_claim = edited_claim.replace(token, typo)
                verified_list.append(
                    {
                        "query_id": row["query_id"],
                        "original_claim": row["query"],
                        "edited_claim": edited_claim,
                    }
                )
            except Exception as e:
                continue

        verified_df = pd.DataFrame(verified_list)
        verified_dataset_path = os.path.join(
            dataset_dir, f"edited_{process}_typos_queries.tsv"
        )
        orig_dataset_path = os.path.join(
            dataset_dir, f"orig_{process}_typos_queries.tsv"
        )

        verified_df[["query_id", "edited_claim"]].to_csv(
            verified_dataset_path, index=False, header=True
        )
        verified_df[["query_id", "original_claim"]].to_csv(
            orig_dataset_path, index=False, header=True
        )

        print(f"Saved the verified typos for {process} to {verified_dataset_path}")

    if not args.no_baseline:
        select_replacements("baseline")

    if not args.no_worstcase:
        select_replacements("worstcase")


if __name__ == "__main__":
    run()
