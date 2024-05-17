# Takes in a file containing the scores and selects a sample with the lowest score
# Creates a file with query id and query
import argparse
import configparser
import os
from claimrobustness import utils
import json
import pandas as pd


def run():
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
    dataset_dir = f"{args.experiment_path}/{dataset}"

    def find_min_max_score_matched_label(group):
        # Filter rows where label is "LABEL_1"
        matched_rows = group[
            group["verifier_scores"].apply(
                lambda x: json.loads(x)["label"] == "LABEL_1"
            )
        ]
        if not matched_rows.empty:
            # Find the row with the least score among the filtered rows
            min_score_row = matched_rows.loc[matched_rows["edit_distance"].idxmin()]
            max_score_row = matched_rows.loc[matched_rows["edit_distance"].idxmax()]
            return pd.DataFrame(
                {
                    "query_id": group["query_id"].iloc[0],
                    "original_claim": group["query"].iloc[0],
                    "baseline_claim": min_score_row["edited_claim"],
                    "baseline_edit_distance": min_score_row["edit_distance"],
                    "worstcase_claim": max_score_row["edited_claim"],
                    "worstcase_edit_distance": max_score_row["edit_distance"],
                },
                index=[0],
            )
        else:
            return None

    # Load the dataset
    dataset_path = os.path.join(dataset_dir, "verified_llm_rewrites.csv")
    df = utils.load_verifier_data(dataset_path)
    result_df = (
        df.groupby("query_id")
        .apply(find_min_max_score_matched_label)
        .reset_index(drop=True)
    )

    # Save the original claims
    result_df[["query_id", "original_claim"]].to_csv(
        os.path.join(dataset_dir, f"orig_llm_rewrites.tsv"),
        index=False,
        header=True,
    )

    # Save the baseline claims
    result_df[["query_id", "baseline_claim"]].to_csv(
        os.path.join(dataset_dir, f"baseline_llm_rewrites.tsv"),
        index=False,
        header=["query_id", "query"],
    )

    result_df[["query_id", "worstcase_claim"]].to_csv(
        os.path.join(dataset_dir, f"worstcase_llm_rewrites.tsv"),
        index=False,
        header=["query_id", "query"],
    )


if __name__ == "__main__":
    run()
