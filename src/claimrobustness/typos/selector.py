# Takes in a file containing the scores and selects a sample with the lowest score
# Creates a file with query id and query
import argparse
import configparser
import os
from claimrobustness import utils
import json


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
    dataset_dir = f"{args.experiment_path}/{dataset}"

    def find_least_score_matched_label(group):
        # Filter rows where label is "LABEL_1"
        matched_rows = group[
            group["verifier_scores"].apply(
                lambda x: json.loads(x)["label"] == "LABEL_1"
            )
        ]
        if not matched_rows.empty:
            # Find the row with the least score among the filtered rows
            min_score_row = matched_rows.loc[
                matched_rows["verifier_scores"]
                .apply(lambda x: json.loads(x)["score"])
                .idxmin()
            ]
            return min_score_row
        else:
            return None

    def select_queries(process: str):
        # Load the dataset
        dataset_path = os.path.join(dataset_dir, f"verified_{process}_typos.csv")
        df = utils.load_verifier_data(dataset_path)
        result_df = (
            df.groupby("query_id")
            .apply(find_least_score_matched_label)
            .reset_index(drop=True)
        )
        result_df[["query_id", "original_claim"]].to_csv(
            os.path.join(dataset_dir, f"orig_{process}_ner_queries.tsv"),
            index=False,
            header=True,
        )

        result_df[["query_id", "edited_claim"]].to_csv(
            os.path.join(dataset_dir, f"edited_{process}_ner_queries.tsv"),
            index=False,
            header=["query_id", "query"],
        )

    if not args.no_baseline:
        select_queries("baseline")

    if not args.no_worstcase:
        select_queries("worstcase")


if __name__ == "__main__":
    run()
