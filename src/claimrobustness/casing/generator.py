# Implement casing changes
# Baseline: Truecasing
# Worstcase: Uppercasing
import argparse
import configparser
import os
from claimrobustness import utils
import truecase
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("dataset", type=str, help="path where config lies")
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
    baseline = config["generation"].get("baseline")
    worstcase = config["generation"].get("worstcase")
    # Load the test data used for generating misinformation edits
    data = utils.load_data(dataset=dataset)
    test_queries, _ = data["test"]
    test_queries["query"] = test_queries["query"].progress_apply(utils.clean_tweet)

    # Save the file output
    save_dir = f"{args.experiment_path}/{dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not args.no_baseline:
        if baseline == "truecase":
            test_queries["baseline_query"] = test_queries["query"].map(
                truecase.get_true_case
            )
            test_queries[["query_id", "query"]].to_csv(
                os.path.join(save_dir, f"orig_baseline_casing_queries.tsv"),
                index=False,
                header=True,
                sep="\t",
            )
            test_queries[["query_id", "baseline_query"]].to_csv(
                os.path.join(save_dir, f"edited_baseline_casing_queries.tsv"),
                index=False,
                header=["query_id", "query"],
                sep="\t",
            )
        # Save output
        else:
            raise ValueError(f"Baseline {baseline} not supported")

    if not args.no_worstcase:
        if worstcase == "uppercase":
            test_queries["worstcase_query"] = test_queries["query"].str.upper()
            test_queries[["query_id", "query"]].to_csv(
                os.path.join(save_dir, f"orig_worstcase_casing_queries.tsv"),
                index=False,
                header=True,
                sep="\t",
            )
            test_queries[["query_id", "worstcase_query"]].to_csv(
                os.path.join(save_dir, f"edited_worstcase_casing_queries.tsv"),
                index=False,
                header=["query_id", "query"],
                sep="\t",
            )
        else:
            raise ValueError(f"Baseline {baseline} not supported")


if __name__ == "__main__":
    run()
