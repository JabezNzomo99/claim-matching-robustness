# Code to evaluate generations of named entity replacements
import argparse
import configparser
import os
from claimrobustness import utils, defaults
import json
import pandas as pd
from transformers import pipeline


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

    verifier = utils.init_pipeline(
        model_name=config["verifier"].get("model_string"),
        model_path=config["verifier"].get("model_path"),
        num_labels=config["verifier"].getint("num_labels"),
    )

    def verify_llm_rewrite(verifier_model: pipeline):
        # Load the dataset
        dataset_dir = f"{args.experiment_path}/{dataset}"
        dataset_path = os.path.join(dataset_dir, "amplify_minimize_rewrites.csv")
        df = utils.load_verifier_data(dataset_path)

        candidate_sentences = []
        verified_list = []

        for _, row in df.iterrows():
            try:
                rewrites = json.loads(row["rewritten_claims"])
                for rewrite in rewrites["rewritten_claims"]:
                    verifier_input = rewrite + defaults.SEPARATOR_TOKEN + row["target"]
                    candidate_sentences.append(verifier_input)
                    verified_list.append(
                        {
                            "query_id": row["query_id"],
                            "original_claim": row["query"],
                            "edited_claim": rewrite,
                            "rouge_score": utils.calculate_rouge_scores(
                                row["query"], rewrite
                            ),
                            "edit_distance": utils.calculate_normalised_levenshtein_dist(
                                row["query"], rewrite
                            ),
                        }
                    )
            except Exception as e:
                continue

        verifier_dataset = utils.VerifierDataset(candidate_sentences)
        verifier_scores = list(verifier_model(verifier_dataset))
        # Convert verifier scores to JSON strings
        verifier_scores_json = [json.dumps(score) for score in verifier_scores]

        verified_df = pd.DataFrame(verified_list)
        verified_df["verifier_scores"] = verifier_scores_json

        verified_dataset_path = os.path.join(
            dataset_dir, f"verified_amplify_minimize_rewrites.csv"
        )
        verified_df.to_csv(verified_dataset_path, index=False, header=True)

        print(
            f"Saved the verified ampliff/minimize rewrites to {verified_dataset_path}"
        )

    verify_llm_rewrite(verifier)


if __name__ == "__main__":
    run()
