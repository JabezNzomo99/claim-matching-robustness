# Code to evaluate generations of named entity replacements
import argparse
import configparser
import os
from claimrobustness import utils, defaults
import json
import pandas as pd
from transformers import pipeline
from itertools import product
from claimrobustness import utils


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

    baseline = config["generation"].getint("baseline")
    worstcase = config["generation"].getint("worstcase")

    verifier = utils.init_pipeline(
        model_name=config["verifier"].get("model_string"),
        model_path=config["verifier"].get("model_path"),
        num_labels=config["verifier"].getint("num_labels"),
    )

    def verify_replacements(process: str, verifier_model: pipeline, budget: int):
        # Load the dataset
        dataset_dir = f"{args.experiment_path}/{dataset}"
        dataset_path = os.path.join(
            dataset_dir, f"{process}_named_entity_replacements.csv"
        )
        df = utils.load_verifier_data(dataset_path)

        candidate_sentences = []
        verified_list = []

        for _, row in df.iterrows():
            try:
                input_claim = row["query"]
                entities = json.loads(row["tokens_to_replace"])
                replaceable_entities = json.loads(row["replaced_token"])[
                    "replaceable_entities"
                ]

                replacements = {}
                for entity in replaceable_entities:
                    token = entity["token"]
                    replacements[token] = entity["replacements"]

                tokens = [entity["token"] for entity in entities[:budget]]
                token_combinations = list(
                    product(*[replacements[token] for token in tokens])
                )

                for combination in token_combinations:
                    edited_claim = input_claim
                    for token, replacement in zip(tokens, combination):
                        edited_claim = edited_claim.replace(token, replacement)
                    verifier_input = (
                        edited_claim + defaults.SEPARATOR_TOKEN + row["target"]
                    )
                    candidate_sentences.append(verifier_input)
                    verified_list.append(
                        {
                            "query_id": row["query_id"],
                            "original_claim": row["query"],
                            "edited_claim": edited_claim,
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
            dataset_dir, f"verified_{process}_named_entity_replacements.csv"
        )
        verified_df.to_csv(verified_dataset_path, index=False, header=True)

        print(
            f"Saved the verified replacements for {process} to {verified_dataset_path}"
        )

    if not args.no_baseline:
        verify_replacements("baseline", verifier, baseline)

    if not args.no_worstcase:
        verify_replacements("worstcase", verifier, worstcase)


if __name__ == "__main__":
    run()
