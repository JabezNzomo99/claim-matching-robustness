# Code to evaluate generations of named entity replacements
import argparse
import configparser
import os
from claimrobustness import utils
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

    # Load the dataset
    dataset_dir = f"{args.experiment_path}/{dataset}"
    min_replacements_path = os.path.join(
        dataset_dir, "min_named_entity_replacements.csv"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["verifier"].get("model_string"))
    model = AutoModelForSequenceClassification.from_pretrained(
        "experiments/train_verifier/debertaV3/",
        num_labels=config["verifier"].getint("num_labels"),
    )
    device = utils.get_device()
    model.to(device)
    verifier = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=device
    )

    min_replacements_df = utils.load_verifier_data(min_replacements_path)
    for index, row in min_replacements_df.iterrows():
        input_claim = row["query"]
        replacable_entities = json.loads(row["min_replacement_response"])[
            "replaceable_entities"
        ]
        new_sentences = []
        for entity in replacable_entities:
            token = entity["token"]
            replacements = entity["replacements"]

            for replacement in replacements:
                new_sentence = input_claim.replace(token, replacement)
                sentence_pred = {
                    "sentence": new_sentence,
                    "verifier_score": verifier(
                        new_sentence + " [SEP] " + row["target"]
                    ),
                }
                new_sentences.append(sentence_pred)
        min_replacements_df.loc[index, "new_sentences"] = json.dumps(new_sentences)

    min_replacements_df.to_csv(
        os.path.join(dataset_dir, "verified_min_named_entity_replacements.csv")
    )


if __name__ == "__main__":
    run()
