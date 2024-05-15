from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from claimrobustness import utils
import argparse
import configparser
import os


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
    model_string = config["model"].get("model_string")
    tokenizer = AutoTokenizer.from_pretrained(model_string)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.experiment_path, num_labels=config["training"].getint("num_labels")
    )
    device = utils.get_device()
    model.to(device)
    verifier = pipeline(
        task="text-classification", model=model, tokenizer=tokenizer, device=device
    )

    pred = verifier("Trump is the winner [SEP] The US Elections have not happened yet.")
    print(pred)


if __name__ == "__main__":
    run()
