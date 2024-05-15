from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from claimrobustness import utils


def run():
    verifier = pipeline(
        model="zero-shot-classification",
        tokenizer="facebook/bart-large-mnli",
    )
    print("Test")


if __name__ == "__main__":
    run()
