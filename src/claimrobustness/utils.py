import json
import os
import pandas as pd
from claimrobustness import defaults
import re
import torch
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge import Rouge
import Levenshtein
import backoff
from openai import RateLimitError


def get_queries(querypath):
    return pd.read_csv(
        querypath, sep="\t", names=["query_id", "query"], skiprows=[0]
    ).drop_duplicates()


def get_queries_dedup(querypath):
    return pd.read_csv(
        querypath, sep="\t", names=["query_id", "query"], skiprows=[0]
    ).drop_duplicates(subset=["query"])


def get_qrels(qrelpath):
    col_names = ["query_id", "query_num", "target_id", "target_num"]
    return pd.read_csv(qrelpath, sep="\t", names=col_names).drop_duplicates()


def get_targets(targetpath, keynames):
    targetpaths = [os.path.join(targetpath, f) for f in os.listdir(targetpath)]

    def load_claim(path):
        with open(path) as f:
            return json.load(f)

    targets = [load_claim(path) for path in targetpaths]
    df = pd.DataFrame(targets)
    df.columns = keynames
    return df.drop_duplicates()


def load_data(dataset) -> dict:
    """Returns pandas data frame for the specified task
    Args
        dataset: name of the task to load
    """
    if dataset == "clef2021-checkthat-task2a--english":
        train_queries = get_queries(defaults.TASK_2A_EN_TRAIN_QUERY_PATH)
        dev_queries = get_queries(defaults.TASK_2A_EN_DEV_QUERY_PATH)
        test_queries = get_queries(defaults.TASK_2A_EN_TEST21_QUERY_PATH)

        train_qrels = get_qrels(defaults.TASK_2A_EN_TRAIN_QREL_PATH)
        dev_qrels = get_qrels(defaults.TASK_2A_EN_DEV_QREL_PATH)
        test_qrels = get_qrels(defaults.TASK_2A_EN_TEST21_QREL_PATH)

        targets = get_targets(
            defaults.TASK_2A_EN_TARGETS_PATH, defaults.TASK_2A_EN_TARGETS_KEY_NAMES
        )

        return dict(
            queries=(train_queries, dev_queries),
            qrels=(train_qrels, dev_qrels),
            targets=targets,
            test=(test_queries, test_qrels),
        )
    elif dataset == "fact-check-tweet":
        train_queries = get_queries(defaults.FC_EN_TRAIN_QUERY_PATH)
        dev_queries = get_queries(defaults.FC_EN_DEV_QUERY_PATH)
        test_queries = get_queries(defaults.FC_EN_TEST_QUERY_PATH)

        train_qrels = get_qrels(defaults.FC_EN_TRAIN_QREL_PATH)
        dev_qrels = get_qrels(defaults.FC_EN_DEV_QREL_PATH)
        test_qrels = get_qrels(defaults.FC_EN_TEST_QREL_PATH)

        targets = get_targets(defaults.FC_TARGETS_PATH, defaults.FC_TARGETS_KEY_NAMES)
        # Filter only english targets
        targets = targets[targets["lang"] == "en"]

        return dict(
            queries=(train_queries, dev_queries),
            qrels=(train_qrels, dev_qrels),
            targets=targets,
            test=(test_queries, test_qrels),
        )
    elif dataset == "ood-dataset":
        test_queries = get_queries_dedup(defaults.OOD_EN_TEST_QUERY_PATH)
        test_qrels = get_qrels(defaults.OOD_EN_TEST_QREL_PATH)

        targets = get_targets(defaults.OOD_TARGETS_PATH, defaults.OOD_TARGETS_KEY_NAMES)
        return dict(
            targets=targets,
            test=(test_queries, test_qrels),
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported yet test")


def load_verifier_data(dataset_path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def clean_tweet(tweet):
    # Replace any URL with the token 'URL'
    tweet = re.sub(r"http\S+", "", tweet).strip()

    # Remove any text after a hyphen that contains a date
    tweet = re.sub(
        r"—[^—]*\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b.*\d{4}.*$",
        "",
        tweet,
        flags=re.IGNORECASE,
    )

    # Remove any links to media files
    tweet = re.sub(r"pic.twitter.com/[\w]*", "", tweet).strip()

    # Remove any user mentions
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)

    # Remove extra white spaces
    tweet = re.sub(r"\s\s+", "", tweet)

    return tweet


def get_device() -> torch.device:
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device


def combine_features(df):
    sentences = []
    labels = []
    # For each of the samples...
    for _, row in df.iterrows():
        # Piece it together...
        claim = row["query"]
        claim = clean_tweet(claim)
        verified_claim = row["claim"]
        combined_text = claim + " [SEP] " + verified_claim
        sentences.append(combined_text)
        labels.append(row["label"])
    return sentences, labels


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def good_update_interval(total_iters, num_desired_updates):
    """
    This function will try to pick an intelligent progress update interval
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the
                              course of the for-loop.
    """
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller.
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval


def plot_training_stats(df_stats: pd.DataFrame, output_dir: str):
    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats["Training Loss"], "b-o", label="Training")
    plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure.
    plt.savefig(os.path.join(output_dir, "training_stats_plot.svg"), dpi=300)


class VerifierDataset(Dataset):

    def __init__(self, candidate_sentences):
        self.candidate_sentences = candidate_sentences

    def __len__(self):
        return len(self.candidate_sentences)

    def __getitem__(self, idx):
        return self.candidate_sentences[idx]


@backoff.on_exception(backoff.expo, RateLimitError)
async def request_llm(client, prompt, model_name, temparature):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent social media user.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model_name,
        temperature=temparature,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format --> disabled for the moment
        # Some papers have shown that json formatting can impact model generations
        # response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


def init_pipeline(
    model_name: str,
    model_path: str,
    num_labels: int,
    task: str = "text-classification",
):
    """
    Initialize the pipeline for the verifier model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
    device = get_device()
    model.to(device)
    pipe = pipeline(task=task, model=model, tokenizer=tokenizer, device=device)
    return pipe


def calculate_rouge_scores(sentence1, sentence2):
    scorer = Rouge()
    scores = scorer.get_scores(sentence1, sentence2)
    return json.dumps(
        {
            "rouge-1": scores[0]["rouge-1"]["f"],
            "rouge-2": scores[0]["rouge-2"]["f"],
            "rouge-l": scores[0]["rouge-l"]["f"],
        }
    )


def calculate_normalised_levenshtein_dist(
    sentence1: str, sentence2: str, model_name: str = "bert-base-uncased"
):
    """
    Calculate the normalized Levenshtein distance between two sentences.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens1 = tokenizer.tokenize(sentence1)
    tokens2 = tokenizer.tokenize(sentence2)

    token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

    levenshtein_dist = Levenshtein.distance(token_ids1, token_ids2)
    max_len = max(len(token_ids1), len(token_ids2))
    normalized_lev_dist = levenshtein_dist / max_len

    return normalized_lev_dist


@backoff.on_exception(backoff.expo, RateLimitError)
async def request_llm(client, prompt, model_name, temparature, enable_json=False):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent social media user.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model_name,
        temperature=temparature,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format --> disabled for the moment
        # Some papers have shown that json formatting can impact model generations
        # response_format={"type": "json_object"},
        response_format={"type": "json_object"} if enable_json else None,
    )
    return chat_completion.choices[0].message.content


def get_bm25_preprocess_fn(dataset):
    if dataset == "clef2021-checkthat-task2a--english":
        return (
            lambda targets: targets[["title", "subtitle", "target"]]
            .apply(lambda x: x[0] + " " + x[1] + " " + x[2], axis=1)
            .to_list()
        )

    elif dataset == "clef2022-checkthat-task2b--english":
        return (
            lambda targets: targets[["title", "target"]]
            .apply(lambda x: x[0] + " " + x[1], axis=1)
            .to_list()
        )

    elif dataset == "clef2022-checkthat-task2a--arabic":
        return (
            lambda targets: targets[["title", "target"]]
            .apply(lambda x: x[0] + " " + x[1], axis=1)
            .to_list()
        )

    elif dataset == "that-is-a-known-lie-snopes":
        return lambda targets: targets.target.to_list()

    elif dataset == "fact-check-tweet":
        return lambda targets: targets.target.to_list()

    elif dataset == "that-is-a-known-lie-politifact":
        pass


# Special parameters for NV-Embed
task_name_to_instruct = {
    "example": "Given a claim, retrieve fact checks that can help verify the claim",
}
nv_embed_query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "


def add_eos(input_examples, model):
    input_examples = [
        input_example + model.tokenizer.eos_token for input_example in input_examples
    ]
    return input_examples


# Dictionary to support loading of fine-tuned models
BASE_PATH = os.environ.get("CLAIM_ROBUSTNESS_BASE", "/path/to/claimrobustness")

# Dictionary to support loading of fine-tuned models
finetuned_models_path_mapping = {
    "sentence-t5-large-ft": f"{BASE_PATH}/sentence-t5-large-ft",
    "all-mpnet-base-v2-ft": f"{BASE_PATH}/all-mpnet-base-v2",
    "all-mpnet-base-v2-adapted-lite": f"{BASE_PATH}/output/original-perturbed-2025-01-28_10-27-36/final",
    "all-mpnet-basev2-adapted-full": f"{BASE_PATH}/output/original-perturbed-full-2025-01-28_10-34-06/final",
    "all-mpnet-basev2-adapted-ft": f"{BASE_PATH}/all-mpnet-base-v2-adapted",
    "all-mpnet-basev2-robust": f"{BASE_PATH}/output/original-perturbed-full-2025-01-28_11/final",
    "all-mpnet-basev2-robust-ft": f"{BASE_PATH}/all-mpnet-base-v2-adapted-full",
}
