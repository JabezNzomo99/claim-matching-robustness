import json
import os
import pandas as pd
import defaults
import re
import torch


def get_queries(querypath):
    return pd.read_csv(
        querypath, sep="\t", names=["query_id", "query"]
    ).drop_duplicates()


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


def load_data(dataset) -> pd.DataFrame:
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
    else:
        raise ValueError(f"Dataset {dataset} not supported yet")


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
    tweet = re.sub("\s\s+", "", tweet)

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
