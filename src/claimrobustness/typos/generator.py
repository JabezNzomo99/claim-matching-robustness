import os
import argparse
from groq import Groq
from claimrobustness import utils, defaults
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from datetime import timedelta
from time import sleep
import stanza
import configparser
import pandas as pd
from openai import OpenAI
import json
import random

tqdm.pandas()


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
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

    model_name = config["model"].get("model_string")
    temparature = config["model"].getfloat("temperature")
    prompt_template = config["model"].get("prompt_template")

    baseline = config["generation"].getfloat("baseline")
    worstcase = config["generation"].getfloat("worstcase")
    min_length = config["generation"].getint("min_length")
    samples = config["generation"].getint("samples")

    # Load the test data used for generating misinformation edits
    data = utils.load_data(dataset=dataset)
    test_queries, test_qrels = data["test"]
    targets = data["targets"]

    run_queries = test_queries.merge(
        test_qrels, left_on="query_id", right_on="query_id", how="inner"
    )
    run_queries = run_queries.merge(
        targets, left_on="target_id", right_on="target_id", how="inner"
    )
    run_queries = run_queries[["query_id", "query", "target"]]
    print("Shape of run_queries: ", run_queries.shape)

    # Clean the tweet query
    print("Cleaning the tweet query")
    run_queries["query"] = run_queries["query"].progress_apply(utils.clean_tweet)

    # For each input claim, identify tokens to be misspelt depending on the min length
    def select_typo_tokens(
        input_text: str, nlp: stanza.models, proportion: int, min_length: int
    ):
        # Process both baseline and worstcase queries
        doc = nlp(input_text)
        tokens_to_misspell = [
            token.text
            for sent in doc.sentences
            for token in sent.tokens
            if len(token.text) >= min_length
        ]
        num_tokens_to_misspell = int(proportion * len(tokens_to_misspell))
        tokens_selected = random.sample(tokens_to_misspell, num_tokens_to_misspell)
        return json.dumps(tokens_selected)

    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    if not args.no_baseline:
        run_queries[f"baseline_tokens_to_misspell"] = run_queries[
            "query"
        ].progress_apply(
            lambda query: select_typo_tokens(
                input_text=query, nlp=nlp, proportion=baseline, min_length=min_length
            )
        )

    if not args.no_worstcase:
        run_queries[f"worstcase_tokens_to_misspell"] = run_queries[
            "query"
        ].progress_apply(
            lambda query: select_typo_tokens(
                input_text=query, nlp=nlp, proportion=worstcase, min_length=min_length
            )
        )

    if "gpt" in model_name:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif "llama" in model_name:
        client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
        )

    @sleep_and_retry
    @limits(calls=500, period=timedelta(seconds=60).total_seconds())
    def get_typo_replacements(
        client: Groq,
        prompt_template: str,
        claim: str,
        fact_check: str,
        tokens_to_misspell: list,
    ):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                prompt = prompt_template.format(
                    number_of_samples=samples,
                    claim=claim,
                    fact_check=fact_check,
                    tokens_to_misspell=tokens_to_misspell,
                )
                chat_completion = client.chat.completions.create(
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
                    # Enable JSON mode by setting the response format
                    response_format={"type": "json_object"},
                )

                return chat_completion.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    print(
                        f"Retrying ({retries}/{max_retries}) after BadRequestError: {e}"
                    )
                    sleep(2)  # Wait for 2 seconds before retrying
                else:
                    print(
                        f"Failed after {max_retries} retries. Last BadRequestError: {e}"
                    )
                    raise  # Raise the exception after max retries

    def process_queries(df: pd.DataFrame, type: str):
        df.loc[:, "replaced_token"] = df.progress_apply(
            lambda row: get_typo_replacements(
                client=client,
                prompt_template=prompt_template,
                claim=row["query"],
                fact_check=row["target"],
                tokens_to_misspell=json.loads(row[f"{type}_tokens_to_misspell"]),
            ),
            axis=1,
        )

        # Save the file output
        save_dir = f"{args.experiment_path}/{dataset}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df.to_csv(os.path.join(save_dir, f"{type}_typos.csv"), index=False)

        print(f"Saved {type} token typos to {save_dir}")

    if not args.no_baseline:
        process_queries(run_queries, "baseline")

    if not args.no_worstcase:
        process_queries(run_queries, "worstcase")


if __name__ == "__main__":
    run()
