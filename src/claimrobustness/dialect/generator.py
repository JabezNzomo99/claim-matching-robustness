import os
import argparse
from groq import Groq
from openai import OpenAI
from claimrobustness import utils
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from datetime import timedelta
from time import sleep
import configparser
import pandas as pd

tqdm.pandas()


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    dataset = config["data"].get("dataset")

    model_name = config["model"].get("model_string")
    temparature = config["model"].getfloat("temperature")

    samples = config["generation"].getint("number_of_samples")
    prompt_template = config["generation"].get("prompt_template")

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

    # Currently supporting Llama3, need to add support for GPT models
    if "gpt" in model_name:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif "llama" in model_name:
        client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
        )

    @sleep_and_retry
    @limits(calls=500, period=timedelta(seconds=60).total_seconds())
    def dialect_rewrite_claim(
        client, prompt_template: str, claim: str, fact_check: str
    ):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                prompt = prompt_template.format(
                    number_of_samples=samples,
                    claim=claim,
                    fact_check=fact_check,
                )
                print(prompt)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a highly intelligent social media user.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model=model_name,
                    temperature=temparature,
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

    run_queries.loc[:, "rewritten_claims"] = run_queries.progress_apply(
        lambda row: dialect_rewrite_claim(
            client=client,
            prompt_template=prompt_template,
            claim=row["query"],
            fact_check=row["target"],
        ),
        axis=1,
    )

    # Save the file output
    save_dir = f"{args.experiment_path}/{dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_queries.to_csv(os.path.join(save_dir, "dialect_rewrites.csv"), index=False)
    print(f"Saved llm dialect rewritten claims to {save_dir}")


if __name__ == "__main__":
    run()
