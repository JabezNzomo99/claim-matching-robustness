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
import json

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

    baseline = config["generation"].getint("baseline")
    worstcase = config["generation"].getint("worstcase")
    samples = config["generation"].getint("number_of_samples")

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

    # Identify named entity tokens to replace using Stanza for NER
    def parse_ner(input_text, nlp: stanza.models, types_to_exclude=[]):
        doc = nlp(input_text)
        entities = [
            {
                "token": ent.text,
                "type": ent.type,
                "description": defaults.NAMED_ENTITIES_TYPE_DEFINITIONS[ent.type],
            }
            for sent in doc.sentences
            for ent in sent.ents
            if ent.type not in types_to_exclude
        ]
        return json.dumps(entities)

    nlp = stanza.Pipeline(lang="en", processors="tokenize,ner")
    run_queries["tokens_to_replace"] = run_queries["query"].progress_apply(
        lambda query: parse_ner(query, nlp)
    )

    # Currently supporting Llama3, need to add support for GPT models
    client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
    )

    @sleep_and_retry
    @limits(calls=30, period=timedelta(seconds=60).total_seconds())
    def get_ner_replacements(
        client: Groq,
        prompt_template: str,
        claim: str,
        fact_check: str,
        named_entities: list,
        budget: int,
    ):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                prompt = prompt_template.format(
                    number_of_samples=samples,
                    claim=claim,
                    fact_check=fact_check,
                    named_entities=named_entities[:budget],
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

    def process_queries(queries: pd.DataFrame, budget: int, type: str):
        df = queries[queries["tokens_to_replace"].apply(len) >= budget]

        df.loc[:, "replaced_token"] = df.progress_apply(
            lambda row: get_ner_replacements(
                client=client,
                prompt_template=prompt_template,
                claim=row["query"],
                fact_check=row["target"],
                named_entities=json.loads(row["tokens_to_replace"]),
                budget=budget,
            ),
            axis=1,
        )

        # Save the file output
        save_dir = f"{args.experiment_path}/{dataset}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df.to_csv(
            os.path.join(save_dir, f"{type}_named_entity_replacements.csv"), index=False
        )

        print(f"Saved {type} named entity replacements to {save_dir}")

    if not args.no_baseline:
        process_queries(run_queries, baseline, "baseline")

    if not args.no_worstcase:
        process_queries(run_queries, worstcase, "worstcase")


if __name__ == "__main__":
    run()
