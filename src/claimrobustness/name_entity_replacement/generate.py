import os
import argparse
from groq import Groq
import utils
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
from datetime import timedelta
import stanza
import defaults
import configparser

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
    prompt_template = config["model"].get("prompt_template")

    min_replacements = config["generation"].getint("min_replaceable_entities")
    max_replacements = config["generation"].getint("max_replaceable_entities")
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
        return entities

    nlp = stanza.Pipeline(lang="en", processors="tokenize,ner")
    run_queries["tokens_to_replace"] = run_queries["query"].progress_apply(
        lambda query: parse_ner(query, nlp)
    )

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

    # Filter out queries with less min_replacement size
    min_filtered_queries = run_queries[
        run_queries["tokens_to_replace"].apply(len) >= min_replacements
    ]
    print("Shape of min filtered_queries: ", min_filtered_queries.shape)

    # Run min number of replacements
    min_filtered_queries["min_replacement_response"] = (
        min_filtered_queries.progress_apply(
            lambda row: get_ner_replacements(
                client=client,
                prompt_template=prompt_template,
                claim=row["query"],
                fact_check=row["target"],
                named_entities=row["tokens_to_replace"],
                budget=min_replacements,
            ),
            axis=1,
        )
    )

    # Save the file output
    save_dir = f"{args.experiment_path}/{dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    min_filtered_queries.to_csv(
        os.path.join(save_dir, "min_named_entity_replacements.csv"), index=False
    )

    # Filter out queries with less min_replacement size
    max_filtered_queries = run_queries[
        run_queries["tokens_to_replace"].apply(len) >= max_replacements
    ]
    print("Shape of max filtered_queries: ", max_filtered_queries.shape)
    # Run min number of replacements
    max_filtered_queries["max_replacement_response"] = (
        max_filtered_queries.progress_apply(
            lambda row: get_ner_replacements(
                client=client,
                prompt_template=prompt_template,
                claim=row["query"],
                fact_check=row["target"],
                named_entities=row["tokens_to_replace"],
                budget=max_replacements,
            ),
            axis=1,
        )
    )

    max_filtered_queries.to_csv(
        os.path.join(save_dir, "max_named_entity_replacements.csv"), index=False
    )


if __name__ == "__main__":
    run()
