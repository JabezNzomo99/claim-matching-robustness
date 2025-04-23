import os
import argparse
from openai import RateLimitError, AsyncOpenAI
from claimrobustness import utils
from tqdm import tqdm
import configparser
import pandas as pd
import asyncio
import backoff
import jsonlines

tqdm.pandas()

# Define the data path for the experiments

data_paths = {
    # "typos": "experiments/typos/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_typos.tsv",
    # "entity_replacement": "experiments/named_entity_replacement/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_named_entity_replacements.tsv",
    # "dialect_pidgin": "experiments/dialect/gpt4o/clef2021-checkthat-task2a--english/pidgin/orig_baseline_dialect.tsv",
    # "negation": "experiments/negation/gpt4o/clef2021-checkthat-task2a--english/orig_worstcase_negation_queries.tsv",
    "ood": "experiments/ood/OOD-EN-Queries.tsv"
}


@backoff.on_exception(backoff.expo, RateLimitError)
async def request_llm(client, prompt, model_name, temparature=0.5):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
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


async def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("dataset", type=str, help="path where config lies")

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))
    dataset = args.dataset

    model_name = config["model"].get("model_string")
    prompt_template = config["model"].get("prompt_template")

    def load_evaluation_data(path: str) -> pd.DataFrame:
        return pd.read_csv(
            path, names=["query_id", "query"], skiprows=[0], sep="\t"
        ).drop_duplicates()

    # Init AsyncOpenAI client
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def process_queries(queries: pd.DataFrame, experiment_name: str):
        df = queries.copy()

        # Save the file output
        save_dir = f"{args.experiment_path}/{dataset}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Define jsonl file to save the generated edits
        output_file = os.path.join(save_dir, f"orig_{experiment_name}_normalised.jsonl")

        # Load the existing ids
        existing_ids = set()
        with open(output_file, "a+", encoding="utf-8") as f:
            f.seek(0)
            try:
                with jsonlines.Reader(f) as reader:
                    data = list(reader.iter())
                for obj in data:
                    existing_ids.add(obj["query_id"])
            except jsonlines.jsonlines.InvalidLineError:
                # This will be raised if the file is empty. You can handle it as you need.
                pass

        filtered_df = df[~df["query_id"].isin(existing_ids)]
        print(f"Generating rewrites for {len(filtered_df)} tweets")

        with jsonlines.open(output_file, mode="a") as writer:
            for _, row in tqdm(filtered_df.iterrows()):
                prompt = prompt_template.format(
                    claim=row["query"],
                )
                llm_response = await request_llm(
                    client,
                    prompt,
                    model_name,
                )

                json_obj = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "normalised": llm_response,
                }
                writer.write(json_obj)

    for experiment_name, path in data_paths.items():
        print(f"Running experiment: {experiment_name}")
        run_queries = load_evaluation_data(path)
        run_queries = run_queries[["query_id", "query"]]
        print("Shape of run_queries: ", run_queries.shape)
        await process_queries(run_queries, experiment_name)


if __name__ == "__main__":
    asyncio.run(run())
