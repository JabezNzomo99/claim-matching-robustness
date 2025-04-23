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


async def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("dataset", type=str, help="path where config lies")
    # Add argument to determine the split to generate the perturbations on
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default="test",
        help="split to generate the perturbations on",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip generating baseline edits"
    )
    parser.add_argument(
        "--no-worstcase", action="store_true", help="Skip generating worstcase edits"
    )

    # Parse the arguments
    args = parser.parse_args()
    splits = args.splits
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    model_name = config["model"].get("model_string")
    temparature = config["model"].getfloat("temperature")
    prompt_template = config["model"].get("prompt_template")

    dataset = args.dataset
    # Load the test data used for generating misinformation edits
    data = utils.load_data(dataset=args.dataset)

    for split in splits:
        if split == "test":
            queries, qrels = data["test"]
        elif split == "train":
            queries = data["queries"][0]
            qrels = data["qrels"][0]
        elif split == "dev":
            queries = data["queries"][1]
            qrels = data["qrels"][1]
        else:
            raise ValueError("Invalid split")

        targets = data["targets"]
        run_queries = queries.merge(
            qrels, left_on="query_id", right_on="query_id", how="inner"
        )
        run_queries = run_queries.merge(
            targets, left_on="target_id", right_on="target_id", how="inner"
        )
        run_queries = run_queries[["query_id", "query", "target"]]
        print("Shape of run_queries: ", run_queries.shape)

        # Clean the tweet query
        print("Cleaning the tweet query")
        run_queries["query"] = run_queries["query"].progress_apply(utils.clean_tweet)

        # Init AsyncOpenAI client
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        async def process_queries(queries: pd.DataFrame):
            df = queries.copy()

            # Save the file output
            save_dir = f"{args.experiment_path}/{dataset}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Define jsonl file to save the generated edits
            output_file = os.path.join(save_dir, f"{split}_llm_typos.jsonl")

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
                        fact_check=row["target"],
                    )
                    llm_response = await request_llm(
                        client, prompt, model_name, temparature
                    )

                    json_obj = {
                        "query_id": row["query_id"],
                        "query": row["query"],
                        "target": row["target"],
                        "rewrites": llm_response,
                    }
                    writer.write(json_obj)

        await process_queries(run_queries)


if __name__ == "__main__":
    asyncio.run(run())
