import os
import argparse
from openai import AsyncOpenAI
from claimrobustness import utils
from tqdm import tqdm
import configparser
import pandas as pd
import asyncio
import jsonlines

tqdm.pandas()


async def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    parser.add_argument("dataset", type=str, help="path where config lies")
    parser.add_argument(
        "--split",
        type=str,
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
    config = configparser.ConfigParser()
    print("We are here...")
    config.read(os.path.join(args.experiment_path, "config.ini"))

    dataset = args.dataset
    split = args.split

    print("We are here again...")
    model_name = config["model"].get("model_string")
    temparature = config["model"].getfloat("temperature")
    verification_prompt_template = config["model"].get("verification_prompt_template")

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Create file containing the
    # Read jsonl file containing the rewrites
    # For each rewrite, generate the verification prompt

    async def verify_rewrites(budget: str):
        if split == "test":
            rewrites_file_path = f"{args.experiment_path}/{dataset}/{budget}_named_entity_replacements.jsonl"
            output_file = f"{args.experiment_path}/{dataset}/{budget}_named_entity_replacements_verified.jsonl"
        else:
            rewrites_file_path = f"{args.experiment_path}/{dataset}/{split}_{budget}_named_entity_replacements_fixed.jsonl"
            output_file = f"{args.experiment_path}/{dataset}/{split}_{budget}_named_entity_replacements_verified.jsonl"

        print("The problem is here . . .")
        try:
            with open(rewrites_file_path, "r") as f:
                for i, line in enumerate(f, 1):  # Start counting lines from 1
                    try:
                        pd.read_json(line, lines=False)  # Try to parse each line
                    except ValueError as ve:
                        print(f"Error in line {i}: {line.strip()}")
                        print("Error details:", ve)
                        raise  # Re-raise the exception if needed
            rewrites_df = pd.read_json(
                rewrites_file_path, lines=True
            )  # Proceed if no errors
        except Exception as e:
            print("Error while loading the JSON file:", e)

        rewrites_df = pd.read_json(rewrites_file_path, lines=True)
        # Load the existing verification file
        # Load the existing ids
        print("The problem is here . . .")

        existing_ids = set()
        with open(output_file, "a+", encoding="utf-8") as f:
            f.seek(0)
            try:
                with jsonlines.Reader(f) as reader:
                    data = list(reader.iter())
                for obj in data:
                    existing_ids.add(obj["query_id"])
            except jsonlines.jsonlines.InvalidLineError:
                print("WE ARE HERE :InvalidLineError")
                # This will be raised if the file is empty. You can handle it as you need.
                pass

        filtered_df = rewrites_df[~rewrites_df["query_id"].isin(existing_ids)]
        print(f"Generating verifications for {len(filtered_df)} tweets")

        with jsonlines.open(output_file, mode="a") as writer:
            for _, row in tqdm(filtered_df.iterrows()):
                prompt = verification_prompt_template.format(
                    claim=row["query"],
                    fact_check=row["target"],
                    rewrites=row["rewrites"],
                )
                llm_response = await utils.request_llm(
                    client, prompt, model_name, temparature, enable_json=True
                )

                json_obj = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "target": row["target"],
                    "rewrites": row["rewrites"],
                    "verification": llm_response,
                    "prompt": prompt,
                }
                writer.write(json_obj)

    if not args.no_baseline:
        await verify_rewrites("baseline")

    if not args.no_worstcase:
        await verify_rewrites("worstcase")


if __name__ == "__main__":
    asyncio.run(run())
