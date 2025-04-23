import configparser
import argparse
import os
from claimrobustness import utils
from sentence_transformers import SentenceTransformer

# from laser_encoders import LaserEncoderPipeline
# from rolase import RoLaserEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import jsonlines
import json
import logging
import torch

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

# Create a logger
logger = logging.getLogger(__name__)

experiments_dict = {
    "named_entity_replacement": "experiments/named_entity_replacement/gpt4o/",
    "typos": "experiments/typos/gpt4o/",
    "dialect": "experiments/dialect/gpt4o/",
    # "casing": "experiments/casing/",
    # "rewrite": "experiments/rewrite/gpt4o/",
    # "amplify_minimize": "experiments/amplify_minimize/gpt4o/",
    "negation": "experiments/negation/gpt4o/",
    # "ood": "experiments/ood/",
}

file_paths = {
    "named_entity_replacement": "experiments/named_entity_replacement/gpt4o/clef2021-checkthat-task2a--english/edited_worstcase_named_entity_replacements_normalised.tsv",
    "typos": "experiments/typos/gpt4o/clef2021-checkthat-task2a--english/edited_worstcase_typos_normalised.tsv",
    "dialect": "experiments/dialect/gpt4o/clef2021-checkthat-task2a--english/pidgin/edited_baseline_dialect_normalised.tsv",
    "negation": "experiments/negation/gpt4o/clef2021-checkthat-task2a--english/edited_worstcase_negation_normalised.tsv",
    # "ood": "experiments/ood/ood-dataset/ood_normalised_queries.tsv"
    # "ood": "experiments/ood/OOD-EN-Queries.tsv"
}

# Set the model and vocab paths, and pick the corresponding tokenizer type
# (spm for LASER, roberta for RoLASER, and char for c-RoLASER ):
base_path = "/data/kebl7383/laser/RoLASER"  # TODO: Change this to the path where the model is stored
rolaser_model_path = os.path.join(base_path, "rolaser.pt")
vocab = os.path.join(base_path, "rolaser.cvocab")
tokenizer = "roberta"

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


def run():
    # Load the config file from parameters
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path where config lies")
    parser.add_argument("--save-embs", action="store_true")
    parser.add_argument("--save-ranks", action="store_true")
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip generating baseline edits"
    )
    parser.add_argument(
        "--no-worstcase", action="store_true", help="Skip generating worstcase edits"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the dataset name
    dataset = args.dataset

    for experiment, experiment_path in experiments_dict.items():
        config = configparser.ConfigParser()
        config.read(os.path.join(experiment_path, "config.ini"))

        embedding_models = config["evaluation"].get("embedding_models").split(",")
        logger.info(f"Running evaluation on the following models: {embedding_models}")

        data_directory = config["evaluation"].get("data_directory")

        # # Load the original baseline and edited baseline
        # original_baseline_path = os.path.join(
        #     experiment_path,
        #     dataset,
        #     config["evaluation"].get("original_baseline_path"),
        # )
        # edited_baseline_path = os.path.join(
        #     experiment_path,
        #     dataset,
        #     config["evaluation"].get("edited_baseline_path"),
        # )

        # # Load the original worstcase and edited worstcase
        # original_worstcase_path = os.path.join(
        #     experiment_path,
        #     dataset,
        #     config["evaluation"].get("original_worstcase_path"),
        # )
        # edited_worstcase_path = os.path.join(
        #     experiment_path,
        #     dataset,
        #     config["evaluation"].get("edited_worstcase_path"),
        # )

        def load_evaluation_data(path: str) -> pd.DataFrame:
            return pd.read_csv(
                path, names=["query_id", "query"], skiprows=[0], sep="\t"
            ).drop_duplicates()

        # Load the datasets
        data = utils.load_data(dataset=dataset)
        targets = data["targets"]
        test_queries, test_qrels = data["test"]

        # logger.info("Loading the evaluation data")
        # original_baseline = load_evaluation_data(original_baseline_path)
        # edited_baseline = load_evaluation_data(edited_baseline_path)
        # original_worstcase = load_evaluation_data(original_worstcase_path)
        # edited_worstcase = load_evaluation_data(edited_worstcase_path)

        # print("original_baseline", original_baseline.shape)
        # print("edited_baseline", edited_baseline.shape)
        # print("original_worstcase", original_worstcase.shape)
        # print("edited_worstcase", edited_worstcase.shape)
        # evaluation_data_dict = {
        #     # "original_baseline": original_baseline,
        #     # "edited_baseline": edited_baseline,
        #     "original_worstcase": original_worstcase,
        #     "edited_worstcase": edited_worstcase,
        # }
        evaluation_data_dict = {
            "normalised_claims": load_evaluation_data(file_paths[experiment])
        }
        evaluation_data_dict = {"ood": load_evaluation_data(file_paths[experiment])}

        print("targets", targets.shape)
        print("test_queries", test_queries.shape)
        print("test_qrels", test_qrels.shape)

        # eval fns
        def get_idx(connections, claims, tweets):
            run_tweets = tweets.merge(connections, on="query_id", how="inner")
            run_tweets = run_tweets.merge(targets, on="target_id", how="inner")
            run_tweets = run_tweets[["query", "target"]].reset_index()
            claim_idx = [
                targets.target.to_list().index(t_claim)
                for t_claim in run_tweets.target.to_list()
            ]
            return run_tweets, claim_idx

        def avg_prec(gold, rankings, n):
            is_rel = (np.array(rankings)[:n] == gold).astype(float)
            return (is_rel / np.arange(1, n + 1)).sum()

        def recall(gold, rankings, n):
            is_rel = (np.array(rankings)[:n] == gold).astype(float)
            return is_rel.sum()

        def mean_avg_prec(golds, rankings, n):
            avg_precs = [
                avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)
            ]
            return np.array(avg_precs).mean()

        def mean_recall(golds, rankings, n):
            avg_precs = [recall(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
            return np.array(avg_precs).mean()

        def get_negative_ranks(ranks, gold):
            return [r for r in ranks if r != gold]

        def get_negative_ranks_arr(ranks, claim_idx):
            n_ranks = [get_negative_ranks(r, g) for r, g in zip(ranks, claim_idx)]
            return np.array(n_ranks)

        # Create a results directory if not exists
        # Save the file output
        results_directory = f"{experiment_path}/{dataset}/results"
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        cache_dir = os.path.join(data_directory, experiment, dataset)
        # Create directory if it does not exist
        os.makedirs(cache_dir, exist_ok=True)

        all_claims_dir = os.path.join(data_directory, "embeddings", dataset)

        device = utils.get_device()
        results_file_path = os.path.join(
            results_directory, "mitigation_cm_edited_results.jsonl"
        )
        # Load existing results to check for already evaluated models
        evaluated_models = set()
        if os.path.exists(results_file_path):
            with jsonlines.open(results_file_path, mode="r") as reader:
                for obj in reader:
                    for model in embedding_models:
                        if model in obj:
                            evaluated_models.add(model)

        with jsonlines.open(results_file_path, mode="a") as writer:
            for embedding_model_path in tqdm(embedding_models):
                # Check if the model has already been evaluated
                if embedding_model_path in evaluated_models:
                    print(f"Skipping evaluation for model: {embedding_model_path}")
                    continue

                logger.info(f"Running evaluation on model: {embedding_model_path}")
                cache_embedding_dir = os.path.join(all_claims_dir, embedding_model_path)
                os.makedirs(cache_embedding_dir, exist_ok=True)
                cached_emb_path = os.path.join(cache_embedding_dir, f"claim_embs.npy")

                # Add special support for NVEmbed
                if embedding_model_path == "nvidia/NV-Embed-v2":
                    model = SentenceTransformer(
                        embedding_model_path,
                        trust_remote_code=True,
                        device=device,
                        model_kwargs={"torch_dtype": "bfloat16"},
                    )
                    model.max_seq_length = 32768
                    model.tokenizer.padding_side = "right"
                # Add support for finetuned models
                elif embedding_model_path in utils.finetuned_models_path_mapping.keys():
                    model = SentenceTransformer(
                        utils.finetuned_models_path_mapping[embedding_model_path],
                        # device=device,
                    )
                # Add support for LASER
                elif embedding_model_path in ["laser2", "laser3"]:
                    model = LaserEncoderPipeline(laser="laser2", lang="en")
                elif embedding_model_path == "rolaser":
                    model = RoLaserEncoder(
                        model_path=rolaser_model_path, vocab=vocab, tokenizer=tokenizer
                    )
                else:
                    model = SentenceTransformer(embedding_model_path)
                # pool = model.start_multi_process_pool()
                # Check if embeddings are already cached
                if os.path.exists(cached_emb_path):
                    logger.info(f"Loading embeddings from {cached_emb_path}")
                    embs = np.load(cached_emb_path)
                elif embedding_model_path == "rolaser":
                    embs = model.encode(targets.target.tolist())
                elif embedding_model_path in ["laser2", "laser3"]:
                    logger.info(f"Embeddings not found in cache. Generating embeddings")
                    embs = model.encode_sentences(
                        targets.target.tolist(), normalize_embeddings=True
                    )
                else:
                    logger.info(f"Embeddings not found in cache. Generating embeddings")
                    embs = model.encode(
                        (
                            add_eos(targets.target.tolist(), model=model)
                            if embedding_model_path == "nvidia/NV-Embed-v2"
                            else targets.target.tolist()
                        ),
                        prompt=(
                            None
                            if embedding_model_path
                            in [
                                "nvidia/NV-Embed-v2",
                                "Salesforce/SFR-Embedding-Mistral",
                            ]
                            else "Represent the evidence for retrieval:"
                        ),
                        device=device,
                        batch_size=4,
                        show_progress_bar=True,
                        normalize_embeddings=(
                            True
                            if embedding_model_path == "nvidia/NV-Embed-v2"
                            else False
                        ),
                    )
                    if args.save_embs:
                        # np.save(
                        #     os.path.join(embedding_save_dir, f"{key}_tweet_embs.npy"),
                        #     all_tweet_embs,
                        # )
                        np.save(cached_emb_path, embs)
                embedding_model_results = {}
                for key, test_queries in tqdm(evaluation_data_dict.items()):
                    logger.info(f"Running evaluation on {key}")
                    map_results = {}
                    map_recall_results = {}
                    all_tweet_embs = {}
                    ptn = "test"
                    run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
                    print("run_tweets", run_tweets.shape)
                    if embedding_model_path in ["laser2", "laser3"]:
                        tweet_embs = model.encode_sentences(
                            run_tweets["query"].to_list(), normalize_embeddings=True
                        )
                    elif embedding_model_path == "rolaser":
                        tweet_embs = model.encode(run_tweets["query"].to_list())
                    else:
                        tweet_embs = model.encode(
                            (
                                add_eos(run_tweets["query"].to_list(), model=model)
                                if embedding_model_path == "nvidia/NV-Embed-v2"
                                else run_tweets["query"].tolist()
                            ),
                            prompt=(
                                nv_embed_query_prefix
                                if embedding_model_path
                                in [
                                    "nvidia/NV-Embed-v2",
                                    "Salesforce/SFR-Embedding-Mistral",
                                ]
                                else "Represent the tweet for retrieving supporting evidence:"
                            ),
                            batch_size=4,
                            # pool=pool,
                            device=device,
                            show_progress_bar=True,
                        )
                    all_tweet_embs[ptn] = tweet_embs
                    print("tweet_embs shape", tweet_embs.shape)
                    print("embs shape", embs.shape)
                    if embedding_model_path == "lydianish/RoLASER-v2":
                        scores = model.similarity(tweet_embs, embs).cpu().numpy()
                    else:
                        scores = tweet_embs @ embs.T
                    print("scores shape", scores.shape)
                    ranks = [score.argsort()[::-1] for score in scores]
                    if args.save_ranks:
                        embedding_save_dir = os.path.join(
                            cache_dir, embedding_model_path
                        )
                        if not os.path.exists(embedding_save_dir):
                            os.makedirs(embedding_save_dir)
                        np.save(
                            os.path.join(embedding_save_dir, f"{key}_ranks_{ptn}.npy"),
                            np.array(ranks),
                        )
                        # Log info that we have saved the ranks
                        logger.info(
                            f"Saved ranks for {embedding_model_path} and {key} to {embedding_save_dir}"
                        )
                        # np.save(
                        #     os.path.join(save_dir, f"{key}_ranks_{ptn}_negatives.npy"),
                        #     get_negative_ranks_arr(ranks, claim_idx),
                        # )

                    map_results[ptn] = []
                    # for n in [1, 5, 10, 20, 50, 100]:
                    #     map_results[ptn].append(
                    #         {f"map@{n}": mean_avg_prec(claim_idx, ranks, n)}
                    #     )

                    for n in [20]:
                        map_results[ptn].append(
                            {f"map@{n}": mean_avg_prec(claim_idx, ranks, n)}
                        )

                    # map_recall_results[ptn] = []
                    # for n in [1, 5, 10, 20, 50, 100]:
                    #     map_recall_results[ptn].append(
                    #         {f"recall@{n}": mean_recall(claim_idx, ranks, n)}
                    #     )

                    results = {
                        "map_results": map_results,
                        # "map_recall_results": map_recall_results,
                    }
                    embedding_model_results[key] = results
                all_embedding_model_results = {
                    embedding_model_path: embedding_model_results
                }
                # Clear model from memory
                del model
                torch.cuda.empty_cache()
                # model.stop_multi_process_pool(pool)
                writer.write(all_embedding_model_results)


if __name__ == "__main__":
    run()
