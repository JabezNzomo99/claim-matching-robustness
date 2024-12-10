# Run threshold classifier separately for each transformation type
# Pass the config file for the specific perturbation type
# Load the files and the claims
# For each model, perform encodings
import configparser
import argparse
import os
from claimrobustness import utils
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import jsonlines


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="path where config lies")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    embedding_models = config["evaluation"].get("embedding_models").split(",")
    print(f"Running evaluation on the following models: {embedding_models}")

    # Load the dataset name
    dataset = config["data"].get("dataset")
    threshold_classifier_path = config["evaluation"].get("threshold_classifier_path")

    # Load the original baseline and edited baseline
    original_baseline_path = os.path.join(
        args.experiment_path,
        dataset,
        config["evaluation"].get("original_baseline_path"),
    )
    edited_baseline_path = os.path.join(
        args.experiment_path, dataset, config["evaluation"].get("edited_baseline_path")
    )

    # Load the original worstcase and edited worstcase
    original_worstcase_path = os.path.join(
        args.experiment_path,
        dataset,
        config["evaluation"].get("original_worstcase_path"),
    )
    edited_worstcase_path = os.path.join(
        args.experiment_path, dataset, config["evaluation"].get("edited_worstcase_path")
    )

    def load_evaluation_data(path: str) -> pd.DataFrame:
        return pd.read_csv(
            path, names=["query_id", "query"], skiprows=[0]
        ).drop_duplicates()

    # Load the datasets
    data = utils.load_data(dataset=dataset)
    targets = data["targets"]
    _, test_qrels = data["test"]

    original_baseline = load_evaluation_data(original_baseline_path)
    edited_baseline = load_evaluation_data(edited_baseline_path)
    original_worstcase = load_evaluation_data(original_worstcase_path)
    edited_worstcase = load_evaluation_data(edited_worstcase_path)

    evaluation_data_dict = {
        "original_baseline": original_baseline,
        "edited_baseline": edited_baseline,
        "original_worstcase": original_worstcase,
        "edited_worstcase": edited_worstcase,
    }

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

    # Create a results directory if not exists
    save_dir = f"{args.experiment_path}/results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = utils.get_device()
    results_file_path = os.path.join(save_dir, "threshold_classifier_results.jsonl")
    # Load existing results
    evaluated_models = set()
    if os.path.exists(results_file_path):
        with jsonlines.open(results_file_path, mode="r") as reader:
            for obj in reader:
                for model in embedding_models:
                    if model in obj:
                        evaluated_models.add(model)

    thresholds = {}
    if os.path.exists(threshold_classifier_path):
        with jsonlines.open(threshold_classifier_path, mode="r") as reader:
            for obj in reader:
                for model in embedding_models:
                    if model in obj:
                        thresholds[model] = obj[model]

    with jsonlines.open(results_file_path, mode="a") as writer:
        for embedding_model_path in tqdm(embedding_models):
            # Check if the model has already been evaluated
            if embedding_model_path in evaluated_models:
                print(f"Skipping evaluation for model: {embedding_model_path}")
                continue

            print(f"Running evaluation on model: {embedding_model_path}")

            if embedding_model_path == "LLM2Vec-Meta-Llama-3-8B":
                llm_encoder = utils.init_llm_encoder()
            else:
                model = SentenceTransformer(embedding_model_path, device=device)

            embedding_model_results = {}
            print(thresholds)
            # Get the threshold for the classifier
            threshold = thresholds.get(embedding_model_path)["best_threshold"]
            for key, test_queries in tqdm(evaluation_data_dict.items()):
                print(f"Running evaluation for {key}")
                run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
                if embedding_model_path == "LLM2Vec-Meta-Llama-3-8B":
                    query_embs = utils.encode_with_llm(
                        model=llm_encoder,
                        sentences=run_tweets["query"].to_list(),
                        encoding_type=utils.EncodingType.QUERY,
                        instruction="Given a claim, retrieve relevant fact checks that match the claim:",
                    )
                    claim_embs = utils.encode_with_llm(
                        model=llm_encoder,
                        sentences=run_tweets["target"].to_list(),
                        encoding_type=utils.EncodingType.DOCUMENT,
                    )
                    cosine_scores = utils.cosine_similarity_llm(query_embs, claim_embs)
                else:
                    query_embs = model.encode(
                        run_tweets["query"].to_list(),
                        prompt="Represent the fact for retrieving supporting evidence:",
                        show_progress_bar=True,
                        device=device,
                    )
                    claim_embs = model.encode(
                        run_tweets["target"].to_list(),
                        prompt="Represent the evidence for retrieval:",
                        show_progress_bar=True,
                        device=device,
                    )
                    cosine_scores = utils.cosine_similarity(query_embs, claim_embs)

                true_labels = np.ones(len(run_tweets)).astype(int)

                predictions = (cosine_scores >= threshold).astype(int)
                test_f1_score = f1_score(true_labels, predictions)

                results = {
                    "f1_score": test_f1_score,
                    "threshold": threshold,
                }
                embedding_model_results[key] = results
            all_results = {embedding_model_path: embedding_model_results}
            writer.write(all_results)


if __name__ == "__main__":
    run()
