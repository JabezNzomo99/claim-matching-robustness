import configparser
import argparse
import os
from claimrobustness import utils
from sentence_transformers import CrossEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import jsonlines
import json
import logging
import torch
import random
from transformers import T5ForConditionalGeneration
from rerank_t5 import MonoT5Reranker
from bge_reranker import BGEReranker
from bge_llm_reranker import BGELLMReranker

from rerank_castorini import rerank

from rank_llm.data import Request, Candidate, Query
from rank_llm.rerank import PromptMode

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

# Create a logger
logger = logging.getLogger(__name__)

all_experiments = {
    "named_entity_replacement": "experiments/named_entity_replacement/gpt4o/",
    "typos": "experiments/typos/gpt4o/",
    # "dialect_aae": "experiments/dialect/gpt4o/",
    "dialect_patois": "experiments/dialect/gpt4o/",
    # "dialect_pidgin": "experiments/dialect/gpt4o/",
    # "dialect_singlish": "experiments/dialect/gpt4o/",
    "casing": "experiments/casing/",
    "rewrite": "experiments/rewrite/gpt4o/",
    # "amplify_minimize": "experiments/amplify_minimize/gpt4o/",
    "negation": "experiments/negation/gpt4o/",
}

dialect_folder_mapping = {
    "dialect_aae": "aae",
    "dialect_patois": "patois",
    "dialect_pidgin": "pidgin",
    "dialect_singlish": "singlish",
}

rank_folder_mapping = {
    "dialect_aae": "african_american_english",
    "dialect_patois": "jamaican_patois",
    "dialect_pidgin": "pidgin",
    "dialect_singlish": "singlish",
}


class CrossEncoderDataset:
    def __init__(self, queries, claims, connections, test_ranks, n_candidates):
        """
        Initializes the dataset.
        :param queries: DataFrame of test queries (tweets).
        :param claims: DataFrame of target claims.
        :param connections: DataFrame of connections (query_id, target_id).
        :param test_ranks: Array of ranked candidate indices for each query (output of bi-encoder).
        :param n_candidates: Number of candidates to consider for each query.
        """
        self.dataset = []
        self.queries = queries
        self.claims = claims
        self.connections = connections
        self.test_ranks = test_ranks
        self.n_candidates = n_candidates

        # Build the dataset
        self._create_dataset()

    def _create_dataset(self):
        # Merge queries with connections to identify valid query-candidate pairs
        run_tweets = self.queries.merge(self.connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(self.claims, on="target_id", how="inner")

        # Map query IDs to their test ranks
        query_to_ranks = {
            query_id: ranks
            for query_id, ranks in zip(run_tweets["query_id"], self.test_ranks)
        }

        # Iterate over each query
        for query_id, query_text in run_tweets.groupby("query_id", sort=False)["query"]:

            # Get the bi-encoder's ranked candidate indices for this query
            ranked_indices = query_to_ranks.get(query_id, [])

            # Print the length of the ranked indices
            # print(f"Length of ranked indices: {len(ranked_indices)}")

            # Select the top candidates based on the randomized indices
            top_candidate_ids = ranked_indices[: self.n_candidates]
            # print(f"Length of top candidate IDS: {len(top_candidate_ids)}")
            # print(f"Top candidate IDS: {top_candidate_ids}")

            targets = self.claims["target"].to_list()
            # Added this to clean the targets - espeially for fact-check-tweet dataset
            candidate_texts = [
                targets[candidate_id].replace("\n", " ").strip()
                for candidate_id in top_candidate_ids
            ]

            # print(f"Length of candidate texts: {len(candidate_texts)}")

            # Add query and its top candidates to the dataset
            self.dataset.append(
                {"query": query_text.iloc[0], "candidates": candidate_texts}
            )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument(
        "--n-candidates", type=int, default=50, help="overwrite number of candidates"
    )
    parser.add_argument(
        "--model-name", type=str, default="", help="Pass the name of the reranker model"
    )
    # Add boolean argument for including bm25 to the embedding model list and set default to True
    parser.add_argument(
        "--include-bm25",
        action="store_true",
        default=False,
        help="Include BM25 in the list of embedding models",
    )
    # Add argument to pass the list of experiments to run
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=list(all_experiments.keys()),
        help="Pass the list of experiments to run",
    )
    args = parser.parse_args()
    dataset = args.dataset
    n_candidates = args.n_candidates
    model_name = args.model_name
    include_bm25 = args.include_bm25
    experiments = args.experiments

    print(f"Running evaluation on the following experiments: {experiments}")

    # Filter experiments
    experiments_dict = {
        experiment: experiment_path
        for experiment, experiment_path in all_experiments.items()
        if experiment in experiments
    }

    for experiment, experiment_path in experiments_dict.items():
        config = configparser.ConfigParser()
        config.read(os.path.join(experiment_path, "config.ini"))

        embedding_models = config["evaluation"].get("embedding_models").split(",")
        if include_bm25:
            embedding_models.append("bm25")
        logging.info(f"Running evaluation on the following models: {embedding_models}")

        data_directory = config["evaluation"].get("data_directory")

        logger.info("Loading the evaluation data")

        def load_evaluation_data(path: str) -> pd.DataFrame:
            return pd.read_csv(
                path, names=["query_id", "query"], skiprows=[0], sep="\t"
            ).drop_duplicates()

        # Load the datasets
        data = utils.load_data(dataset=dataset)
        targets = data["targets"]
        _, test_qrels = data["test"]
        print(f"Length of targets {len(targets)}")

        if "dialect" in experiment:
            # Load the original baseline and edited baseline
            original_baseline_path = os.path.join(
                experiment_path,
                dataset,
                dialect_folder_mapping[experiment],
                "orig_baseline_dialect.tsv",
            )
            edited_baseline_path = os.path.join(
                experiment_path,
                dataset,
                dialect_folder_mapping[experiment],
                "edited_baseline_dialect.tsv",
            )

            original_baseline = load_evaluation_data(original_baseline_path)
            edited_baseline = load_evaluation_data(edited_baseline_path)

            evaluation_data_dict = {
                "original_baseline": original_baseline,
                "edited_baseline": edited_baseline,
            }
        else:
            # Load data paths
            original_baseline_path = os.path.join(
                experiment_path,
                dataset,
                config["evaluation"].get("original_baseline_path"),
            )
            edited_baseline_path = os.path.join(
                experiment_path,
                dataset,
                config["evaluation"].get("edited_baseline_path"),
            )
            original_worstcase_path = os.path.join(
                experiment_path,
                dataset,
                config["evaluation"].get("original_worstcase_path"),
            )
            edited_worstcase_path = os.path.join(
                experiment_path,
                dataset,
                config["evaluation"].get("edited_worstcase_path"),
            )

            original_baseline = load_evaluation_data(original_baseline_path)
            edited_baseline = load_evaluation_data(edited_baseline_path)
            original_worstcase = load_evaluation_data(original_worstcase_path)
            edited_worstcase = load_evaluation_data(edited_worstcase_path)

            # Print the shape of the data
            print(f"Original Baseline Shape: {original_baseline.shape}")
            print(f"Edited Baseline Shape: {edited_baseline.shape}")
            print(f"Original Worstcase Shape: {original_worstcase.shape}")
            print(f"Edited Worstcase Shape: {edited_worstcase.shape}")

            evaluation_data_dict = {
                # "original_baseline": original_baseline,
                "edited_baseline": edited_baseline,
                # "original_worstcase": original_worstcase,
                "edited_worstcase": edited_worstcase,
            }

        def get_idx(connections, claims, tweets):
            run_tweets = tweets.merge(connections, on="query_id", how="inner")
            run_tweets = run_tweets.merge(claims, on="target_id", how="inner")
            run_tweets = run_tweets[["query", "target"]].reset_index()
            claim_idx = [
                claims.target.to_list().index(t_claim)
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

        def get_reranks(dataset, device):
            reranks = []
            cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device
            )  # Specify the appropriate cross-encoder model

            for i in tqdm(range(len(dataset))):
                item = dataset[i]
                query = item["query"]
                candidates = item["candidates"]
                # Generate query-candidate pairs for scoring
                query_candidate_pairs = [(query, candidate) for candidate in candidates]
                # Compute relevance scores using the cross-encoder
                scores = cross_encoder.predict(query_candidate_pairs)
                # Rank the candidates based on the scores
                ranked_indices = np.argsort(scores)[
                    ::-1
                ]  # Sort scores in descending order
                reranks.append(ranked_indices)
            return reranks

        # Implement MonoT5 reranker
        def get_reranks_t5(dataset, device):
            reranks = []
            # Initialize the MonoT5Reranker
            reranker = MonoT5Reranker(
                device=device,
                model_name_or_path="castorini/monot5-3b-msmarco-10k",
                torch_compile=False,
            )
            for i in tqdm(range(len(dataset))):
                item = dataset[i]
                query = item["query"]
                candidates = item["candidates"]
                # Generate query-candidate pairs for scoring
                query_candidate_pairs = [(query, candidate) for candidate in candidates]
                # Compute relevance scores using the cross-encoder
                scores = reranker.rescore(query_candidate_pairs)
                # Rank the candidates based on the scores
                ranked_indices = np.argsort(scores)[
                    ::-1
                ]  # Sort scores in descending order
                reranks.append(ranked_indices)
            return reranks

        def get_reranks_bge(dataset, device, batch_size=16):
            reranker = BGELLMReranker(device=device, batch_size=batch_size)
            reranks = []

            all_pairs = []
            query_indices = []

            # Prepare all pairs and track query indices
            for i, item in enumerate(dataset):
                query = item["query"]
                candidates = item["candidates"]
                query_candidate_pairs = [(query, candidate) for candidate in candidates]
                all_pairs.extend(query_candidate_pairs)
                query_indices.extend([i] * len(candidates))

            # Rescore in batches
            scores = reranker.rescore(all_pairs)

            # Group scores back by query
            current_index = 0
            for i, item in enumerate(dataset):
                num_candidates = len(item["candidates"])
                query_scores = scores[current_index : current_index + num_candidates]
                ranked_indices = np.argsort(query_scores)[::-1]
                reranks.append(ranked_indices)
                current_index += num_candidates

            return reranks

        def get_reranks_llms(n_candidates, model_path, dataset, test_queries, targets):
            # List to store the requests
            # Prepare the requests for evaluation
            requests = []

            for item in dataset:
                # Extract query data
                query_text = item["query"]  # This is the query text
                query_qid = test_queries.loc[
                    test_queries["query"] == query_text, "query_id"
                ].iloc[0]
                # Retrieve the query_id from test_queries DataFrame
                query = Query(text=query_text, qid=query_qid)

                # Extract candidates data
                candidates = []
                for candidate_text in item["candidates"]:
                    # Locate the candidate in the targets DataFrame
                    candidate_row = targets[targets["target"] == candidate_text].iloc[0]
                    candidate_docid = candidate_row["target_id"]
                    candidate_score = (
                        0  # Placeholder: Update with actual score if available
                    )
                    candidate_doc = {
                        "passage": candidate_text,
                    }  # Full row as the document

                    # Create Candidate object
                    candidate = Candidate(
                        docid=candidate_docid, score=candidate_score, doc=candidate_doc
                    )
                    candidates.append(candidate)

                # Create the Request object
                request = Request(query=query, candidates=candidates)

                # Append to the requests list
                requests.append(request)
            results = rerank(
                model_path=model_path,
                requests=requests,
                top_k_retrieve=n_candidates,
                top_k_rerank=n_candidates,
                prompt_mode=PromptMode.RANK_GPT,
                context_size=4096,
                use_azure_openai=False,
                # vllm_batched=True,
                # # batch_size=16,
                # variable_passages=True,
                # window_size=n_candidates,
                # num_gpus=1,
            )
            # Process results and return reranks
            reranks = []
            for result, item in zip(results, dataset):
                # Use the ranked order in result and map docid back to the original indices
                docid_to_index = {
                    candidate_row.target_id: idx
                    for idx, candidate_text in enumerate(item["candidates"])
                    for candidate_row in targets[
                        targets["target"] == candidate_text
                    ].itertuples()
                }
                ranked_indices = [
                    docid_to_index[candidate.docid]
                    for candidate in result.candidates
                    if candidate.docid in docid_to_index
                ]
                reranks.append(ranked_indices)

            return reranks

        # Create a results directory if not exists
        # Save the file output
        if "dialect" in experiment:
            results_directory = f"{experiment_path}/{dataset}/results/{experiment}"
        else:
            results_directory = f"{experiment_path}/{dataset}/results"
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        if "dialect" in experiment:
            cache_dir = os.path.join(data_directory, "dialect", dataset)
        else:
            cache_dir = os.path.join(data_directory, experiment, dataset)
        # Create directory if it does not exist
        os.makedirs(cache_dir, exist_ok=True)

        all_claims_dir = os.path.join(data_directory, "embeddings", dataset)

        device = utils.get_device()

        results_file_path = os.path.join(
            results_directory,
            f"after_reranking_n_{n_candidates}_rebuttal.jsonl",
        )

        # Load existing results to check for already evaluated models
        evaluated_models = set()
        if os.path.exists(results_file_path):
            with jsonlines.open(results_file_path, mode="r") as reader:
                for obj in reader:
                    for model in embedding_models:
                        if model in obj:
                            evaluated_models.add(model)
        ptn = "test"
        with jsonlines.open(results_file_path, mode="a") as writer:
            for embedding_model_path in tqdm(embedding_models):
                # Check if the model has already been evaluated
                if embedding_model_path in evaluated_models:
                    print(f"Skipping evaluation for model: {embedding_model_path}")
                    continue

                print(f"Running evaluation on model: {embedding_model_path}")
                embedding_save_dir = os.path.join(cache_dir, embedding_model_path)
                embedding_model_results = {}
                for key, test_queries in tqdm(evaluation_data_dict.items()):
                    print(f"Running evaluation on {key}")
                    if "dialect" in experiment:
                        embedding_save_dir = os.path.join(
                            cache_dir,
                            rank_folder_mapping[experiment],
                            embedding_model_path,
                        )
                        test_ranks_path = os.path.join(
                            embedding_save_dir, f"{key}_ranks_{ptn}.npy"
                        )
                    else:
                        test_ranks_path = os.path.join(
                            embedding_save_dir, f"{key}_ranks_{ptn}.npy"
                        )
                    test_ranks = np.load(test_ranks_path)

                    # Print the shape of the test ranks
                    logger.info(f"Test Ranks Shape: {test_ranks.shape}")
                    map_results = {}
                    map_recall_results = {}
                    run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
                    k_values = list(
                        filter(lambda k: k <= n_candidates, [1, 5, 10, 20, 50])
                    )

                    # Add CrossEncoder implementation
                    test_dataset = CrossEncoderDataset(
                        queries=test_queries,
                        claims=targets,
                        connections=test_qrels,
                        test_ranks=test_ranks,
                        n_candidates=n_candidates,
                    )

                    # TODO: Add support for dictionary that loads the function to rerank
                    test_reranks = get_reranks_bge(test_dataset, device)
                    # # Get the first item in the dataset
                    # first_item = test_dataset[0]

                    # # Prepare the content for the text file
                    # query_text = first_item["query"]
                    # query_qid = test_queries.loc[
                    #     test_queries["query"] == query_text, "query_id"
                    # ].iloc[0]
                    # before_reranking_text = first_item["candidates"]
                    # candidate_texts = "\n".join(
                    #     [
                    #         f"{index + 1}. {candidate}"
                    #         for index, candidate in enumerate(before_reranking_text)
                    #     ]
                    # )

                    # reranks_sample = test_reranks[0]
                    # # Reorder candidates based on new ranks
                    # reordered_candidates = [
                    #     candidate
                    #     for _, candidate in sorted(
                    #         zip(reranks_sample, before_reranking_text)
                    #     )
                    # ]
                    # reordered_candidates_text = "\n".join(
                    #     [
                    #         f"{index + 1}. {candidate}"
                    #         for index, candidate in enumerate(reordered_candidates)
                    #     ]
                    # )

                    # # Gold index
                    # gold_index = claim_idx[0]
                    # # Retrieve the target related to the gold index
                    # gold_target = (
                    #     (run_tweets.loc[run_tweets.index == 0, "target"].iloc[0])
                    #     .replace("\n", " ")
                    #     .strip()
                    # )
                    # print(run_tweets.head(1))

                    # # Reorder candidates with new ranks
                    # output_text = f"Query ID:{query_qid}\nQuery:\n{query_text}\n\nGold:{gold_target}\n\nBefore Ranking Candidates:\n{candidate_texts}\n\nAfter Ranking Candidates:\n\n{reordered_candidates_text}"

                    # # Write the content to a text file
                    # file_path = "first_item_query_and_candidates.txt"
                    # with open(file_path, "w") as f:
                    #     f.write(output_text)
                    # print(
                    #     "Here is an example of the test_reranks:",
                    #     test_ranks[0][n_candidates],
                    # )
                    # print("Here is an example of the test_reranks:", test_reranks[0])
                    # # test_reranks = get_reranks_t5(test_dataset, device)
                    # # test_reranks = get_reranks_llms(
                    # #     n_candidates,
                    # #     "castorini/rank_zephyr_7b_v1_full",
                    # #     test_dataset,
                    # #     test_queries,
                    # #     targets,
                    # # )
                    # test_reranks = get_reranks_llms(
                    #     n_candidates,
                    #     "castorini/monot5-3b-msmarco-10k",
                    #     test_dataset,
                    #     test_queries,
                    #     targets,
                    # )
                    # test_reranks = get_reranks_llms(
                    #     n_candidates,
                    #     "gpt-4o",
                    #     test_dataset,
                    #     test_queries,
                    #     targets,
                    # )
                    # # test_reranks = get_reranks_llms(
                    # #     n_candidates,
                    # #     "castorini/LiT5-Distill-large-v2",
                    # #     test_dataset,
                    # #     test_queries,
                    # #     targets,
                    # # )

                    # # Debugging detour
                    # # Check lengths of test_ranks and test_reranks
                    # rank_lengths = [len(ids) for ids in test_ranks]
                    # rerank_lengths = [len(rerank) for rerank in test_reranks]

                    # print("Lengths of test_ranks:", rank_lengths)
                    # print("Lengths of test_reranks:", rerank_lengths)

                    # # Check consistency
                    # consistent_ranks = all(
                    #     length == rank_lengths[0] for length in rank_lengths
                    # )
                    # consistent_reranks = all(
                    #     length == rerank_lengths[0] for length in rerank_lengths
                    # )

                    # print("test_ranks lengths are consistent:", consistent_ranks)
                    # print("test_reranks lengths are consistent:", consistent_reranks)

                    ranks = np.array(
                        [ids[rerank] for ids, rerank in zip(test_ranks, test_reranks)]
                    )

                    # Print an example of test ranks for the first item limited to n_candidates
                    print(
                        "Here is an example of the test ranks before ranking:",
                        test_ranks[0][:n_candidates],
                    )
                    print("Here is an example of the ranks after ranking:", ranks[0])

                    map_results[ptn] = []
                    for n in k_values:
                        map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

                    map_recall_results[ptn] = []
                    for n in k_values:
                        map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))

                    results = {
                        "map_results": map_results,
                        "map_recall_results": map_recall_results,
                    }
                    embedding_model_results[key] = results
                all_embedding_model_results = {
                    embedding_model_path: embedding_model_results
                }
                writer.write(all_embedding_model_results)
                # Clear GPU
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
