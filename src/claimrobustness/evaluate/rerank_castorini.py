import copy
from typing import Any, Dict, List, Union

from rank_llm.data import Request
from rank_llm.rerank import IdentityReranker, RankLLM, Reranker


def rerank(
    model_path: str,
    requests: List[Request],
    top_k_retrieve: int = 50,
    top_k_rerank: int = 20,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    num_passes: int = 1,
    interactive: bool = False,
    default_agent: RankLLM = None,
    **kwargs: Any,
):
    """Retrieve candidates using Anserini API and rerank them

    Returns:
        - List of top_k_rerank candidates
    """

    # Get reranking agent
    reranker = Reranker(
        Reranker.create_agent(model_path, default_agent, interactive, **kwargs)
    )

    # Reranking stages
    print(f"Reranking and returning {top_k_rerank} passages with {model_path}...")
    if reranker.get_agent() is None:
        # No reranker. IdentityReranker leaves retrieve candidate results as is or randomizes the order.
        shuffle_candidates = True if model_path == "rank_random" else False
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
        )
    else:
        # Reranker is of type RankLLM
        for pass_ct in range(num_passes):
            print(f"Pass {pass_ct + 1} of {num_passes}:")

            rerank_results = reranker.rerank_batch(
                requests,
                rank_end=top_k_retrieve,
                rank_start=0,
                shuffle_candidates=shuffle_candidates,
                logging=print_prompts_responses,
                top_k_retrieve=top_k_retrieve,
                **kwargs,
            )

        if num_passes > 1:
            requests = [
                Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                for r in rerank_results
            ]

    for rr in rerank_results:
        rr.candidates = rr.candidates[:top_k_rerank]

    if interactive:
        return (rerank_results, reranker.get_agent())
    else:
        return rerank_results
