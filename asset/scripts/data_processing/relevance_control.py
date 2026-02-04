import torch
from typing import Any, Dict, Optional, Tuple
from bert_score import score as bert_score
from scipy.stats import skew, kurtosis
import numpy as np
import json

def _default_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    if getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bertscore_relevance(
    response: str,
    conversation_history: str,
    *,
    model_name: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    device: Optional[str] = None,
    rescale_with_baseline: bool = True,
    ) -> Dict[str, float]:

    response = (response or "").strip()
    if not response:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    context = (conversation_history or "").strip()
    if not context:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


    if device is None:
        device = _default_device()

    P, R, F1 = bert_score(
        cands=[response],
        refs=[context],
        model_type=model_name,
        lang=lang,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )

    return {
        "precision": float(P[0].item()),
        "recall": float(R[0].item()),
        "f1": float(F1[0].item()),
    }


def score_sample_models_bertscore(
    sample: Dict[str, Any],
    *,
    model_keys: Optional[Tuple[str, ...]] = None,
    response_field: str = "response",
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    device: Optional[str] = None,
) -> Dict[str, Any]:

    conversation_history = (sample.get("conversation_history", "") or "").strip()

    if model_keys is None:
        inferred = []
        for k, v in sample.items():
            if isinstance(v, dict) and response_field in v and isinstance(v.get(response_field), str):
                inferred.append(k)
        model_keys = tuple(inferred)

    out: Dict[str, Any] = {}

    for mk in model_keys:
        resp = ""
        block = sample.get(mk)
        if isinstance(block, dict):
            resp = block.get(response_field, "") or ""

        scores = bertscore_relevance(
            resp,
            conversation_history,
            model_name=bertscore_model,
            device=device,
        )
        out[mk] = {**scores}

    return out

def filter_irrelevant_responses(scored_sample: Dict[str, Any], threshold: float = 3.5):
    model_data = scored_sample
    model_names = list(model_data.keys())
    scores = np.array([model_data[m]['f1'] for m in model_names])
    
    # 1. Calculate Robust Statistics
    median_score = np.median(scores)
    # MAD = Median of absolute differences from the median
    mad = np.median(np.abs(scores - median_score))
    
    # 0.6745 is the standard consistency constant for normal distributions
    # We use a tiny value for mad if it's 0 to avoid division by zero
    modified_z_scores = 0.6745 * (scores - median_score) / (mad if mad > 0 else 1e-6)

    kept_models = {}
    discarded_models = []

    for i, m_name in enumerate(model_names):
        # We discard anything that deviates too far from the median
        if abs(modified_z_scores[i]) > threshold:
            discarded_models.append(m_name)
        else:
            kept_models[m_name] = model_data[m_name]

    # 2. Safety Check: Relevance Floor
    # Even if they are consistent, are they all consistently BAD?
    # If the median relevance is below 0.80, the whole set is probably noise.
    if median_score < 0.80:
        return {"filter_status": "discarded_set_low_relevance_consensus"}

    # 3. Safety Check: Remaining Diversity
    if len(kept_models) < 2:
        return {"filter_status": "discarded_set_no_consensus"}

    return {
        "filter_status": "processed",
        "kept_models": kept_models,
        "discarded_models": discarded_models,
        "median_relevance": float(median_score)
    }

if __name__ == "__main__":
    # Open and load sample data
    data_file = '../data/current_data_v2.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Process each sample
    processed_results = []
    for sample in data:
        scored = score_sample_models_bertscore(sample)
        filtered = filter_irrelevant_responses(scored, threshold=3.5)
        processed_results.append({
            "conversation_id": sample.get("conversation_id"),
            "scored": scored,
            "filtered": filtered
        })
    
    # Print summary of how many samples were processed
    total_samples = len(processed_results)
    discarded_count = sum(1 for r in processed_results if r['filtered']['filter_status'] != 'processed')
    print(f"Total Samples Processed: {total_samples}")
    print(f"Samples Discarded due to Irrelevance: {discarded_count}")
    print(f"Samples Kept: {total_samples - discarded_count}")

    # Print summary of how many models were discarded per sample
    for result in processed_results:
        filtered = result['filtered']
        num_discarded = len(filtered['discarded_models'])
        print(f"Conversation ID: {result['conversation_id']} - Discarded Models: {num_discarded}")

    # Save processed results
    output_file = 'relevance_filtered_results.json'
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=4)
    

