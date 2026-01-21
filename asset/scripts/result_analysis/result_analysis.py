import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from itertools import product

# ------------------------------------------------------------
# Step 1: Collect per-response scores
# ------------------------------------------------------------
"""
    Returns a list of dicts:
    [
        {
            "model": str,
            "logprob": float,
            "guidance": 0 or 1
        }
    ]
"""
 

# ------------------------------------------------------------
# Step 2: Normalize scores within each conversation
# ------------------------------------------------------------

def normalize_within_conversation(results):
    scores = np.array([r["logprob"] for r in results])
    mean_score = scores.mean()

    for r in results:
        r["norm_logprob"] = r["logprob"] - mean_score

    return results


# ------------------------------------------------------------
# Step 3: Aggregate across conversations
# ------------------------------------------------------------

def aggregate_results(all_conversations_results):
    norm_scores = []
    guidance_labels = []

    for conv_results in all_conversations_results:
        for r in conv_results:
            norm_scores.append(r["norm_logprob"])
            guidance_labels.append(r["guidance"])

    return np.array(norm_scores), np.array(guidance_labels)


# ------------------------------------------------------------
# Step 4a: Correlation analysis
# ------------------------------------------------------------

def compute_correlations(norm_scores, guidance_labels):
    pearson_corr, pearson_p = pointbiserialr(guidance_labels, norm_scores)
    spearman_corr, spearman_p = spearmanr(guidance_labels, norm_scores)

    return {
        "point_biserial_corr": pearson_corr,
        "point_biserial_p": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
    }


# ------------------------------------------------------------
# Step 4b: Pairwise accuracy (preferred)
# ------------------------------------------------------------

def compute_pairwise_accuracy(all_conversations_results):
    correct = 0
    total = 0

    for conv_results in all_conversations_results:
        yes_items = [r for r in conv_results if r["guidance"] == 1]
        no_items = [r for r in conv_results if r["guidance"] == 0]

        for y, n in product(yes_items, no_items):
            total += 1
            if y["norm_logprob"] > n["norm_logprob"]:
                correct += 1

    return correct / total if total > 0 else np.nan


# ------------------------------------------------------------
# Example driver
# ------------------------------------------------------------

def run_evaluation(dataset, logprob_fn):
    all_conv_results = []

    for conversation in dataset:
        conv_results = compute_scores_for_conversation(
            conversation,
            logprob_fn,
        )
        conv_results = normalize_within_conversation(conv_results)
        all_conv_results.append(conv_results)

    norm_scores, guidance_labels = aggregate_results(all_conv_results)

    correlations = compute_correlations(norm_scores, guidance_labels)
    pairwise_acc = compute_pairwise_accuracy(all_conv_results)

    return {
        "correlations": correlations,
        "pairwise_accuracy": pairwise_acc,
        "num_samples": len(norm_scores),
    }
