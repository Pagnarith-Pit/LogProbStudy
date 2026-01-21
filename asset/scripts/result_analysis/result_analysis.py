import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from itertools import product

# ------------------------------------------------------------
# Step 1: Collect per-response scores
# ------------------------------------------------------------
"""
# Importing from JSON
samples = [
[
  {
    "conversation_id": "930-b01cb51d-748d-460c-841a-08e4d5cd5cc7",
    "ablation": false,
    "models": {
      "Gemini": {
        "response": "It looks like you're on the right track, but remember that Elliott still had 2,000 steps left to take *after* his jog.",
        "relevance_F1": -0.2107011079788208,
        "providing_guidance": "Yes",
        "actionability": "Yes",
        "total_log_prob": -251.84703063964844,
        "avg_log_prob": -1.4642269611358643,
        "perplexity": 4.324199199676514
      },
      "Phi3": {
        "response": "Great job! To further enhance your understanding, let's explore more examples and practice solving similar problems together.",
        "relevance_F1": -0.4032723307609558,
        "providing_guidance": "No",
        "actionability": "No",
        "total_log_prob": -248.02317810058594,
        "avg_log_prob": -1.4419952630996704,
        "perplexity": 4.229125499725342
      },
      "Llama318B": {
        "response": "That's a good attempt, but I'm not seeing how you factored in the remaining 2000 steps Elliott had left after his jog into your calculation - could you show me how you think those steps fit into the equation?",
        "relevance_F1": -0.13260850310325623,
        "providing_guidance": "Yes",
        "actionability": "Yes",
        "total_log_prob": -257.743408203125,
        "avg_log_prob": -1.4985082149505615,
        "perplexity": 4.475008010864258
      },
      "Llama31405B": {
        "response": "Here's a tutor response that acknowledges the student's effort and encourages them to revisit their thinking:\n\n\"I see that you've carefully calculated the steps Elliott took before his jog, but I'm wondering if you considered the fact that he only had 2,000 steps left to take after his jog, which might affect the number of steps he took during the jog itself?\"",
        "relevance_F1": -0.04744883254170418,
        "providing_guidance": "Yes",
        "actionability": "Yes",
        "total_log_prob": -246.78189086914062,
        "avg_log_prob": -1.4347784519195557,
        "perplexity": 4.198714733123779
      },
      "Mistral": {
        "response": "It seems like you forgot to consider the 2,000 steps Elliott still had left after his jog.",
        "relevance_F1": -0.2344737946987152,
        "providing_guidance": "Yes",
        "actionability": "Yes",
        "total_log_prob": -253.64093017578125,
        "avg_log_prob": -1.474656581878662,
        "perplexity": 4.369534969329834
      },
      "Expert": {
        "response": "But after his jog he still had 2000 steps left",
        "relevance_F1": -0.317008912563324,
        "providing_guidance": "Yes",
        "actionability": "Yes",
        "total_log_prob": -255.57855224609375,
        "avg_log_prob": -1.485921859741211,
        "perplexity": 4.419037342071533
      },
      "GPT4": {
        "response": "I see where your confusion is. You subtracted the total steps he had already taken from his goal, but the problem states that after his jog, Elliott had 2,000 steps left, which means he must have done 2,000 steps during his jog, not 4,000. Let's try to recalculate it together.",
        "relevance_F1": -0.016751287505030632,
        "providing_guidance": "Yes",
        "actionability": "No",
        "total_log_prob": -252.46620178222656,
        "avg_log_prob": -1.4678267240524292,
        "perplexity": 4.339793682098389
      },
      "Sonnet": {
        "response": "Great job showing your work, but let's take another look at the last part of the problem statement to make sure we're considering all the information given.",
        "relevance_F1": -0.33594223856925964,
        "providing_guidance": "To some extent",
        "actionability": "To some extent",
        "total_log_prob": -272.9019775390625,
        "avg_log_prob": -1.586639404296875,
        "perplexity": 4.8872971534729
      }
    }
  },
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
