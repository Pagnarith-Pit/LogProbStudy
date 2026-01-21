import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from itertools import product

# ------------------------------------------------------------
# Step 1: Preprocess & Map
# ------------------------------------------------------------

def preprocess_samples(samples):
    processed_data = []
    
    # Ordinal mapping to allow mathematical comparison
    guidance_map = {
        "Yes": 1.0,
        "To some extent": 0.5,
        "No": 0.0
    }

    for conv in samples:
        conv_results = []
        models_dict = conv.get("models", {})
        
        for model_name, metrics in models_dict.items():
            # We exclude the 'Expert' reference from the model correlation pool
            if model_name == "Expert": 
                continue 
            
            conv_results.append({
                "model": model_name,
                "logprob": metrics.get("avg_log_prob", 0),
                "guidance": guidance_map.get(metrics.get("providing_guidance"), 0)
            })
        
        # Normalize logprobs relative to other models in the same conversation
        if conv_results:
            normalized_conv = normalize_within_conversation(conv_results)
            processed_data.append(normalized_conv)
        
    return processed_data

# ------------------------------------------------------------
# Step 2: Normalize
# ------------------------------------------------------------

def normalize_within_conversation(results):
    scores = np.array([r["logprob"] for r in results])
    if len(scores) == 0: return results
    
    mean_score = scores.mean()
    for r in results:
        r["norm_logprob"] = r["logprob"] - mean_score
    return results

# ------------------------------------------------------------
# Step 3: Robust Pairwise Accuracy (Better vs. Worse)
# ------------------------------------------------------------

def compute_pairwise_accuracy(all_conversations_results):
    """
    Computes accuracy by checking if the model gives a higher logprob 
    to the response with the higher guidance score.
    """
    correct = 0
    total = 0

    for conv_results in all_conversations_results:
        # Compare every model response against every other model response in the same conv
        for r1, r2 in product(conv_results, conv_results):
            # Only count the pair if one is strictly better than the other
            if r1["guidance"] > r2["guidance"]:
                total += 1
                # Check if the logprob reflects that superiority
                if r1["logprob"] > r2["logprob"]:
                    correct += 1

    return correct / total if total > 0 else np.nan

# ------------------------------------------------------------
# Step 4: Correlation & Execution
# ------------------------------------------------------------

def run_full_analysis(samples):
    processed_convs = preprocess_samples(samples)
    
    # Flatten for global correlation
    all_norm_scores = []
    all_guidance_labels = []
    for conv in processed_convs:
        for r in conv:
            all_norm_scores.append(r["norm_logprob"])
            all_guidance_labels.append(r["guidance"])

    # Calculate Metrics
    # Spearman is ideal here as it handles the 0.5 (ordinal) rank correctly
    s_corr, _ = spearmanr(all_guidance_labels, all_norm_scores)
    pb_corr, _ = pointbiserialr(all_guidance_labels, all_norm_scores)
    
    pairwise_acc = compute_pairwise_accuracy(processed_convs)

    return {
        "spearman_rho": round(s_corr, 4),
        "point_biserial": round(pb_corr, 4),
        "pairwise_accuracy": round(pairwise_acc, 4),
        "total_responses_analyzed": len(all_norm_scores)
    }

# Example call:
# results = run_full_analysis(samples)
# print(results)