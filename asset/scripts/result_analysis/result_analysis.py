import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from itertools import product

# ------------------------------------------------------------
# Step 1: Preprocess & Map
# ------------------------------------------------------------

def preprocess_samples_with_relevance(samples, guidance_dimension):
    processed_data = []
    guidance_map = {"Yes": 1.0, "To some extent": 0.5, "No": 0.0}

    for conv in samples:
        conv_results = []
        models_dict = conv.get("models", {})
        
        for model_name, metrics in models_dict.items():      
            # Extract text to calculate length
            response_text = metrics.get("response", "")
            word_count = len(response_text.split())
            
            conv_results.append({
                "model": model_name,
                "logprob": metrics.get("perplexity", 0), # Note: Higher = More Confused
                "guidance": guidance_map.get(metrics.get(guidance_dimension)),
                "relevance": metrics.get("relevance_F1", 0),
                "length": word_count 
            })
        
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

def run_full_analysis(samples, guidance_dimension):
    processed_convs = preprocess_samples_with_relevance(samples, guidance_dimension)
    
    all_norm_scores = []
    all_guidance_labels = []
    all_relevance_scores = []
    all_lengths = []
    
    for conv in processed_convs:
        for r in conv:
            all_norm_scores.append(r["norm_logprob"])
            all_guidance_labels.append(r["guidance"])
            all_relevance_scores.append(r["relevance"])
            all_lengths.append(r["length"])

    # 1. Pedagogy Signal
    s_corr_pedagogy, _ = spearmanr(all_guidance_labels, all_norm_scores)
    
    # 2. Relevance Signal 
    s_corr_relevance, _ = spearmanr(all_relevance_scores, all_norm_scores)

    # 3. Length Bias Check (The Reviewer Shield)
    # Does the model just get more confused (higher perplexity) by longer responses?
    length_bias_rho, _ = spearmanr(all_lengths, all_norm_scores)
    
    # Are the 'good' hints just significantly longer?
    pedagogy_length_rho, _ = spearmanr(all_guidance_labels, all_lengths)
    
    # 4. Pairwise Accuracy
    pairwise_acc = compute_pairwise_accuracy(processed_convs)

    return {
        "pedagogy_signal_rho": round(s_corr_pedagogy, 4),
        "relevance_signal_rho": round(s_corr_relevance, 4),
        "length_bias_rho": round(length_bias_rho, 4),
        "pedagogy_vs_length_rho": round(pedagogy_length_rho, 4),
        "pairwise_accuracy": round(pairwise_acc, 4),
        "total_responses_analyzed": len(all_norm_scores)
    }