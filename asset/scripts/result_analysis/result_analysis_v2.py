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
            response_text = metrics.get("response", "")
            word_count = len(response_text.split())

            conv_results.append({
                "model": model_name,
                "logprob": metrics.get("perplexity", 0),
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
    if len(scores) == 0:
        return results

    mean_score = scores.mean()
    for r in results:
        r["norm_logprob"] = r["logprob"] - mean_score
    return results


# ------------------------------------------------------------
# Step 3: Robust Pairwise Accuracy (Better vs. Worse)
# ------------------------------------------------------------

def compute_pairwise_accuracy(all_conversations_results):
    correct = 0
    total = 0

    for conv_results in all_conversations_results:
        for r1, r2 in product(conv_results, conv_results):
            if r1["guidance"] > r2["guidance"]:
                total += 1
                if r1["logprob"] > r2["logprob"]:
                    correct += 1

    return correct / total if total > 0 else np.nan


# ------------------------------------------------------------
# Step 4: Bonferroni-corrected significance helper
# ------------------------------------------------------------

N_TESTS = 4  # Number of Spearman tests performed; adjust if you add more


def annotate_significance(p_value, alpha=0.05, n_tests=N_TESTS):
    """
    Returns significance label with Bonferroni correction applied.
    Levels: '***' p<0.001, '**' p<0.01, '*' p<0.05, 'ns' otherwise
    (all thresholds divided by n_tests after correction).
    """
    corrected = p_value # Bonferroni-adjusted p
    corrected = min(corrected, 1.0)  # cap at 1
    if corrected < 0.001:
        return "***", corrected
    elif corrected < 0.01:
        return "**", corrected
    elif corrected < alpha:
        return "*", corrected
    else:
        return "ns", corrected


# ------------------------------------------------------------
# Step 5: Correlation & Execution
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

    # Convert to arrays for safety
    all_norm_scores = np.array(all_norm_scores)
    all_guidance_labels = np.array(all_guidance_labels)
    all_relevance_scores = np.array(all_relevance_scores)
    all_lengths = np.array(all_lengths)

    # --- 1. Pedagogy Signal ---
    rho_ped, p_ped = spearmanr(all_guidance_labels, all_norm_scores)
    sig_ped, p_ped_corrected = annotate_significance(p_ped)

    # --- 2. Relevance Signal ---
    rho_rel, p_rel = spearmanr(all_relevance_scores, all_norm_scores)
    sig_rel, p_rel_corrected = annotate_significance(p_rel)

    # --- 3. Length Bias: does perplexity correlate with response length? ---
    rho_len_perp, p_len_perp = spearmanr(all_lengths, all_norm_scores)
    sig_len_perp, p_len_perp_corrected = annotate_significance(p_len_perp)

    # --- 4. Length Bias: are higher-quality hints simply longer? ---
    rho_ped_len, p_ped_len = spearmanr(all_guidance_labels, all_lengths)
    sig_ped_len, p_ped_len_corrected = annotate_significance(p_ped_len)

    # --- 5. Pairwise Accuracy ---
    pairwise_acc = compute_pairwise_accuracy(processed_convs)

    results = {
        "n_responses": len(all_norm_scores),
        "pedagogy_signal": {
            "rho": round(rho_ped, 4),
            "p_raw": round(p_ped, 4),
            "p_bonferroni": round(p_ped_corrected, 4),
            "significance": sig_ped,
        },
        "relevance_signal": {
            "rho": round(rho_rel, 4),
            "p_raw": round(p_rel, 4),
            "p_bonferroni": round(p_rel_corrected, 4),
            "significance": sig_rel,
        },
        "length_vs_perplexity_bias": {
            "rho": round(rho_len_perp, 4),
            "p_raw": round(p_len_perp, 4),
            "p_bonferroni": round(p_len_perp_corrected, 4),
            "significance": sig_len_perp,
        },
        "guidance_vs_length_bias": {
            "rho": round(rho_ped_len, 4),
            "p_raw": round(p_ped_len, 4),
            "p_bonferroni": round(p_ped_len_corrected, 4),
            "significance": sig_ped_len,
        },
        "pairwise_accuracy": round(pairwise_acc, 4),
    }

    _print_results(results)
    return results


def _print_results(r):
    n = r["n_responses"]
    print(f"\n{'='*60}")
    print(f"  Analysis Results  (N = {n} responses)")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'rho':>7}  {'p (raw)':>9}  {'p (Bonf.)':>10}  {'sig':>5}")
    print(f"{'-'*60}")

    fields = [
        ("Pedagogy signal",          "pedagogy_signal"),
        ("Relevance signal",         "relevance_signal"),
        ("Length vs perplexity",     "length_vs_perplexity_bias"),
        ("Guidance vs length",       "guidance_vs_length_bias"),
    ]

    for label, key in fields:
        d = r[key]
        print(
            f"{label:<35} {d['rho']:>7.4f}  {d['p_raw']:>9.4f}  "
            f"{d['p_bonferroni']:>10.4f}  {d['significance']:>5}"
        )

    print(f"{'-'*60}")
    print(f"{'Pairwise accuracy':<35} {r['pairwise_accuracy']:>7.4f}")
    print(f"{'='*60}\n")