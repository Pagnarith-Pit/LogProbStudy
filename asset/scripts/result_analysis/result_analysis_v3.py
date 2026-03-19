import json
import argparse
import numpy as np
from pathlib import Path
from itertools import product
from scipy.stats import spearmanr, wilcoxon, kruskal


# ============================================================
# Constants
# ============================================================

GUIDANCE_MAP = {"Yes": 1.0, "To some extent": 0.5, "No": 0.0}
N_TESTS = 4  # for Bonferroni correction across Spearman tests


# ============================================================
# Step 1: Loading
# ============================================================

def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Step 2: Preprocessing
# ============================================================

def preprocess_regular(samples: list, guidance_dimension: str) -> list:
    """
    For regular and mismatch experiments.
    Returns a list of conversations, each a list of per-model dicts.
    """
    processed = []
    for conv in samples:
        conv_results = []
        for model_name, metrics in conv.get("models", {}).items():
            response_text = metrics.get("response", "")
            guidance_raw = metrics.get(guidance_dimension)
            guidance_val = GUIDANCE_MAP.get(guidance_raw)

            if guidance_val is None:
                continue  # skip unmapped labels

            conv_results.append({
                "conversation_id": conv.get("conversation_id"),
                "model": model_name,
                "logprob": metrics.get("avg_log_prob", 0.0),
                "perplexity": metrics.get("perplexity", 0.0),
                "guidance": guidance_val,
                "relevance": metrics.get("relevance_F1", 0.0),
                "length": len(response_text.split()),
            })

        if conv_results:
            processed.append(normalize_within_conversation(conv_results))

    return processed


def preprocess_no_last_response(samples: list) -> dict:
    """
    For no-last-response experiment.
    Since all models share identical context, compute ONE score per conversation.
    Returns a dict keyed by conversation_id -> single score dict.
    """
    baseline = {}
    for conv in samples:
        conv_id = conv.get("conversation_id")
        models = conv.get("models", {})
        if not models:
            continue

        # All scores are identical — just take the first model's score
        first_metrics = next(iter(models.values()))
        baseline[conv_id] = {
            "logprob": first_metrics.get("avg_log_prob", 0.0),
            "perplexity": first_metrics.get("perplexity", 0.0),
        }

    return baseline


# ============================================================
# Step 3: Normalization
# ============================================================

def normalize_within_conversation(results: list) -> list:
    scores = np.array([r["logprob"] for r in results])
    mean_score = scores.mean()
    for r in results:
        r["norm_logprob"] = r["logprob"] - mean_score
    return results


# ============================================================
# Step 4: Significance helper (Bonferroni-corrected)
# ============================================================

def annotate_significance(p_value: float, alpha: float = 0.05, n_tests: int = N_TESTS) -> tuple:
    corrected = min(p_value * n_tests, 1.0)  # actual Bonferroni correction
    if corrected < 0.001:
        return "***", corrected
    elif corrected < 0.01:
        return "**", corrected
    elif corrected < alpha:
        return "*", corrected
    else:
        return "ns", corrected


# ============================================================
# Step 5: Pairwise Accuracy
# ============================================================

def compute_pairwise_accuracy(all_conversations_results: list) -> float:
    correct = 0
    total = 0
    for conv_results in all_conversations_results:
        for r1, r2 in product(conv_results, conv_results):
            if r1["guidance"] > r2["guidance"]:
                total += 1
                if r1["logprob"] > r2["logprob"]:
                    correct += 1
    return correct / total if total > 0 else float("nan")


# ============================================================
# Step 6: Ablation — Regular vs No-Last-Response (paired)
# ============================================================

def ablation_vs_no_last_response(
    regular_convs: list,
    baseline_dict: dict,
) -> dict:
    """
    Paired Wilcoxon test: for each conversation, compare the mean
    regular log-prob against the no-last-response baseline score.
    """
    paired_regular = []
    paired_baseline = []

    diffs = []
    tutor_lengths = []

    for conv_results in regular_convs:
        conv_id = conv_results[0]["conversation_id"]
        if conv_id not in baseline_dict:
            continue

        # Filter to only Yes (1.0) and To Some Extent (0.5)
        informative = [r for r in conv_results if r["guidance"] >= 0.5]
        if not informative:
            continue  # skip conversations where all responses were "No"

        mean_regular = np.mean([r["logprob"] for r in conv_results])
        baseline_score = baseline_dict[conv_id]["logprob"]

        diff = mean_regular - baseline_score
        mean_length = np.mean([r["length"] for r in conv_results])
        diffs.append(diff)
        tutor_lengths.append(mean_length)

        paired_regular.append(mean_regular)
        paired_baseline.append(baseline_score)

    if len(paired_regular) < 2:
        return {"error": "Insufficient paired samples"}
    
    print('####' * 10)
    print("Running Spearman correlation for length vs log-prob difference...")
    rho, p = spearmanr(tutor_lengths, diffs)
    print(f"Length vs log-prob drop: rho={rho:.4f}, p={p:.4f}")

    paired_regular = np.array(paired_regular)
    paired_baseline = np.array(paired_baseline)
    diff = paired_regular - paired_baseline

    stat, p = wilcoxon(diff)
    mean_diff = diff.mean()
    sig, p_corr = annotate_significance(p, n_tests=2)  # only 2 ablation tests

    return {
        "n_pairs": len(paired_regular),
        "mean_regular_logprob": round(paired_regular.mean(), 4),
        "mean_baseline_logprob": round(paired_baseline.mean(), 4),
        "mean_difference": round(mean_diff, 4),
        "wilcoxon_stat": round(stat, 4),
        "p_raw": round(p, 4),
        "p_bonferroni": round(p_corr, 4),
        "significance": sig,
        "direction": "regular > baseline" if mean_diff > 0 else "baseline > regular",
    }


# ============================================================
# Step 7: Ablation — Regular vs Mismatch (paired)
# ============================================================

def ablation_vs_mismatch(
    regular_convs: list,
    mismatch_convs: list,
) -> dict:
    """
    Paired Wilcoxon test: for each conversation, compare mean regular
    log-prob vs mean mismatch log-prob.
    """
    regular_by_id = {
        conv[0]["conversation_id"]: np.mean([r["logprob"] for r in conv])
        for conv in regular_convs
    }
    mismatch_by_id = {
        conv[0]["conversation_id"]: np.mean([r["logprob"] for r in conv])
        for conv in mismatch_convs
        if conv  # guard empty
    }

    shared_ids = sorted(set(regular_by_id) & set(mismatch_by_id))
    if len(shared_ids) < 2:
        return {"error": "Insufficient paired samples"}

    reg_scores = np.array([regular_by_id[cid] for cid in shared_ids])
    mis_scores = np.array([mismatch_by_id[cid] for cid in shared_ids])
    diff = reg_scores - mis_scores

    stat, p = wilcoxon(diff)
    mean_diff = diff.mean()
    sig, p_corr = annotate_significance(p, n_tests=2)

    return {
        "n_pairs": len(shared_ids),
        "mean_regular_logprob": round(reg_scores.mean(), 4),
        "mean_mismatch_logprob": round(mis_scores.mean(), 4),
        "mean_difference": round(mean_diff, 4),
        "wilcoxon_stat": round(stat, 4),
        "p_raw": round(p, 4),
        "p_bonferroni": round(p_corr, 4),
        "significance": sig,
        "direction": "regular > mismatch" if mean_diff > 0 else "mismatch > regular",
    }


# ============================================================
# Step 8: Ordinal Group Comparison (Kruskal-Wallis)
# ============================================================

def kruskal_wallis_by_guidance(all_conversations_results: list) -> dict:
    """
    Groups log-prob scores by guidance label and tests whether
    Yes > To Some Extent > No using Kruskal-Wallis + pairwise Wilcoxon.
    """
    groups = {0.0: [], 0.5: [], 1.0: []}
    for conv in all_conversations_results:
        for r in conv:
            g = r["guidance"]
            if g in groups:
                groups[g].append(r["logprob"])

    group_no   = np.array(groups[0.0])
    group_some = np.array(groups[0.5])
    group_yes  = np.array(groups[1.0])

    if any(len(g) == 0 for g in [group_no, group_some, group_yes]):
        return {"error": "One or more guidance groups are empty"}

    stat, p = kruskal(group_no, group_some, group_yes)
    sig, p_corr = annotate_significance(p, n_tests=1)

    # Pairwise post-hoc (uncorrected — just directional checks)
    def pairwise(a, b, label):
        try:
            s, pv = wilcoxon(
                a[:min(len(a), len(b))],
                b[:min(len(a), len(b))],
            )
            return {"label": label, "stat": round(s, 4), "p": round(pv, 4)}
        except Exception as e:
            return {"label": label, "error": str(e)}

    return {
        "group_sizes": {"No": len(group_no), "To_some_extent": len(group_some), "Yes": len(group_yes)},
        "group_means": {
            "No": round(group_no.mean(), 4),
            "To_some_extent": round(group_some.mean(), 4),
            "Yes": round(group_yes.mean(), 4),
        },
        "kruskal_stat": round(stat, 4),
        "p_raw": round(p, 4),
        "p_bonferroni": round(p_corr, 4),
        "significance": sig,
        "pairwise": [
            pairwise(group_yes, group_some, "Yes vs To-Some-Extent"),
            pairwise(group_yes, group_no,   "Yes vs No"),
            pairwise(group_some, group_no,  "To-Some-Extent vs No"),
        ],
    }


# ============================================================
# Step 9: Main Correlation Analysis (existing, fixed)
# ============================================================

def run_correlation_analysis(all_conversations_results: list) -> dict:
    all_norm_scores, all_guidance, all_relevance, all_lengths = [], [], [], []

    for conv in all_conversations_results:
        for r in conv:
            all_norm_scores.append(r["norm_logprob"])
            all_guidance.append(r["guidance"])
            all_relevance.append(r["relevance"])
            all_lengths.append(r["length"])

    all_norm_scores = np.array(all_norm_scores)
    all_guidance    = np.array(all_guidance)
    all_relevance   = np.array(all_relevance)
    all_lengths     = np.array(all_lengths)

    rho_ped,     p_ped     = spearmanr(all_guidance,  all_norm_scores)
    rho_rel,     p_rel     = spearmanr(all_relevance, all_norm_scores)
    rho_len_perp,p_len_perp= spearmanr(all_lengths,   all_norm_scores)
    rho_ped_len, p_ped_len = spearmanr(all_guidance,  all_lengths)

    def fmt(rho, p_raw):
        sig, p_corr = annotate_significance(p_raw)
        return {
            "rho": round(rho, 4),
            "p_raw": round(p_raw, 4),
            "p_bonferroni": round(p_corr, 4),
            "significance": sig,
        }

    return {
        "n_responses": len(all_norm_scores),
        "pedagogy_signal":          fmt(rho_ped,      p_ped),
        "relevance_signal":         fmt(rho_rel,      p_rel),
        "length_vs_logprob_bias":   fmt(rho_len_perp, p_len_perp),
        "guidance_vs_length_bias":  fmt(rho_ped_len,  p_ped_len),
        "pairwise_accuracy":        round(compute_pairwise_accuracy(all_conversations_results), 4),
    }


# ============================================================
# Step 10: Printing
# ============================================================

def print_section(title: str, data: dict):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    {item}")
        else:
            print(f"  {k}: {v}")


def print_correlation_table(r: dict):
    print(f"\n{'='*65}")
    print(f"  Correlation Analysis  (N = {r['n_responses']} responses)")
    print(f"{'='*65}")
    print(f"{'Metric':<35} {'rho':>7}  {'p (raw)':>9}  {'p (Bonf.)':>10}  {'sig':>5}")
    print(f"{'-'*65}")
    fields = [
        ("Pedagogy signal",        "pedagogy_signal"),
        ("Relevance signal",       "relevance_signal"),
        ("Length vs log-prob",     "length_vs_logprob_bias"),
        ("Guidance vs length",     "guidance_vs_length_bias"),
    ]
    for label, key in fields:
        d = r[key]
        print(
            f"{label:<35} {d['rho']:>7.4f}  {d['p_raw']:>9.4f}  "
            f"{d['p_bonferroni']:>10.4f}  {d['significance']:>5}"
        )
    print(f"{'-'*65}")
    print(f"{'Pairwise accuracy':<35} {r['pairwise_accuracy']:>7.4f}")
    print(f"{'='*65}\n")


# ============================================================
# Entry Point
# ============================================================

def main(args):

    results_dir = Path(args.results_dir)
    guidance_dim = args.guidance_dimension

    # print(f"\nLoading results from: {results_dir}")
    regular_data   = load_json(results_dir / "regular.json")
    mismatch_data  = load_json(results_dir / "mismatch.json")
    no_lr_data     = load_json(results_dir / "no_last_response.json")

    # print(f"Loaded: {len(regular_data)} regular | {len(mismatch_data)} mismatch | {len(no_lr_data)} no-last-response conversations")

    # Preprocess
    regular_convs  = preprocess_regular(regular_data,  guidance_dim)
    mismatch_convs = preprocess_regular(mismatch_data, guidance_dim)
    baseline_dict  = preprocess_no_last_response(no_lr_data)

    # Run analyses
    corr_results   = run_correlation_analysis(regular_convs)
    kw_results     = kruskal_wallis_by_guidance(regular_convs)
    ablation_nlr   = ablation_vs_no_last_response(regular_convs, baseline_dict)
    ablation_mis   = ablation_vs_mismatch(regular_convs, mismatch_convs)

    # Print
    print_correlation_table(corr_results)
    print_section("Kruskal-Wallis: Guidance Quality Groups", kw_results)
    print_section("Ablation: Regular vs No-Last-Response (Wilcoxon paired)", ablation_nlr)
    print_section("Ablation: Regular vs Mismatch (Wilcoxon paired)", ablation_mis)

    # Save
    if args.output:
        output = {
            "guidance_dimension": guidance_dim,
            "correlation_analysis": corr_results,
            "kruskal_wallis": kw_results,
            "ablation_no_last_response": ablation_nlr,
            "ablation_mismatch": ablation_mis,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    
    PATH = [
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/google_gemma-3-12b-it',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/meta-llama_Llama-3.2-3B-Instruct',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/microsoft_Phi-4-reasoning-plus',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/openai_gpt-oss-20b',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8'
]
    # Extract model name from data path
    def extract_model_name_from_path(p):
        return p.split('/')[-1]
    
    guidance_dimension = ["providing_guidance", "actionability"]

    for p in PATH:
        print(f"\n{'#'*80}")
        print("Running results from model: ", extract_model_name_from_path(p))
        print(f"{'#'*80}\n")
        for dim in guidance_dimension:
            print(f"\n{'-'*60}")
            print(f"Analyzing guidance dimension: {dim}")
            print(f"{'-'*60}\n")
            args = argparse.Namespace(results_dir=p, guidance_dimension=dim, output="main_results.json")
            main(args)