import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


DEFAULT_OUTPUT_NAME = "Style Analysis.json"
DEFAULT_GROUND_TRUTH_PATH = Path(
    # Add path to ground truth json file -  ReadyForLogProb.json"
)


def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def jaccard_distance(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return 1.0 - len(intersection) / len(union)


def length_ratio(text_a: str, text_b: str) -> float:
    len_a = max(len(text_a.split()), 1)
    len_b = max(len(text_b.split()), 1)
    return abs(np.log(len_a / len_b))


# Load and build paired records
def _load_ground_truth_index() -> dict:
    gt_data = load_json(DEFAULT_GROUND_TRUTH_PATH)
    return {
        conv.get("conversation_id"): conv.get("Ground_Truth_Solution", "")
        for conv in gt_data
        if conv.get("conversation_id") is not None
    }


GROUND_TRUTH_BY_ID = _load_ground_truth_index()


def retrieveGroundTruthById(conv_id: str) -> str:
    return GROUND_TRUTH_BY_ID.get(conv_id, "")

def build_pairs(regular_data: list, no_lr_data: list) -> list:
    # Index no-last-response by conversation_id
    # All models share the same score per conversation, so just take first
    baseline_by_id = {}
    for conv in no_lr_data:
        cid = conv.get("conversation_id")
        models = conv.get("models", {})
        if models:
            first = next(iter(models.values()))
            baseline_by_id[cid] = first.get("avg_log_prob", None)

    pairs = []
    for conv in regular_data:
        cid = conv.get("conversation_id")
        ground_truth = retrieveGroundTruthById(cid)
        baseline_lp = baseline_by_id.get(cid)

        if baseline_lp is None or not ground_truth:
            continue

        for model_name, metrics in conv.get("models", {}).items():
            tutor_response = metrics.get("response", "")
            regular_lp = metrics.get("avg_log_prob")

            if not tutor_response or regular_lp is None:
                continue

            pairs.append({
                "conversation_id": cid,
                "model": model_name,
                "tutor_response": tutor_response,
                "ground_truth": ground_truth,
                "logprob_diff": regular_lp - baseline_lp,  # negative = baseline wins
                "regular_lp": regular_lp,
                "baseline_lp": baseline_lp,
            })

    return pairs

def add_lexical_distances(pairs: list) -> list:
    for p in pairs:
        p["jaccard_dist"] = jaccard_distance(p["tutor_response"], p["ground_truth"])
        p["length_ratio"] = length_ratio(p["tutor_response"], p["ground_truth"])
    return pairs


def add_semantic_distances(pairs: list, sbert_model, batch_size: int) -> list:
    print(f"  Computing SBERT embeddings for {len(pairs)} pairs (batch_size={batch_size})...")

    tutor_texts = [p["tutor_response"] for p in pairs]
    gt_texts = [p["ground_truth"] for p in pairs]
    all_texts = tutor_texts + gt_texts

    all_embs = sbert_model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    split_index = len(pairs)
    tutor_embs = all_embs[:split_index]
    gt_embs = all_embs[split_index:]

    # Cosine similarity row-wise, then convert to distance
    sims = (tutor_embs * gt_embs).sum(axis=1) 
    for i, p in enumerate(pairs):
        p["sbert_cos_dist"] = float(1.0 - sims[i])

    return pairs

def run_correlations(pairs: list, label: str) -> dict:
    diffs    = np.array([p["logprob_diff"]   for p in pairs])
    jaccard  = np.array([p["jaccard_dist"]   for p in pairs])
    length_r = np.array([p["length_ratio"]   for p in pairs])
    sbert    = np.array([p["sbert_cos_dist"] for p in pairs])

    def fmt(rho, p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        return {"rho": round(float(rho), 4), "p": round(float(p), 4), "sig": stars}

    rho_j, p_j = spearmanr(jaccard,  diffs)
    rho_l, p_l = spearmanr(length_r, diffs)
    rho_s, p_s = spearmanr(sbert,    diffs)

    result = {
        "model": label,
        "n_pairs": len(pairs),
        "mean_logprob_diff": round(float(diffs.mean()), 4),
        "jaccard_vs_diff":   fmt(rho_j, p_j),
        "length_ratio_vs_diff": fmt(rho_l, p_l),
        "sbert_cos_dist_vs_diff": fmt(rho_s, p_s),
    }

    return result


def resolve_device(requested_device: str) -> str:
    device = requested_device.lower()
    if device != "auto":
        return device

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_batch_size(requested_batch_size: int | None, device: str) -> int:
    if requested_batch_size is not None:
        return requested_batch_size

    if device == "mps":
        return 128
    if device == "cuda":
        return 256
    return 64


def load_sbert_model(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    if device == "mps":
        model = model.half()
    return model


def resolve_datasets(args) -> list:
    datasets = []

    if args.data_paths:
        for raw_path in args.data_paths:
            data_path = Path(raw_path).expanduser().resolve()
            regular_path = data_path / "regular.json"
            no_lr_path = data_path / "no_last_response.json"

            if not regular_path.is_file():
                raise FileNotFoundError(f"regular.json not found in {data_path}")
            if not no_lr_path.is_file():
                raise FileNotFoundError(f"no_last_response.json not found in {data_path}")

            datasets.append({
                "label": data_path.name,
                "data_path": str(data_path),
                "regular_path": regular_path,
                "no_lr_path": no_lr_path,
            })

        return datasets

    if not args.regular or not args.no_lr:
        raise ValueError("Need --data-path or both --regular and --no-last-response.")

    regular_path = Path(args.regular).expanduser().resolve()
    no_lr_path = Path(args.no_lr).expanduser().resolve()
    label = args.label or regular_path.parent.name or "dataset"

    datasets.append({
        "label": label,
        "data_path": str(regular_path.parent),
        "regular_path": regular_path,
        "no_lr_path": no_lr_path,
    })
    return datasets


def analyze_dataset(
    dataset: dict,
    sbert,
    per_model: bool,
    batch_size: int,
    dataset_index: int,
    total_datasets: int,
) -> dict:
    label = dataset["label"]
    regular_path = dataset["regular_path"]
    no_lr_path = dataset["no_lr_path"]

    print(f"\n[{dataset_index}/{total_datasets}] Starting dataset: {label}")
    print(f"  Source: {dataset['data_path']}")

    print(f"\nLoading data for {label}...")
    regular_data = load_json(regular_path)
    no_lr_data = load_json(no_lr_path)
    print(f"  Regular: {len(regular_data)} conversations")
    print(f"  No-last-response: {len(no_lr_data)} conversations")

    print(f"\nBuilding paired records for {label}...")
    pairs = build_pairs(regular_data, no_lr_data)
    print(f"  {len(pairs)} (conversation, model) pairs built")

    if not pairs:
        raise ValueError(f"No valid pairs found for dataset: {label}")

    print(f"\nComputing lexical distances for {label}...")
    pairs = add_lexical_distances(pairs)
    pairs = add_semantic_distances(pairs, sbert, batch_size)

    results = {
        "dataset": label,
        "data_path": dataset["data_path"],
        "overall": run_correlations(pairs, label=label),
    }

    if per_model:
        from itertools import groupby

        print(f"\n[{dataset_index}/{total_datasets}] Running per-model breakdown for {label}...")
        sorted_pairs = sorted(pairs, key=lambda x: x["model"])
        per_model_results = {}
        for model_name, group in groupby(sorted_pairs, key=lambda x: x["model"]):
            model_pairs = list(group)
            per_model_results[model_name] = run_correlations(model_pairs, label=model_name)
        results["per_model"] = per_model_results

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stylistic anchoring diagnostic.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        dest="data_paths",
        default=None,
        help="One or more directories containing regular.json and no_last_response.json",
    )
    parser.add_argument("--regular", default=None, help="Path to regular.json")
    parser.add_argument(
        "--no-last-response",
        default=None,
        dest="no_lr",
        help="Path to no_last_response.json",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label when using --regular and --no-last-response directly",
    )
    parser.add_argument("--sbert-model",      default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device for SentenceTransformer inference (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size; defaults are tuned by device",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Path to save results as JSON (default: {DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument("--per-model",        action="store_true",
                        help="Also break down results per tutor model")
    return parser


def main(args: argparse.Namespace) -> dict:
    run_start = time.perf_counter()

    datasets = resolve_datasets(args)
    total_datasets = len(datasets)

    device = resolve_device(args.device)
    batch_size = resolve_batch_size(args.batch_size, device)

    # Attempt 15: Code hanged here. Print to debug 
    print(f"\nLoading SBERT model: {args.sbert_model}")
    sbert = load_sbert_model(args.sbert_model, device)

    all_results = {}
    for dataset_index, dataset in enumerate(datasets, start=1):
        all_results[dataset["label"]] = analyze_dataset(
            dataset,
            sbert,
            args.per_model,
            batch_size,
            dataset_index,
            total_datasets,
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        payload = next(iter(all_results.values())) if len(all_results) == 1 else all_results
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    total_elapsed = time.perf_counter() - run_start
    print(f"\nRun complete: {total_datasets} dataset(s) processed in {format_duration(total_elapsed)}")

    return all_results


if __name__ == "__main__":
    default_data_paths = [
        # Add path to directories containing regular.json and no_last_response.json
        # Removed since we used absolute path, which may reveal name during review process
    ]

    use_namespace_defaults = True # Set to False to use command-line arguments instead

    if use_namespace_defaults:
        args = argparse.Namespace(
            data_paths=default_data_paths,
            regular=None,
            no_lr=None,
            label=None,
            sbert_model="all-MiniLM-L6-v2",
            device="auto",
            batch_size=None,
            output=DEFAULT_OUTPUT_NAME,
            per_model=False,
        )
    else:
        args = build_parser().parse_args()

    main(args)
