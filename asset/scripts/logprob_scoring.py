import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def sample_irrelevant_ground_truth(
    current_conversation_id: str,
    all_items: list,
) -> str:
    """
    Sample a ground-truth solution from a different conversation.
    """
    candidates = [
        item["Ground_Truth_Solution"]
        for item in all_items
        if item.get("conversation_id") != current_conversation_id
        and item.get("Ground_Truth_Solution")
        and item["Ground_Truth_Solution"].strip() != "Not Available"
    ]

    if not candidates:
        raise ValueError("No valid candidates for ground-truth ablation.")

    return random.choice(candidates)

def sanitize_model_name(model_name: str) -> str:
	return model_name.replace("/", "_")

def logProbCalculation(
	prompt: str,
	conversation_history: str,
	model_response: str,
	ground_truth_solution: str,
	model_name: str,
	device: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Compute the log probability that a pretrained model would generate
	`ground_truth_solution` given the provided context.

	Context is constructed from:
	- prompt
	- conversation_history
	- model_response

	Returns total log-probability, average log-probability per token, and
	perplexity based on the ground truth tokens.
	"""

	context = (
		"Prompt:\n"
		f"{prompt.strip()}\n\n"
		"Conversation_History:\n"
		f"{conversation_history.strip()}\n\n"
		"Model_Response:\n"
		f"{model_response.strip()}\n\n"
		"Ground_Truth_Solution:\n"
	)

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	model.to(device)
	model.eval()

	context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
	target_ids = tokenizer(ground_truth_solution, return_tensors="pt").input_ids.to(device)

	input_ids = torch.cat([context_ids, target_ids], dim=1)

	with torch.no_grad():
		outputs = model(input_ids)
		logits = outputs.logits

	log_probs = torch.log_softmax(logits, dim=-1)

	target_len = target_ids.shape[1]
	start_index = input_ids.shape[1] - target_len

	target_token_log_probs = []
	for i in range(target_len):
		token_id = target_ids[0, i]
		token_log_prob = log_probs[0, start_index + i - 1, token_id]
		target_token_log_probs.append(token_log_prob)

	target_token_log_probs_tensor = torch.stack(target_token_log_probs)
	total_log_prob = target_token_log_probs_tensor.sum().item()
	avg_log_prob = target_token_log_probs_tensor.mean().item()
	perplexity = float(torch.exp(-target_token_log_probs_tensor.mean()).item())

	return {
		"total_log_prob": total_log_prob,
		"avg_log_prob": avg_log_prob,
		"perplexity": perplexity,
	}

from mpi4py import MPI
if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()        # Task ID
    size = comm.Get_size()        # Total number of MPI tasks

    PROMPT = """You are an AI assistant helping with problem solving. Given a conversation history and a model's response, evaluate how well the model's response leads to the ground truth solution."""
    
    parser = argparse.ArgumentParser(description="Compute log-probability scores (MPI-enabled).")
    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        required=True,
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "-a",
        "--ablation",
        action="store_true",
        help="Perform target-alignment ablation by replacing the ground truth solution.",
    )

    args = parser.parse_args()
    model_name = args.model_name
    model_tag = sanitize_model_name(model_name)
    ablation_tag = "ablation" if args.ablation else "regular"
    perform_ablation = args.ablation

    # Data paths
    data_path = Path(__file__).resolve().parents[1] / "data" / "ReadyForLogProb.json"
    output_dir = Path(__file__).resolve().parents[1] / "data" / "logprob_results" / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each rank writes its own file to avoid conflicts
    output_path = output_dir / f"{ablation_tag}_rank{rank}.json"

    # Load full dataset once
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Split data evenly across ranks
    n = len(data)
    chunk_size = (n + size - 1) // size  # ceil division to cover all data
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, n)
    data_chunk = data[start_idx:end_idx]

    # For reproducibility in ablation
    random.seed(rank)

    results = []
    processed = 0

    for item in data_chunk:
        conversation_id = item.get("conversation_id")
        conversation_history = item.get("conversation_history", "")
        original_ground_truth = item.get("Ground_Truth_Solution", "")

        if not conversation_id or not original_ground_truth or original_ground_truth.strip() == "Not Available":
            continue

        # ---- Ground-truth selection (normal vs ablated) ----
        if perform_ablation:
            ground_truth_solution = sample_irrelevant_ground_truth(
                current_conversation_id=conversation_id,
                all_items=data,
            )
        else:
            ground_truth_solution = original_ground_truth

        entry_result: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "ablation": perform_ablation,
            "models": {},
        }

        for model_key, model_obj in item.items():
            if model_key in {"conversation_id", "conversation_history", "Ground_Truth_Solution"}:
                continue
            if not isinstance(model_obj, dict):
                continue

            model_response = model_obj.get("response")
            if not model_response:
                continue

            result = logProbCalculation(
                prompt=PROMPT,
                conversation_history=conversation_history,
                model_response=model_response,
                ground_truth_solution=ground_truth_solution,
                model_name=model_name,
                device=f"cuda:{rank % torch.cuda.device_count()}",  # assign GPU per rank
            )

            entry_result["models"][model_key] = {
                "response": model_response,
                "relevance_F1": model_obj.get("relevance_F1"),
                "providing_guidance": model_obj.get("Providing_Guidance"),
                "actionability": model_obj.get("Actionability"),
                **result,
            }

        if entry_result["models"]:
            results.append(entry_result)
            processed += 1

    # Write results for this rank
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"[Rank {rank}] Processed {len(data_chunk)} items, "
        f"saved {len(results)} conversation results to {output_path}"
    )
