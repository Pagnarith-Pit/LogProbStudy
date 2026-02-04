import argparse
import json
import torch
import random
from pathlib import Path
from typing import Any, Dict, Optional
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
    model,
    tokenizer,
    prompt: str,
    conversation_history: str,
    model_response: str,
    ground_truth_solution: str,
) -> Dict[str, Any]:
    """
    Compute log probability that a model would generate `ground_truth_solution`.
    """
    context = (
        "System Prompt:\n"
        f"{prompt.strip()}\n\n"
        "Conversation_History:\n"
        f"{conversation_history.strip()}\n\n"
        "Last_Tutor_Response:\n"
        f"{model_response.strip()}\n\n"
        "Ground_Truth_Solution:\n"
    )

    context_ids = tokenizer(context, return_tensors="pt").input_ids.to("cuda")
    target_ids = tokenizer(ground_truth_solution, return_tensors="pt").input_ids.to(
        "cuda"
    )

    # Concatenate context and target
    input_ids = torch.cat([context_ids, target_ids], dim=1)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift logits to align with targets
    log_probs = torch.log_softmax(logits, dim=-1)

    target_len = target_ids.shape[1]
    start_index = input_ids.shape[1] - target_len

    target_token_log_probs = []
    for i in range(target_len):
        token_id = target_ids[0, i]
        # We look at the logit of the token BEFORE the target token to predict the target
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


if __name__ == "__main__":
    # Updated 'noice' to 'novice' for better persona acting
    PROMPT = """You are a novice **student** asking your tutor for help on different problems. Your goal is to respond naturally to the tutor's last message. You may reflect, reason, ask questions, or even provide an answer if that feels like what a real student would do. Base your response **only on the Last_Tutor_Response and your current understanding**. Instructions:
1. Read the tutor's last message carefully.
2. Respond as a student would, showing your **thought process or understanding**.
3. Your response can include reasoning, clarifications, or a solution — whatever a real student might say next.
4. Keep your response concise but realistic.
5. Ensure your response is relevant to the topic.
**Now, respond as a student**."""

    parser = argparse.ArgumentParser(description="Compute log-probability scores.")
    parser.add_argument(
        "-m", "--model", dest="model_name", required=True, help="HF model name"
    )
    parser.add_argument(
        "-a", "--ablation", action="store_true", help="Perform ablation"
    )

    args = parser.parse_args()
    model_name = args.model_name

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    model.eval()

    model_tag = sanitize_model_name(model_name)
    ablation_tag = "ablation" if args.ablation else "regular"
    perform_ablation = args.ablation

    data_path = Path(
        "/home/ppit/punim2402/IndirectScore/asset/data/ReadyForLogProb.json"
    )
    output_dir = (
        Path("/home/ppit/punim2402/IndirectScore/asset/data/logprob_results")
        / model_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ablation_tag}.json"

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(0)
    results = []
    processed = 0

    for item in data:
        conversation_id = item.get("conversation_id")
        conversation_history = item.get("conversation_history", "")
        original_ground_truth = item.get("Ground_Truth_Solution", "")

        if (
            not conversation_id
            or not original_ground_truth
            or original_ground_truth.strip() == "Not Available"
        ):
            continue

        ground_truth_solution = original_ground_truth
        if perform_ablation:
            ground_truth_solution = sample_irrelevant_ground_truth(
                conversation_id, data
            )

        entry_result = {
            "conversation_id": conversation_id,
            "ablation": perform_ablation,
            "models": {},
        }

        for model_key, model_obj in item.items():
            if model_key in {
                "conversation_id",
                "conversation_history",
                "Ground_Truth_Solution",
            }:
                continue
            if not isinstance(model_obj, dict):
                continue

            model_response = model_obj.get("response")
            if not model_response:
                continue

            scores = logProbCalculation(
                model=model,
                tokenizer=tokenizer,
                prompt=PROMPT,
                conversation_history=conversation_history,
                model_response=model_response,
                ground_truth_solution=ground_truth_solution,
            )

            entry_result["models"][model_key] = {
                "response": model_response,
                "relevance_F1": model_obj.get("relevance_F1"),
                "providing_guidance": model_obj.get("Providing_Guidance"),
                "actionability": model_obj.get("Actionability"),
                **scores,
            }

        if entry_result["models"]:
            results.append(entry_result)
            processed += 1
            if processed % 10 == 0:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Processed {processed} items.", end="\r")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nFinished. Saved {len(results)} results to {output_path}")
