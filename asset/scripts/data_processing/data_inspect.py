# This file is to understand how many conversation thread has ground truth solutions
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def count_ground_truth_solutions(data):
    count = 0
    for item in data:
        if 'Ground_Truth_Solution' in item and item['Ground_Truth_Solution'] != "Not Available":
            count += 1
    return count

def analyze_variations(data):
    count_both_vary = 0
    count_actionability_vary = 0
    count_guidance_vary = 0
    
    ids_both_vary = []

    for item in data:
        responses = item.get("anno_llm_responses", {})
        if not responses:
            continue
            
        actionabilities = []
        guidances = []
        
        for model_name, model_data in responses.items():
            annotation = model_data.get("annotation", {})
            
            if "Actionability" in annotation:
                actionabilities.append(annotation["Actionability"])
            if "Providing_Guidance" in annotation:
                guidances.append(annotation["Providing_Guidance"])
                
        if not actionabilities or not guidances:
            continue

        unique_act = set(actionabilities)
        unique_guid = set(guidances)
        
        act_varies = len(unique_act) > 1
        guid_varies = len(unique_guid) > 1
        
        if act_varies:
            count_actionability_vary += 1
        if guid_varies:
            count_guidance_vary += 1
            
        if act_varies and guid_varies:
            count_both_vary += 1
            ids_both_vary.append(item.get("conversation_id"))
            
    return count_actionability_vary, count_guidance_vary, count_both_vary

if __name__ == "__main__":
    data_file = '../data/MRBench_v2.json'
    data = load_json(data_file)
    total_conversations = len(data)
    conversations_with_solutions = count_ground_truth_solutions(data)
    
    print(f"Total Conversations: {total_conversations}")
    print(f"Conversations with Ground Truth Solutions: {conversations_with_solutions}")

    actionability_vary, guidance_vary, both_vary = analyze_variations(data)
    print(f"Conversations where Actionability varies: {actionability_vary}")
    print(f"Conversations where Providing Guidance varies: {guidance_vary}")
    print(f"Conversations where both vary: {both_vary}")