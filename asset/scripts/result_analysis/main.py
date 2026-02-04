from result_analysis import run_full_analysis

import json

path = [
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results/google_gemma-3-12b-it',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results/meta-llama_Llama-3.2-3B-Instruct',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results/microsoft_Phi-4-reasoning-plus',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results/openai_gpt-oss-20b',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8'
]

# Find the model name from the path
def extract_model_name_from_path(p):
    return p.split('/')[-1]

# Loop through each path and process the data
full_result = {}
for p in path:
    with open(f'{p}/regular.json', 'r') as f:
        data = json.load(f) 

    with open(f'{p}/ablation.json', 'r') as f:
        ablation_data = json.load(f)

    reg_results = run_full_analysis(data)
    ablation_results = run_full_analysis(ablation_data)
    
    results = {
        "regular": reg_results,
        "ablation": ablation_results
    }

    model_name = extract_model_name_from_path(p)
    full_result[model_name] = results

# Save the combined results to a JSON file
with open("actionability_corr.json", "w") as outfile:
    json.dump(full_result, outfile, indent=4)
