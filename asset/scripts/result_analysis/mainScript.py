from result_analysis_v3 import run_full_analysis

import json

path = [
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/google_gemma-3-12b-it',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/meta-llama_Llama-3.2-3B-Instruct',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/microsoft_Phi-4-reasoning-plus',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/openai_gpt-oss-20b',
    #'/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8'
]

# Find the model name from the path
def extract_model_name_from_path(p):
    return p.split('/')[-1]

# Loop through each path and process the data
full_result = {}
guidance_dimension = ["providing_guidance", "actionability"]

for p in path:
    # Extract model name from data path
    model_name = extract_model_name_from_path(p)

    # Load all three categories of data for the current model
    with open(f'{p}/regular.json', 'r') as f:
        data = json.load(f) 

    with open(f'{p}/mismatch.json', 'r') as f:
        mismatch_data = json.load(f)
    
    with open(f'{p}/no_last_response.json', 'r') as f:
        null_tutor = json.load(f)
    
    # Iterate over both guidance dimensions for analysis
    for dimension in guidance_dimension:
        reg_results = run_full_analysis(data, dimension)
        mismatch_results = run_full_analysis(mismatch_data, dimension)  
        null_results = run_full_analysis(null_tutor, dimension)

        # Store results in a structured format for each model and guidance dimension
        results = {
            "regular": reg_results,
            "mismatch": mismatch_results,
            "null_tutor": null_results
        }

        if model_name not in full_result:
            full_result[model_name] = {}
        
        full_result[model_name][dimension] = results

# Save the full results to a JSON file for later analysis
with open('full_analysis_results.json', 'w') as f:
    json.dump(full_result, f, indent=4)
