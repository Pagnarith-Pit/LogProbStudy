from result_analysis_v2 import run_full_analysis

import json

path = [
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/google_gemma-3-12b-it',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/meta-llama_Llama-3.2-3B-Instruct',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/microsoft_Phi-4-reasoning-plus',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/openai_gpt-oss-20b',
    '/Users/ppit/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/Desktop/IndirectScore/asset/data/logprob_results_v2/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8'
]

# Find the model name from the path
def extract_model_name_from_path(p):
    return p.split('/')[-1]

# Loop through each path and process the data
full_result = {}
guidance_dimension = ["providing_guidance", "actionability"]

for p in path:
    with open(f'{p}/regular.json', 'r') as f:
        data = json.load(f) 

    with open(f'{p}/mismatch.json', 'r') as f:
        mismatch_data = json.load(f)
    
    with open(f'{p}/no_last_response.json', 'r') as f:
        null_tutor = json.load(f)
    
    for dimension in guidance_dimension:
        reg_results = run_full_analysis(data, dimension)
        mismatch_results = run_full_analysis(mismatch_data, dimension)  # Use the first dimension for mismatch results
        null_results = run_full_analysis(null_tutor, dimension)  # Use the first dimension for null tutor results

    # results = {
    #     "regular": reg_results,
    #     "mismatch": mismatch_results
    # }

    # model_name = extract_model_name_from_path(p)
    # full_result[model_name] = results


