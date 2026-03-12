import json
from typing import List, Dict

def extract_critical_fields(
        data, 
        critical_fields: List, 
        dimensions: List
        )-> List[Dict]:
    
    extracted = []
    for item in data:
        filtered_item = {field: item.get(field) for field in critical_fields}

        # Extracting Tutor Responses
        response_dict = item['anno_llm_responses']
        for model_name, model_data in response_dict.items():
            tutor_response = model_data.get('response', None)
            filtered_item[model_name] = {'response': tutor_response}
            
            annotation = model_data.get('annotation', {})
            for dimension in dimensions:
                if dimension in annotation:
                    filtered_item[model_name][dimension] = annotation[dimension]
        
        extracted.append(filtered_item)

    return extracted

if __name__ == "__main__":
    data_file = '../data/MRBench_v2.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    critical_fields = ['conversation_id', 'conversation_history', 'Ground_Truth_Solution']
    dimensions = ['Actionability', 'Providing_Guidance']
    extracted_data = extract_critical_fields(data, critical_fields, dimensions)
    
    # Save file
    with open('current_data_v2.json', 'w') as outfile:
        json.dump(extracted_data, outfile, indent=4)