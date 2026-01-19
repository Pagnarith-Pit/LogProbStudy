# Load json final_questions_only.json
import json

with open('Dataset/final_questions_only.json', 'r') as f:
    correct = json.load(f)[0]

# Load full_question_solution_latest.json
with open('Dataset/full_question_solution_latest.json', 'r') as f:
    full_data = json.load(f)[0]

# For each element in full_data, change the "solution" key to the value of the "solution" key in correct
for i, j in zip(full_data, correct):
    if i == j:
        full_data[i]['solution'] = correct[j]['solution']

# Save the full_data to a new json file
with open('Dataset/full_question_solution_latest_corrected.json', 'w') as f:
    json.dump(full_data, f, indent=4)