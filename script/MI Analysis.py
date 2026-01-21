import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

def extract_last_teacher_questions(data):
    # Initialize lists to store questions by category
    last_question = []
    
    # Process the data
    for _ , problem_data in data[0].items():
        # Extract from each category if available
        new_dict = {'solution': problem_data['solution']}

        for category in ["least", "more", "most"]:
            if category in problem_data:
                dialogue = problem_data[category]
                
                # Find the last teacher's question
                lines = dialogue.split('\n')
                teacher_lines = [line for line in lines if line.startswith('Teacher: ')]
                
                if teacher_lines:
                    new_dict[category] = teacher_lines[-1]
                
        # Append the new dictionary to the list
        last_question.append(new_dict)
    
    return last_question
    
# Load the JSON data
with open('full_question_solution_latest.json', 'r') as f:
    data = json.load(f)

# Extract and group the questions
grouped_questions = extract_last_teacher_questions(data)


def prepare_corpus_and_labels(data):
    corpus = []
    labels = []
    for label, texts in data.items():
        corpus.extend(texts)
        labels.extend([label] * len(texts))
    
    # Vectorize the corpus
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=1)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Compute the mutual information scores
    results = {}
    for cls in data:
        y_bin = np.array([1 if lbl == cls else 0 for lbl in labels])
        mi_scores = mutual_info_classif(X, y_bin, discrete_features=True)
        freq_pos = X[y_bin == 1].sum(axis=0).A1 / y_bin.sum()
        freq_neg = X[y_bin == 0].sum(axis=0).A1 / (len(labels) - y_bin.sum())
        df = pd.DataFrame({
            'feature': feature_names,
            'mi': mi_scores,
            'freq_in_pos': freq_pos,
            'freq_in_neg': freq_neg
        })

        # Keep only those present in-class
        # To see how "explain" is contrastive to "most", remove this line
        df = df[df['freq_in_pos'] > 0].copy()
        df.sort_values('mi', ascending=False, inplace=True)
        results[cls] = df


    for cls, df in results.items():

        # Change cls from "least" to "Best", from "more" to "Medium", and from "most" to "Worst"
        if cls == "least":
            cls = "Best"
        elif cls == "more":
            cls = "Medium"
        elif cls == "most": 
            cls = "Worst"
        print(f"\nTop 10 features for '{cls}':")
        print(df[['feature','mi','freq_in_pos']].head(10).to_string(index=False))

prepare_corpus_and_labels(grouped_questions)