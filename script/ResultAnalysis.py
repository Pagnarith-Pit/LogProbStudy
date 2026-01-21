# Opening the results file
import json
import pingouin

# Change the path to the desired result file for correct analysis
with open('../Result/log_prob_score_final_question_codeLlama.json', 'r') as file:
    results = json.load(file)

for key, values in results.items():
    for score_type in values:
        values[score_type] = abs(values[score_type])

def compare(a, b):
    return 1 if a > b else 0

def compute_accuracy(data):
    # Result dictionary to store comparison results
    result = {}

    # Counters for total and correct comparisons
    least_more = []
    more_most = []
    least_most = []
    total_comparisons = 0


    # Iterate through the data
    for key, values in data.items():
        comparisons = {}
        # check = 0

        # Perform comparisons if the keys exist
        if 'least' in values and 'more' in values:
            res = compare(values['least'], values['more'])
            comparisons['least_vs_more'] = res
            total_comparisons += 1
            least_more.append(res)

            # if not res:
            #     check += 1
            

        if 'more' in values and 'most' in values:
            res = compare(values['more'], values['most'])
            comparisons['more_vs_most'] = res
            total_comparisons += 1
            more_most.append(res)

            # if not res:
            #     check += 1

        if 'least' in values and 'most' in values:
            # if check % 2 == 0:
            #     continue
            res = compare(values['least'], values['most'])
            comparisons['least_vs_most'] = res
            total_comparisons += 1
            least_most.append(res)
        
        correct_comparisons = sum(least_more) + sum(more_most) + sum(least_most)
        result[key] = comparisons

    # Calculate percentage of correct comparisons
    accuracy_percentage = (correct_comparisons / total_comparisons) * 100 if total_comparisons > 0 else 0

    # Display the results
    # print("Comparison Results:", result)
    print("\n")
    print("Results for CodeLlama - Next Question Only")
    print("Unique conversation:", len(data))
    print("Total Comparisons:", total_comparisons, "\n")

    print(f"Overrall Accuracy: {accuracy_percentage:.2f}% ({correct_comparisons}/{total_comparisons} correct)")
    print("Least vs More Accuracy: ", sum(least_more)/len(least_more))
    print("More vs Most Accuracy: ", sum(more_most)/len(more_most))
    print("Least vs Most Accuracy: ", sum(least_most)/len(least_most))
    print("\n")

compute_accuracy(results)

def perform_wilcoxon_test(data):
    # Collect "least" and "most" scores
    least_scores = []
    more_scores = []
    most_scores = []

    for key, values in data.items():
        if 'least' in values and 'most' in values:
            least_scores.append(values['least'])
            most_scores.append(values['most'])
            
        if 'least' in values and 'more' in values:
            more_scores.append(values['more'])
 
    # Perform Wilcoxon Signed-Rank Test
    print("Least with More Test:")
    print("-" * 70)
    stat_more = pingouin.wilcoxon(more_scores, least_scores, alternative='less')
    print(stat_more)

    print("\n" * 3)

    print("Least with Most Test:")
    print("-" * 70)        
    stat_most = pingouin.wilcoxon(most_scores, least_scores, alternative='less')
    print(stat_most)

perform_wilcoxon_test(results)

# def sort_by_least_most_difference(data):
#     # Create list of tuples (key, difference)
#     diff_pairs = []
    
#     for key, values in data.items():
#         if 'least' in values and 'most' in values:
#             diff = abs(values['least']) - abs(values['most'])
#             diff_pairs.append((key, diff, values['least'], values['most']))
    
#     # Sort by difference (smallest to largest)
#     sorted_pairs = sorted(diff_pairs, key=lambda x: x[1])
    
#     # Print sorted results
#     print("\nExamples sorted by least-most difference (smallest to largest):")
#     print("-" * 70)
#     print(f"{'Key':<30} {'Least':<10} {'Most':<10} {'Difference':<10}")
#     print("-" * 70)
    
#     for key, diff, least, most in sorted_pairs:
#         print(f"{key:<30} {least:<10.5f} {most:<10.5f} {diff:<10.5f}")
    
#     # Return sorted keys in case you need them
#     return [pair[0] for pair in sorted_pairs]

# # Call the function after your existing code
# sorted_keys = sort_by_least_most_difference(results)