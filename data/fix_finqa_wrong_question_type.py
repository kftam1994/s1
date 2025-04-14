from datasets import load_dataset

# akftam/financial-qa-s1decontaminate-filtered-v1.0
# akftam/financial-qa-filtered-v1.0-deepseek
# akftam/financial-qa-s1decontaminate-v1.0
dataset_name = "akftam/financial-qa-filtered-v1.0-deepseek"
# Load the dataset

ds = load_dataset(dataset_name)

# Print initial statistics
print("Initial statistics:")
flare_finqa_ds = ds['train'].filter(lambda x: x['dataset'] == "ChanceFocus/flare-finqa")
initial_types = {}
yes_no_count = 0
for example in flare_finqa_ds:
    qtype = example['question_type']
    initial_types[qtype] = initial_types.get(qtype, 0) + 1
    if example['answer'][0].lower() in ['yes', 'no']:
        yes_no_count += 1

print("\nInitial question type distribution:")
for qtype, count in initial_types.items():
    print(f"{qtype}: {count}")
print(f"Total yes/no answers found: {yes_no_count}")

# Function to update question type for yes/no answers
def update_question_type(example):
    if example['dataset'] == "ChanceFocus/flare-finqa":
        answer = example['answer'][0].lower() if example['answer'] else ''
        if answer in ['yes', 'no']:
            if example['question_type'] != 'Choice':
                print(f"Converting question type from {example['question_type']} to Choice for answer: {answer}")
                print(f"Question: {example['question']}")
                return {'question_type': 'Choice'}
    return {'question_type': example['question_type']}

# Apply the update function
ds['train'] = ds['train'].map(update_question_type)

# Print final statistics
print("\nFinal statistics:")
flare_finqa_ds = ds['train'].filter(lambda x: x['dataset'] == "ChanceFocus/flare-finqa")
final_types = {}
for example in flare_finqa_ds:
    qtype = example['question_type']
    final_types[qtype] = final_types.get(qtype, 0) + 1

print("\nFinal question type distribution:")
for qtype, count in final_types.items():
    print(f"{qtype}: {count}")

# Print changes summary
print("\nChanges summary:")
for qtype in set(initial_types.keys()) | set(final_types.keys()):
    initial = initial_types.get(qtype, 0)
    final = final_types.get(qtype, 0)
    if initial != final:
        print(f"{qtype}: {initial} -> {final} (Change: {final - initial})")

# Push updated dataset to hub
ds.push_to_hub(dataset_name)