from collections import Counter
from functools import partial
import random
import re
from pathlib import Path
import pandas as pd
import datasets
from datasets import Features, Value, load_dataset, Dataset

from decontaminate_util import build_ngram_lookup, find_contaminated_questions

CURRENT_FILE = Path(__file__)
PARENT_DATA_CODE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PARENT_DATA_CODE_DIR.parent

def load_fingpt_forecaster():
    dataset_names = ["FinGPT/fingpt-forecaster-dow30-202305-202405",
                     "FinGPT/fingpt-forecaster-sz50-20230201-20240101"]
    all_datasets = []
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        # Try to load all splits
        ds = load_dataset(dataset_name)
        # Convert to list to check available splits
        splits = list(ds.keys())
        split_datasets = []
        
        for split in splits:
            # Extract only query and answer columns
            split_data = ds[split].select_columns(['prompt', 'answer', 'label'])

            # Process the prompt column
            def process_prompt(example):
                # Extract content inside <<SYS>> and remove the tags
                system_prompt_match = re.search(r"<<SYS>>(.*?)<</SYS>>", example["prompt"], re.DOTALL)
                system_prompt = system_prompt_match.group(1).strip() if system_prompt_match else ""

                # Remove [INST], <<SYS>> block, and [/INST] from the prompt
                cleaned_prompt = re.sub(r"\[/?INST\]|\<\<SYS\>.*?\<</SYS\>", "", example["prompt"], flags=re.DOTALL)
                
                # Clean up remaining artifacts and normalize whitespace
                cleaned_prompt = re.sub(r'>\n|\n{2,}', '\n', cleaned_prompt)  # Remove >\n and multiple newlines
                cleaned_prompt = cleaned_prompt.strip()  # Remove leading/trailing whitespace

                return {
                    "system_prompt": system_prompt,
                    "question": cleaned_prompt
                }

            # Apply the processing function
            split_data = split_data.map(process_prompt)
            split_data = split_data.rename_column('answer', 'explanation')
            split_data = split_data.rename_column('label', 'answer')
            split_data = split_data.add_column('dataset', [dataset_name] * len(split_data))
            split_data = split_data.add_column('split', [split] * len(split_data))
            split_data = split_data.add_column('question_type', ['Text'] * len(split_data))
            split_datasets.append(split_data)

        # Combine all splits for this dataset
        if split_datasets:
            combined = datasets.concatenate_datasets(split_datasets)
            all_datasets.append(combined)

    final_dataset = datasets.concatenate_datasets(all_datasets)
    print(f"\nTotal number of examples: {len(final_dataset)}")
    
    # Show distribution of examples across datasets
    dataset_counts = Counter(final_dataset['dataset'])
    print("\nExamples per dataset:")
    for dataset, count in dataset_counts.most_common():
        print(f"{dataset}: {count}")
    
    return final_dataset

def load_quant_trading_instruct():
    dataset_name = "lumalik/Quant-Trading-Instruct"
    ds = load_dataset(dataset_name)
    selected_data = ds['train'].select_columns(['context', 'question', 'answer'])
    selected_data = selected_data.map(lambda x: {'question': f"{x['context']}\n{x['question']}"})
    selected_data = selected_data.add_column('dataset', [dataset_name] * len(selected_data))
    selected_data = selected_data.add_column('question_type', ['Text'] * len(selected_data))
    print(f"\nTotal number of examples: {len(selected_data)}")
    return selected_data

def load_frm_qa100():
    dataset_name = "KirkHan/FRM_QA100"
    ds = load_dataset(dataset_name)
    selected_data = ds['train'].select_columns(['Prompt', 'Answer','Choice_Answer'])
    selected_data = selected_data.rename_column('Prompt', 'question')
    selected_data = selected_data.rename_column('Answer', 'answer')
    selected_data = selected_data.rename_column('Choice_Answer', 'solution')
    # Convert single answer to list format
    def convert_answer_to_list(example):
        return {'answer': [example['answer']]}
    selected_data = selected_data.map(convert_answer_to_list)
    selected_data = selected_data.add_column('dataset', [dataset_name] * len(selected_data))
    selected_data = selected_data.add_column('question_type', ['Choice'] * len(selected_data))
    print(f"\nTotal number of examples: {len(selected_data)}")
    return selected_data

def load_finai_flare():
    # dataset_names = ["TheFinAI/flare-convfinqa",
    #                  "TheFinAI/flare-sm-acl",
    #                  "TheFinAI/flare-sm-cikm",
    #                  "TheFinAI/flare-sm-bigdata",
    #                  "TheFinAI/flare-ma",
    #                  "TheFinAI/flare-cfa",
    #                  "ChanceFocus/flare-finqa"]
    dataset_names = ["TheFinAI/flare-cfa",
                     "ChanceFocus/flare-finqa"]
    
    all_datasets = []
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        # Try to load all splits
        ds = load_dataset(dataset_name)
        # Convert to list to check available splits
        splits = list(ds.keys())
        split_datasets = []
        
        for split in splits:
            # Extract only query and answer columns
            split_data = ds[split].select_columns(['query', 'answer'])
            # Rename columns to match our format
            split_data = split_data.rename_column('query', 'question')
            # Convert answer to list format
            def convert_answer_to_list(example):
                return {'answer': [example['answer']]}
            split_data = split_data.map(convert_answer_to_list)
            # Add metadata
            split_data = split_data.add_column('dataset', [dataset_name] * len(split_data))
            split_data = split_data.add_column('split', [split] * len(split_data))
            # Special handling for flare-cfa
            if dataset_name == "TheFinAI/flare-cfa":
                def extract_solution(example):
                    # Regex to extract choices and map the answer to the explicit text
                    match = re.search(
                        r"CHOICES: A: (?P<A>.+?),B: (?P<B>.+?),C: (?P<C>.+?)\. Answer:",
                        example["question"]
                    )
                    if match:
                        choices = match.groupdict()
                        # Handle answer as list - take first element since CFA has single answers
                        answer = example["answer"][0] if example["answer"] else ""  
                        solution = choices.get(answer, "")  # Map answer to explicit text
                        return {"solution": solution}
                    return {"solution": None}
                
                # Apply the solution extraction
                split_data = split_data.map(extract_solution)
            if dataset_name in ["TheFinAI/flare-sm-acl","TheFinAI/flare-sm-bigdata",
                                "TheFinAI/flare-ma","TheFinAI/flare-cfa"]:
                split_data = split_data.add_column('question_type', ['Choice'] * len(split_data))
            elif dataset_name in ["TheFinAI/flare-convfinqa","ChanceFocus/flare-finqa"]:
                split_data = split_data.add_column('question_type', ['Numeric'] * len(split_data))

            split_datasets.append(split_data)
        
        # Combine all splits for this dataset
        if split_datasets:
            combined = datasets.concatenate_datasets(split_datasets)
            all_datasets.append(combined)

    final_dataset = datasets.concatenate_datasets(all_datasets)
    print(f"\nTotal number of examples: {len(final_dataset)}")
    
    # Show distribution of examples across datasets
    dataset_counts = Counter(final_dataset['dataset'])
    print("\nExamples per dataset:")
    for dataset, count in dataset_counts.most_common():
        print(f"{dataset}: {count}")
    
    return final_dataset

def load_fincorpus_fin_exam():
    dataset_name = "Duxiaoman-DI/FinCorpus"
    split = "train"
    ds = load_dataset(dataset_name,
                        data_files={split: "data/fin_exam.jsonl.gz"},
                        split=split)
    
    def extract_columns(example):
        # Use regex to extract question, options, answer, and explanation
        text = example["text"]
        match = re.match(
            r"""(?P<raw_question>.*?)              # Main question text
            \nA[、．](?P<A>.*?)                    # Option A
            \nB[、．](?P<B>.*?)                    # Option B
            \nC[、．](?P<C>.*?)                    # Option C
            \nD[、．](?P<D>.*?)                    # Option D
            (?:\nE[、．](?P<E>.*?))?               # Optional Option E
            \n答案：(?P<answer>.*?)                # Answer with Chinese colon
            (?:\n分析解释：(?P<explanation>.*?))?   # Optional explanation with Chinese colon
            $                                      # End of string
            """,
            text.strip(),                         # Strip any trailing whitespace
            re.DOTALL | re.VERBOSE
        )

        if match:
            result = match.groupdict()
            
            # Format question with choices
            choices_text = []
            for opt in ['A', 'B', 'C', 'D', 'E']:
                if result.get(opt):
                    choices_text.append(f"{opt}: {result[opt]}\n")
            
            # Combine question and formatted choices
            result['question'] = (
                f"{result['raw_question'].strip()}\n" + 
                '\n'.join([f"{opt}、{result[opt]}" for opt in ['A', 'B', 'C', 'D'] if result.get(opt)])
            )
            
            # Handle multiple answers
            answer = result['answer']
            answer = answer.replace('"', '').replace('[', '').replace(']', '')
            answers = [a.strip() for a in answer.split(',')]
            
            # Get solution text (explicit answers)
            solutions = [result.get(ans, '') for ans in answers]
            result['answer'] = answers
            result['solution'] = '; '.join(filter(None, solutions))
            
            # Remove raw_question as it's now part of formatted question
            del result['raw_question']
            
            return result
        else:
            return {
                "question": None,
                "A": None, "B": None, "C": None, "D": None, "E": None,
                "answer": None, "explanation": None,
                "solution": None
            }

    # Filter for questions longer than 500 characters (assume Chinese characters here)
    def is_long_enough(example, threshold=500):
        return example['question'] and len(example['question']) > threshold

    # Apply the extraction function to the dataset
    processed_ds = ds.map(extract_columns)
    processed_ds = processed_ds.filter(is_long_enough)
    processed_ds = processed_ds.add_column('dataset', [dataset_name] * len(processed_ds))
    processed_ds = processed_ds.add_column('split', [split] * len(processed_ds))
    processed_ds = processed_ds.add_column('question_type', ['Choice'] * len(processed_ds))
    print(f"\nTotal number of examples: {len(processed_ds)}")
    return processed_ds

def load_ideafinbench():
    def get_features(df):
        """Dynamically create features based on DataFrame columns"""
        feature_dict = {
            'id': Value('int32'),
            'question': Value('string'),
            'A': Value('string'),
            'B': Value('string'),
            'C': Value('string'),
            'D': Value('string'),
            'answer': datasets.Sequence(Value('string')),
            'solution': Value('string')
        }
        
        # Add explanation column if it exists in the dataset
        if 'explanation' in df.columns:
            feature_dict['explanation'] = Value('string')
            
        return Features(feature_dict)

    def get_solution(row):
        answers = row['answer']
        # Get corresponding solution text for each answer
        solutions = [row[ans] for ans in answers if ans in row]
        return '; '.join(filter(None, solutions))

    def format_question_with_choices(row):
        """Add choices to the question text"""
        choices = []
        # Get available options (A, B, C, D) that exist and are not empty
        available_options = [opt for opt in ['A', 'B', 'C', 'D'] 
                            if opt in row and pd.notna(row[opt]) and row[opt]]
        
        # Format each available option
        for option in available_options:
            choices.append(f"{option}: {row[option]}")
        
        # Combine original question with formatted choices
        formatted_choices = "\n".join(choices)
        full_question = f"{row['question']}\n{formatted_choices}"
        return full_question

    base_path = PROJECT_ROOT / 'IDEAFinBench' / 'datasets'
    
    # Find all CSV files recursively
    all_datasets = []
    for csv_file in base_path.rglob('*.csv'):
        # Get relative parts of the path for metadata
        rel_parts = csv_file.relative_to(base_path).parts
        exam_type = rel_parts[0]  # e.g., 'cfa_l1'
        # Datasets to skip
        if '_old' in exam_type or '_rag' in exam_type or 'cfa_l1' in exam_type or 'cpa' in exam_type:
            continue
        split = rel_parts[1]  # e.g., 'dev', 'test', 'val'
        subject = csv_file.stem.replace(f'_{split}', '')  # Remove split suffix
        if split== 'test':
            continue
        df = pd.read_csv(csv_file)
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        df = df.reset_index(drop=True)
        df['id'] = df['id'].astype('int32')
        # Convert answer column to list format before creating Dataset
        df['answer'] = df['answer'].apply(lambda x: [a.strip() for a in str(x).split(',')])
        # Format questions to include choices
        df['question'] = df.apply(format_question_with_choices, axis=1)
        df['solution'] = df.apply(get_solution, axis=1)
        features = get_features(df)
        dataset = Dataset.from_pandas(df, features=features)
        
        # Add metadata columns
        dataset = dataset.add_column('dataset', ['IDEAFinBench'] * len(dataset))
        dataset = dataset.add_column('split', [split] * len(dataset))
        dataset = dataset.add_column('subject', [subject] * len(dataset))
        dataset = dataset.add_column('exam_type', [exam_type] * len(dataset))
        dataset = dataset.add_column('question_type', ['Choice'] * len(dataset))
        
        all_datasets.append(dataset)
        print(f"Loaded {csv_file}")
        
    # Combine all datasets
    combined_dataset = datasets.concatenate_datasets(all_datasets)
    print(f"\nTotal number of examples: {len(combined_dataset)}")
    return combined_dataset

def load_fineval():
    dataset_name = "Salesforce/FinEval"
    split = "test"
    selected_subset = ["CFA-Challenge",
                          #"CFA-Easy",
                          "FOMC",
                          "MLESG"]
    all_datasets = []
    for selected_subset in selected_subset:
        print(f"Loading {selected_subset}...")
        ds = load_dataset(dataset_name, selected_subset)
        split_data = ds[split].select_columns(['query', 'answer'])
        split_data = split_data.rename_column('query', 'question')

        # Convert single answer string to list format
        def convert_answer_to_list(example):
            return {'answer': [example['answer']]}  # Wrap in list
        split_data = split_data.map(convert_answer_to_list)
        
        # Add metadata columns

        split_data = split_data.add_column('dataset', [dataset_name] * len(split_data))
        split_data = split_data.add_column('dataset_subset', [selected_subset] * len(split_data))
        split_data = split_data.add_column('split', [split] * len(split_data))
        split_data = split_data.add_column('question_type', ['Choice'] * len(split_data))
        all_datasets.append(split_data)

    final_dataset = datasets.concatenate_datasets(all_datasets)
    print(f"\nTotal number of examples: {len(final_dataset)}")
    
    # Show distribution of examples across datasets
    dataset_subset_counts = Counter(final_dataset['dataset_subset'])
    print("\nExamples per dataset subset:")
    for dataset_subset, count in dataset_subset_counts.most_common():
        print(f"{dataset_subset}: {count}")
    
    return final_dataset

def decontaminate_train_data(train_questions, test_questions, ds, ngram_size=8):    
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    test_lookup = build_ngram_lookup(test_questions, ngram_size)

    # Find contaminated questions
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print(f"\nDecontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(f"Contamination rate: {(len(contaminated_ids)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

DS_TO_SELECTION = {
    # Name: [load function, selection function, #samples]
    'FinAI_flare': [load_finai_flare, None, None],
    'FinCorpus_FinExam': [load_fincorpus_fin_exam, None, None],
    'IDEAFinBench': [load_ideafinbench, None, None],
    'FinEval' : [load_fineval, None, None],
    'FRM_QA100' : [load_frm_qa100, None, None],
}

if __name__ == "__main__":
    random.seed(42)

    ds_all = []
    test_questions = []
    for ds_name, (load_fn, selection_fn, n_samples) in DS_TO_SELECTION.items():
        print(f"Processing {ds_name}...")
        ds = load_fn()
        ds = decontaminate_train_data(ds['question'], test_questions, ds, ngram_size=8)
        if selection_fn:
            ds = selection_fn(ds, n_samples)
        else:
            ds = ds.shuffle(seed=42)
            if n_samples:
                ds = ds.select(range(n_samples))
        test_questions += ds['question']
        ds_all.append(ds)
    ds = datasets.concatenate_datasets(ds_all)
    # Add empty/none cot column
    ds = ds.map(lambda x: {"cot": None, **x})
    # Simple deduplication
    memory = set()
    def is_unique(elem, column, memory):
        if elem[column] in memory: return False
        memory.add(elem[column])
        return True
    # Drop duplicates in `ds` on "col1"
    # import pdb; pdb.set_trace()
    ds = ds.filter(partial(is_unique, column="question", memory=memory))
    ds.push_to_hub(repo_id="akftam/financial-qa-s1decontaminate-v1.0")