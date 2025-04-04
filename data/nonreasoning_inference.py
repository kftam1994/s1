import os
import sys
# Add the parent directory of 'data' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional, Sequence
from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
import time
from dotenv import load_dotenv
from data.utils.io_utils import question_hash, jdump, jload
import random
import requests
import aiohttp
import asyncio
import json
import re
import glob
# response = requests.get(
#   url="https://openrouter.ai/api/v1/auth/key",
#   headers={
#     "Authorization": f'Bearer {os.getenv("OPENROUTER_API_KEY")}'
#   }
# )
# print(json.dumps(response.json(), indent=2))

#@dataclass
#class DataModuleConfigs:
#    model_name: str = field(default="qwen/qwen-2.5-72b-instruct:free", metadata={'help': 'OpenRouter model name'})

system_prompt = '''Give your answer at the end and in a list even though there may be only one answer.
Choice answer is only the capital letter e.g. [A]. If there are multiple correct answers, give the answer in a list e.g. [A, C].
Numeric answer is the number e.g. [692].
Don't need to conduct rounding.
If you cannot determine the answer, please give an empty answer which means Answer: [].
Respond in this format: {Explanation: <Explanation>,Answer: <Answer>}'''

async def _nebius_forward(
    prompts: Sequence[str],
    model_name: str,
    output_dir: str,
    max_retries: int = 10,
    temperature: float = 0.05,
) -> Optional[Sequence[str]]:
    """Forward pass using Nebius API with partial results saving"""
    
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    
    # Load existing partial results if any
    partial_results_file = os.path.join(output_dir, "partial_results.json")
    if os.path.exists(partial_results_file):
        with open(partial_results_file, 'r', encoding='utf-8') as f:
            completed_hashes = json.load(f)
    else:
        completed_hashes = {}
    
    results = []
    async with aiohttp.ClientSession() as session:
        for prompt in tqdm(prompts, desc="Processing prompts"):
            prompt_hash = question_hash(prompt['content'])
            
            # Skip if already processed
            if prompt_hash in completed_hashes:
                results.append(completed_hashes[prompt_hash])
                continue
                
            for attempt in range(max_retries):
                try:
                    completion = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", 
                            "content": system_prompt},
                            prompt
                        ],
                        temperature=temperature,
                    )
                    result = completion.choices[0].message.content
                    results.append(result)
                    
                    # Save the result
                    completed_hashes[prompt_hash] = result
                    with open(partial_results_file, 'w', encoding='utf-8') as f:
                        json.dump(completed_hashes, f, ensure_ascii=False, indent=2)
                    break
                    
                except Exception as e:
                    print(f"\nError on attempt {attempt + 1}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts")
                        results.append("")
                    #time.sleep(2)
    
    return results

api_tokens = [
'cpk_22177f3c0ca54c0a8ad61fe8d8e8fcd9.96594afbb577536a88c4321d5cbcdc73.V8XhB8RNFyrCoh5rJX94kdOHwyzXguEh',
'cpk_af627df8294b4119b1295d7f19bd9ff2.9bf1e94884a9597ab3a76c5acc7b65a4.89fu14iKrR937y1CPHAXBthw8Xipkx8j',
'cpk_f55f8587ed6a4acabf037d214a5c4599.940d302484e35e9a82708a12227a4629.unK6RZONKozl1e6JSOuy2fSXKYd9n5R0',
'cpk_ec44ffedc51a4225825949630037a2d0.ea2953c351f95abab30ce831c0c24fbb.tXkebpiDUjHy0YF8cSw32L5kK6WkkXsT',
'cpk_05d78115ab0d46778d0b5b156d66cbe2.e4f8b1068fcc5603b853e9dfc8757027.6sjnFbfGs2axpDJlQ0vFFDh5GiLpbOoz',
'cpk_294db1b48b4f4d76a16ea9b9bfb73cd4.a3cc6f511c145165992c20d6a96e8c5d.Oe6TYsVhsDdsA0Q4xgT4gIkepEPKKvG2',
'cpk_367b65875cc24118b375348d8ca4fbe5.35601091942152a38f38aacdc4ad81fb.EblgPaaLTdXCkiJhv7DwTtRzQak8NqaW',
'cpk_f81608b5fe17432cacfe29b46a3bd4e1.9de67feb50805c5e900c9ebec647ceff.SdsbA2HQh0FqrKnYBizKCH3U93EnRD5K',
'cpk_2484ef623ac24b2aa93f49762be5df11.45be0d0d3b4b5954be75e5839e296a21.TK1rbVihhD9sgVQMBTBs0TQ8e1rCgMfJ',
'cpk_29dd6daa7a8d477191b8a9cceb964a12.28ab0411d0d25e21bc1537f610d1e7a8.DbgiCcKzcEZEUXNx0BqMPbeSpIvkZ5wl',
]

async def _chute_forward(prompts: Sequence[str],
                         model_name: str,
                         output_dir: str,
                         max_retries: int = 10,
                         temperature: float = 0.05,):
    api_token = os.getenv("CHUTES_API_TOKEN")


	
    # Load existing results if any
    partial_results_file = os.path.join(output_dir, "partial_results.json")
    if os.path.exists(partial_results_file):
        with open(partial_results_file, 'r', encoding='utf-8') as f:
            completed_hashes = json.load(f)
    else:
        completed_hashes = {}

    results = []
    async with aiohttp.ClientSession() as session:
        for prompt in tqdm(prompts, desc="Processing prompts"):
            prompt_hash = question_hash(prompt['content'])
            
            # Skip if already processed
            if prompt_hash in completed_hashes:
                results.append(completed_hashes[prompt_hash])
                continue

            for attempt in range(max_retries):
                api_token = random.choice(api_tokens)
                print(f"Using API token: {api_token}")
                headers = {
                    "Authorization": "Bearer " + api_token,
                    "Content-Type": "application/json"
                }
                try:
                    all_chunks = []
                    body = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", 
                             "content": system_prompt}
                        ] + [prompt],
                        "stream": True,
                        "temperature": temperature
                    }
                    async with session.post(
                        "https://llm.chutes.ai/v1/chat/completions", 
                        headers=headers,
                        json=body
                    ) as response:
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data.strip())
                                    if chunk and chunk['choices']:
                                        all_chunks.append(chunk['choices'][0]['delta']['content'])
                                except Exception as e:
                                    print(f"Error parsing chunk: {e}")
                    
                    result = ''.join(all_chunks)
                    print(result)
                    results.append(result)
                    completed_hashes[prompt_hash] = result
                    with open(partial_results_file, 'w', encoding='utf-8') as f:
                        json.dump(completed_hashes, f, ensure_ascii=False, indent=2)
                    break
                except Exception as e:
                    print(f"\nError on attempt {attempt + 1}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts")
                        raise e
                    await asyncio.sleep(2)
    
        return results


def verify_answer(attempt: str, solution: str, question_type: str) -> bool:
    """
    Verify if the model's answer matches the solution based on question type.
    
    """
    try:
        # Extract answer from attempt (format: "Explanation: ..., Answer: [...]")
        answer_match = re.search(r'Answer:\s*\[(.*?)\]', attempt)
        if not answer_match:
            return False
        
        # Convert model_answer to string if it's a list
        model_answer = answer_match.group(1)
        if isinstance(model_answer, list):
            model_answer = ','.join(map(str, model_answer))
        model_answer = str(model_answer).strip()
        
        # Convert solution to string if it's a list
        if isinstance(solution, list):
            solution = ','.join(map(str, solution))
        solution = str(solution).strip()
        
        if question_type == 'Choice':
            # Convert string answers to sets for comparison
            # Handle multiple choice answers (e.g., "A, C" or "[A,C]")
            model_answers = {a.strip() for a in model_answer.replace('[', '').replace(']', '').split(',')}
            correct_answers = {a.strip() for a in solution.replace('[', '').replace(']', '').split(',')}
            return model_answers == correct_answers
            
        elif question_type == 'Numeric':
            # Convert to float for numeric comparison
            try:
                model_num = float(model_answer)
                solution_num = float(solution)
                # Allow for small floating-point differences
                return abs(model_num - solution_num) < 1e-6
            except ValueError:
                return False
                
        else:
            raise ValueError(f"Unsupported question type: {question_type}")
            
    except Exception as e:
        print(f"Error verifying answer: {e}")
        print(f"Attempt: {attempt}")
        print(f"Solution: {solution}")
        print(f"Question type: {question_type}")
        raise e
        return False

def get_pretty_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_").replace(":", "_")

def difficulty_classification(forward_func, model_name: str, sample_size: int = None, upload: bool = False):
    pretty_name = get_pretty_name(model_name)
    output_dir = f"results/difficulty_classification/{pretty_name}"
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("akftam/financial-qa-s1decontaminate-v1.0")['train']
    # Shuffle and sample the dataset
    if sample_size and sample_size < len(dataset):
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(sample_size)) 
    question_hash_map = {question_hash(row['question']): row for row in dataset}
    questions = dataset['question']
    
    # Process all questions
    prompts = [{"role": "user", "content": q} for q in questions]
    results = asyncio.run(forward_func(prompts, model_name, output_dir))
    
    # Create result dictionary
    result_dict = {}
    for question, result in zip(questions, results):
        result_dict[question_hash(question)] = result
    
    os.makedirs(os.path.join(output_dir, "grading_input"), exist_ok=True)
    
    result = []
    for qhash, attempt in tqdm(result_dict.items(), desc="Creating output json"):
        if qhash in question_hash_map:
            row = question_hash_map[qhash]
            example = dict(
                question=row['question'],
                solution=row['answer'],
                attempt=attempt,
                question_type=row['question_type'],
                is_correct=verify_answer(attempt, row['answer'], row['question_type'])
            )
            jdump(example, f"{output_dir}/grading_input/{qhash}.json")
            result.append(example)
    
    jdump(result, f"{output_dir}/inference_output.json")
    if upload:
        new_dataset_list = []
        for example in dataset:
            new_example = dict(example)
            qhash = question_hash(example['question'])
            if qhash in result_dict:
                new_example[pretty_name] = result_dict[qhash]
            else:
                raise ValueError(f"Hash {qhash} not found in results")
            new_dataset_list.append(new_example)
        new_dataset = Dataset.from_list(new_dataset_list)
        new_dataset.push_to_hub(repo_id=f"akftam/financial-qa-s1decontaminate-v1.0_{pretty_name}_inference")

def analyze_and_filter_results(models: list[str]) -> None:
    """
    Analyze grading results and create filtered dataset of questions answered correctly by any model.
    
    """
    # Initialize statistics dictionary
    stats = {
        'Choice': {'correct': 0, 'incorrect': 0},
        'Numeric': {'correct': 0, 'incorrect': 0}
    }
    
    # Track questions answered correctly by any model
    correctly_answered = set()
    question_results = {}  # Store all results for each question
    
    # Process each model's results
    for model in models:
        pretty_name = get_pretty_name(model)
        results_dir = f"results/difficulty_classification/{pretty_name}/grading_input"
        
        print(f"\nAnalyzing results for {model}...")
        model_stats = {
            'Choice': {'correct': 0, 'incorrect': 0},
            'Numeric': {'correct': 0, 'incorrect': 0}
        }
        
        # Read all json files in the grading_input directory
        for json_file in glob.glob(os.path.join(results_dir,"*.json")):
            result = jload(json_file)
            
            # Update statistics
            if result['is_correct']:
                model_stats[result['question_type']]['correct'] += 1
                correctly_answered.add(question_hash(result['question']))
            else:
                model_stats[result['question_type']]['incorrect'] += 1
                
            # Store result for this question
            qhash = question_hash(result['question'])
            if qhash not in question_results:
                question_results[qhash] = result
        
        # Print model statistics
        print(f"\nResults for {model}:")
        for qtype in ['Choice', 'Numeric']:
            total = model_stats[qtype]['correct'] + model_stats[qtype]['incorrect']
            if total > 0:
                accuracy = (model_stats[qtype]['correct'] / total) * 100
                print(f"{qtype} questions: {model_stats[qtype]['correct']}/{total} "
                      f"({accuracy:.2f}% accuracy)")
    
    # Create filtered dataset
    print(f"Correctly answered questions by any model: {len(correctly_answered)}")
    dataset = load_dataset("akftam/financial-qa-s1decontaminate-v1.0")['train']
    filtered_examples = []
    # Only keep those examples that were not answered correctly by any model
    for example in dataset:
        if question_hash(example['question']) not in correctly_answered:
            filtered_examples.append(example)
    

    filtered_dataset = Dataset.from_list(filtered_examples)
    print(f"\nOriginal dataset size: {len(dataset)}")
    print(f"Output filtered dataset size: {len(filtered_examples)}")
    print(f"Questions skipped/filtered out: {len(dataset) - len(filtered_examples)}")
    assert len(dataset) - len(filtered_examples)==len(correctly_answered)
    filtered_dataset.push_to_hub(repo_id="akftam/financial-qa-s1decontaminate-filtered-v1.0")
    

if __name__ == "__main__":
    #parser = HfArgumentParser(DataModuleConfigs)
    #args = parser.parse_args_into_dataclasses()[0]

    # Models available for free in Chutes ai
    chute_models = [
        #"unsloth/gemma-3-12b-it",
        "chutesai/Mistral-Small-3.1-24B-Instruct-2503",
        # Skip not working, strange response: "Qwen/Qwen2.5-VL-32B-Instruct",
    ]
    # for model in chute_models:
    #     difficulty_classification(_chute_forward, model, upload=True)#, sample_size=10)
    # # # Models for Nebius
    nebius_models = ["Qwen/Qwen2.5-32B-Instruct"]
    
    # for model in nebius_models:
    #     difficulty_classification(_nebius_forward, model, upload=True)#, sample_size=10)

    analyze_and_filter_results(chute_models+nebius_models)