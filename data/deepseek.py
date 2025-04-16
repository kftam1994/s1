import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO)
from openai import OpenAI
import os
from datasets import load_dataset, Dataset
from concurrent.futures import ProcessPoolExecutor
import time
from glob import glob
import random
from functools import partial
from tqdm import tqdm
import datetime
from utils.io_utils import question_hash, jdump, jload
#from nonreasoning_inference import verify_answer
import re
import json
from google import genai
from google.genai import types
import time
from collections import deque
from datetime import datetime

# https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/README.md#usage-recommendations
input_prompt_template = \
'''You will be given a problem. Write your response in the same language as the question. Please reason step by step, and put your final answer within \\boxed{{}}.
{instruction}'''

def deepseek_qa(question_prompt: str):
    """
    Query DeepSeek API with retry logic and error handling
    
    Args:
        prompt: Question text to send to the model
    Returns:
        tuple: (thinking trajectory, final answer)
    """
    max_attempts = 10000
    answer = None
    attempts = 0
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    # it can only run between 16:30-00:30 UTC daily off peak hour
    while answer is None and attempts < max_attempts and datetime.datetime.utcnow().hour >= 17:
        try:
            # First get thinking trajectory
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "user", "content": input_prompt_template.format(instruction=question_prompt)}
                ],
                stream=False
            )
            thinking = response.choices[0].message.reasoning_content
            answer = response.choices[0].message.content
            attempts += 1
            
        except Exception as e:
            error_message = str(e)
            if "Insufficient Balance" in error_message:
                logging.error("DeepSeek API credit exhausted. Please recharge your account.")
                raise Exception("Insufficient API credits") from e
            logging.warning(f"Attempt {attempts}/{max_attempts} failed: {error_message}")
            attempts += 1
            time.sleep(60)
    if answer is None or thinking is None:
        raise Exception(f"Failed to get response after {max_attempts} attempts")
            
    return thinking, answer

def process_question(question: str, subdir: str):
    qhash = question_hash(question)
    logging.info(f"Processing question {qhash}")
    thinking, response = deepseek_qa(question)
    result = dict(question_hash=qhash,
                question=question,
                thinking=thinking,
                response=response)
    jdump(result, f"results/deepseek/{subdir}/{qhash}.json")
    logging.info(f"Processed question {qhash}")

def generate_deepseek():
    questions = load_dataset("akftam/financial-qa-s1decontaminate-filtered-v1.0")['train']['question']
    random.shuffle(questions)
    logging.info(f"Processing {len(questions)} total questions")
    subdir = "deepseekall"
    existing_json = glob(os.path.join("results", "deepseek", subdir, "*.json"))
    existing_qhash_list = [os.path.basename(jsonpath).split('.')[0] for jsonpath in existing_json]
    logging.info(f"Found {len(existing_qhash_list)} existing questions")
    # Filter out already processed questions
    remaining_questions = [q for q in questions if question_hash(q) not in existing_qhash_list]
    remaining_count = len(remaining_questions)
    logging.info(f"{remaining_count} questions left after filtering existing question hashes")
    process_map = partial(process_question, subdir=subdir)
    max_workers = min(32, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(process_map, remaining_questions),
            total=len(remaining_questions),
            desc="Processing questions"
        ))
    
    # Verify completion
    final_jsons = glob(os.path.join("results","deepseek",subdir,"*.json"))
    logging.info(f"Completed {len(final_jsons)} out of {len(questions)} questions")
    
    # Report any failures
    completed_hashes = {os.path.basename(j).split('.')[0] for j in final_jsons}
    failed = [q for q in questions if question_hash(q) not in completed_hashes]
    if failed:
        logging.warning(f"Failed to process {len(failed)} questions")
        jdump(failed, f"results/deepseek/{subdir}/failed_questions.json")

# def extract_deepseek_answer(response: str) -> tuple[str, str]:
#     """
#     Extract and standardize answer from DeepSeek response into list format.
#     Returns answers in format: ['123'], ['A'], or ['A','B','C']
    
#     Examples:
#     - "\boxed{ABC}" → ['A','B','C']
#     - "\boxed{42}" → ['42']
#     - "\boxed{-36.5%}" → ['-0.365']
#     - "\boxed{517:476}" → ['1.08613']
#     """
#     # Find all \boxed{} answers
#     boxed_matches = re.findall(r'\\boxed\{(.*?)\}', response)
#     if boxed_matches:
#         uncleaned_answer = boxed_matches[-1]
#         if len(boxed_matches) > 1:
#             logging.info(f"Multiple boxed answers found: {boxed_matches}, using last one: {uncleaned_answer}")
#     else:
#         uncleaned_answer = response
#     # Clean up the answer
#     answer = (
#         uncleaned_answer.strip()
#         .replace('\\', '')
#         .replace('{', '')
#         .replace('}', '')
#         .replace('$', '')
#         .replace('  ', ' ')
#     )
    
#     # Check if this is a multiple choice answer
#     choice_pattern = r'^[A-Z](,\s*[A-Z])*$|^[A-Z]+$'
#     if re.match(choice_pattern, answer.strip()):
#         if ',' in answer:
#             choices = [c.strip() for c in answer.split(',')]
#         else:
#             choices = list(answer)
#         return choices, uncleaned_answer
    
#     # Handle ratio format (e.g., 517:476)
#     ratio_match = re.search(r'(\d+)\s*:\s*(\d+)', answer)
#     if ratio_match:
#         try:
#             numerator = float(ratio_match.group(1))
#             denominator = float(ratio_match.group(2))
#             if denominator != 0:
#                 ratio_value = numerator / denominator
#                 answer = [f"{ratio_value:.5f}"]  # Format to 5 decimal places
#             else:
#                 logging.warning("Found ratio with zero denominator")
#                 answer = [answer.strip()]
#         except ValueError:
#             logging.warning(f"Failed to convert ratio: {answer}")
#             answer = [answer.strip()]
#         return answer, uncleaned_answer
    
#     # Handle percentage values
#     percentage_match = re.search(r'(-?\d+\.?\d*)%', answer)
#     if percentage_match:
#         try:
#             value = float(percentage_match.group(1))
#             answer = [str(value / 100)]
#             return answer, uncleaned_answer
#         except ValueError:
#             logging.warning(f"Failed to convert percentage value: {percentage_match.group(0)}")
#             answer = [answer.strip()]

#     # For regular numeric or text answers
#     answer = [answer.strip()]
#     return answer, uncleaned_answer
#     #return [response.strip()], response.strip()

verification_system_prompt = '''You will be given a question, a question type, an attempt, and a solution. Your task is to determine if the attempt is correct or incorrect.
Firstly extract the final and concluded answer from the attempt. The extracted answer could be inside a \boxed{<answer>}.
Then determine if the attempt is correct or incorrect. If the attempt is correct, return true. If the attempt is incorrect, return false.

Question type is one of the following: [Choice, Numeric].
1. Choice:
- Choice answer can be a single capital letter e.g. A. Or if there are multiple correct answers, the answer will be multiple capital letter e.g. [A, C].
- Extracted choice answer should be in a list of string format e.g. ['A', 'B'] or ['A'].
2. Numeric:
- Numeric answer can be a percentage number e.g. 5%, ratio number e.g. 1:2, decimal number e.g. 0.12345 or a fraction e.g. 1/2. Convert it to float before comparing with the solution.
- Extracted numeric answer should be in a list of string format and the value should be a float number e.g. ['0.5'] or ['0.12345'].

Respond in this format: {"extracted_answer": <extracted_answer>, "is_correct": <is_correct>}. 
'''

def llm_verify_answer(attempt, solution, question, question_type, percentage_error_tolerance, request_times):
    """
    Verify the answer using LLM
    """
    if question_type=='Numeric':
        boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', attempt)
        if boxed_matches:
            all_answer_float = 0
            for uncleaned_answer in boxed_matches:
                answer_float= None
                # Handle LaTeX fraction format (e.g., \dfrac{111}{29})
                frac_match = re.search(r'dfrac\{(\d+)\}\{(\d+)\}', uncleaned_answer)
                if frac_match:
                    numerator = float(frac_match.group(1))
                    denominator = float(frac_match.group(2))
                    answer_float = numerator / denominator
                answer = (
                    uncleaned_answer.strip()
                    .replace('\\', '')
                    .replace('{', '')
                    .replace('}', '')
                    .replace('$', '')
                    .replace('  ', ' ')
                    .replace(',', '')
                )
                # Handle ratio format (e.g., 517:476)
                ratio_match = re.search(r'(-?\d+\.?\d*)\s*:\s*(-?\d+\.?\d*)', answer)
                if ratio_match and answer_float is None:
                    numerator = float(ratio_match.group(1))
                    denominator = float(ratio_match.group(2))
                    if denominator != 0:
                        answer_float = numerator / denominator
                # Handle percentage values
                percentage_match = re.search(r'(-?\d+\.?\d*)%', answer)
                if percentage_match and answer_float is None:
                    value = float(percentage_match.group(1))
                    answer_float = value / 100
                # Regular numeric
                check_numeric = re.search(r'(-?\d+(?:,\d{3})*\.?\d*)', answer)
                if check_numeric and answer_float is None:
                    numeric_part = check_numeric.group(1).replace(',', '')
                    answer_float = float(numeric_part)
                if answer_float is not None:
                    all_answer_float+=answer_float # assume summing up all answers
            if isinstance(all_answer_float, float):
                solution_num = float(solution[0])
                if solution_num == 0:
                    solution_num = 1e-10
                # if answer is {}percentage, {}billion, {}million, change {} without sign
                adj_num = [1, 100, 1e3, 1e6, 1e9]
                is_correct = any([abs((abs(all_answer_float*adj) - abs(solution_num)) / solution_num) < percentage_error_tolerance for adj in adj_num]) or \
                            any([abs((abs(all_answer_float/adj) - abs(solution_num)) / solution_num) < percentage_error_tolerance for adj in adj_num])
                extracted_answer = [str(all_answer_float)]
                return is_correct, extracted_answer
    
    # If we've made 10 requests, check if we need to wait
    if len(request_times) == 10:
        elapsed = (datetime.now() - request_times[0]).total_seconds()
        if elapsed < 60:
            time.sleep(60 - elapsed)  # Wait remaining time until oldest request is 60s old
            request_times.popleft()  # Remove oldest request
    
    # Add current request time
    request_times.append(datetime.now())
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"#"gemini-2.0-pro-exp-02-05"#"gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Question type: {question_type}\nQuestion: {question}\n\nSolution: {solution}\n\nAttempt: {attempt}"),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=verification_system_prompt),
        ],
    )
    response = client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )
    cleaned_text = (response.text
        .replace('```json\n', '')
        .replace('\n```', '')
        .replace('True', 'true')
        .replace('False', 'false')
        .replace("\'", '"')
        .strip())
    response_dict = json.loads(cleaned_text)
    is_correct = response_dict['is_correct']
    extracted_answer = response_dict['extracted_answer']
    # Ensure extracted_answer is list
    if not isinstance(extracted_answer, list):
        extracted_answer = [extracted_answer]
    if len(extracted_answer)==1 and len(solution)==1:
        is_correct = is_correct or extracted_answer[0].lower() == solution[0].lower()
    return is_correct, extracted_answer



def upload_deepseek():
    """Upload processed DeepSeek results with answer verification"""
    jsons = glob(os.path.join("results", "deepseek", "deepseekall", "*.json"))
    checkpoint_path = "results/llm_verify/checkpoint.json"
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = jload(checkpoint_path)
        results = checkpoint['results']
        processed_hashes = set(x['question_hash'] for x in results)
        jsons = [j for j in jsons if os.path.basename(j).split('.')[0] not in processed_hashes]
        logging.info(f"Loaded {len(results)} results from checkpoint")
    else:
        results = []
        processed_hashes = set()

    all_train = load_dataset("akftam/financial-qa-s1decontaminate-filtered-v1.0")['train']
    request_times = deque(maxlen=10)  # Shared queue for rate limiting
    # Create lookup dictionary for training examples
    all_train_dict = {}
    for example in tqdm(all_train, desc="Creating lookup dictionary"):
        all_train_dict[question_hash(example['question'])] = example
    
    # Process each result
    for i, json_path in enumerate(tqdm(jsons, desc="Processing results")):
        qdict = jload(json_path)
        qhash = qdict['question_hash']
        
        if qhash in all_train_dict:
            all_train_example = all_train_dict[qhash].copy()
            
            # Extract answer from response
            response = qdict['response']
            #extracted_answer, uncleaned_extracted_answer = extract_deepseek_answer(response)
            is_correct, extracted_answer= llm_verify_answer(attempt=response,
                              solution=all_train_example['answer'],
                              question=all_train_example['question'],
                              question_type=all_train_example['question_type'],
                              percentage_error_tolerance=5e-2,
                              request_times=request_times)
            # Add thinking trajectories and attempt
            all_train_example['thinking_trajectories'] = qdict['thinking']
            all_train_example['attempt'] = response
            all_train_example['extracted_answer'] = extracted_answer
            
            # # Verify answer with extracted answer
            # is_correct = verify_answer(
            #     attempt=f"Answer: [{','.join(extracted_answer)}]",  # Format for verify_answer
            #     solution=all_train_example['answer'],
            #     question_type=all_train_example['question_type'],
            #     numeric_error_tolerance=5e-2
            # )
            if not is_correct:
                logging.info(f"Incorrect for question {qhash}:")
                logging.info(f"Uncleaned Answer: {response}, Extracted answer: {extracted_answer}, Expected answer: {all_train_example['answer']}")
                #if len(extracted_answer)==1 and len(all_train_example['answer'])==1:
                #    print()
            else:
                logging.info(f"Correct for question {qhash}:")
                logging.info(f"Uncleaned Answer: {response}, Extracted answer: {extracted_answer}, Expected answer: {all_train_example['answer']}")
                #print()
            # Add verification result
            all_train_example['is_correct'] = is_correct
            all_train_example['question_hash'] = qhash
            results.append(all_train_example)

            checkpoint = {
                'results': results,
                'last_processed': json_path
            }
            jdump(checkpoint, checkpoint_path)
    
    # Log statistics
    stats = {'correct': 0, 'incorrect': 0}
    checkpoint = jload(checkpoint_path)
    for k,v in checkpoint.items():
        if k == 'results':
            for result in v:
                is_correct = result['is_correct']
                stats['correct' if is_correct else 'incorrect'] += 1
    total = stats['correct'] + stats['incorrect']
    accuracy = (stats['correct'] / total * 100) if total > 0 else 0
    logging.info(f"Verification complete:")
    logging.info(f"Correct: {stats['correct']}/{total} ({accuracy:.2f}%)")
    logging.info(f"Incorrect: {stats['incorrect']}/{total} ({100-accuracy:.2f}%)")
    
    # Create and upload dataset
    dataset = Dataset.from_list(results)
    dataset.push_to_hub("akftam/financial-qa-filtered-v1.0-deepseek")
    logging.info(f"Dataset uploaded with {len(results)} examples")

if __name__ == "__main__":
    generate_deepseek()
    upload_deepseek()