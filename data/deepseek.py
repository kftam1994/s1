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

def upload_deepseek():
    jsons = glob(os.path.join("results","deepseek","deepseekall","*.json"))
    all_train = load_dataset("akftam/financial-qa-s1decontaminate-filtered-v1.0")['train']
    all_train_dict= {}
    for example in tqdm(all_train):
        all_train_dict[question_hash(example['question'])] = example
    results = []
    for json_path in tqdm(jsons):
        qdict = jload(json_path)
        qhash = qdict['question_hash']
        if qhash in all_train_dict:
            all_train_example = all_train_dict[qhash]
            all_train_example['thinking_trajectories'] = [qdict['thinking']]
            all_train_example['attempt'] = qdict['response']
            results.append(all_train_example)
    dataset = Dataset.from_list(results)
    dataset.push_to_hub("akftam/financial-qa-filtered-v1.0-deepseek")

if __name__ == "__main__":
    generate_deepseek()
    #upload_deepseek()