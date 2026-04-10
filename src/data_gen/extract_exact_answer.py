import argparse
import sys
import json
import re
import os

import numpy as np
import wandb
from sklearn.utils import resample

# 添加正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '..'))

import pandas as pd
import torch
from tqdm import tqdm

from probing_utils import load_model_and_validate_gpu, tokenize, generate, LIST_OF_MODELS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_TEST_DATASETS, LIST_OF_DATASETS
from compute_correctness import compute_correctness_triviaqa, compute_correctness_math, compute_correctness


# System prompt for extraction
EXTRACTION_SYSTEM_PROMPT = """You are an annotation tool.

Goal: From the Model answer text, extract ONE contiguous short answer span (usually an entity / number / date) that represents the model's final answer.

Rules:
- The extracted span must be copied EXACTLY from the Model answer (verbatim). Do not rewrite, paraphrase, or normalize it.
- Do NOT output boilerplate words like "You", "That's", "Q:", "A:", or any special template tokens like "<|...|>".
- If the Model answer contains no concrete answer candidate at all (e.g., only refusal/clarification), output "NO_ANSWER" as key_span_text.
- Even if the Model answer is incorrect, still extract the wrong answer span that the model actually gave (do NOT output "NO_ANSWER" just because it is wrong).

Return ONLY valid JSON with keys:
- key_span_text (string)
- matches_ground_truth ("true" | "false" | "uncertain")
- confidence (number between 0 and 1)
- rationale_short (one short sentence)"""


def tokenize_with_system(prompt, tokenizer, model_name, system_prompt=None):
    """Tokenize with optional system prompt using chat template."""
    if 'instruct' in model_name.lower() or 'qwen' in model_name.lower():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer_kwargs = {'add_generation_prompt': True}
        model_input = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            **tokenizer_kwargs
        ).to('cuda')
    else:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        model_input = tokenizer(full_prompt, return_tensors='pt')
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input


def parse_json_output(raw_output, model_name):
    """Parse JSON from model output, handling various formats."""
    if 'mistral' in model_name.lower():
        raw_output = raw_output.replace("</s>", "")
    elif 'llama' in model_name.lower():
        raw_output = raw_output.replace("<|eot_id|>", "")
    elif 'qwen' in model_name.lower():
        raw_output = raw_output.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        raw_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)

    raw_output = raw_output.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=LIST_OF_DATASETS + LIST_OF_TEST_DATASETS)
    parser.add_argument("--do_resampling", type=int, required=False, default=0, help="If 0, the script will extract exact answers from the model answers. If > 0, the script will extract exact answers from the resampled model answers (looking for a file of do_resampling resamples).")
    parser.add_argument("--get_extraction_stats", action='store_true', default=False, help="Purely for getting statistics. If activated, the file will not be saved.")
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--extraction_model", choices=LIST_OF_MODELS, default='mistralai/Mistral-7B-Instruct-v0.2', help="model used for exact answer extraction")
    parser.add_argument("--model", choices=LIST_OF_MODELS, default='mistralai/Mistral-7B-Instruct-v0.2', help="model which answers are to be extracted")
    # 分片参数，用于多GPU并行
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for multi-GPU parallel processing (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for parallel processing")
    parser.add_argument("--input_file", type=str, default=None, help="Optional: explicit path to input CSV file (overrides default path)")

    args = parser.parse_args()
    wandb.init(
        project="extract_exact_answer",
        config=vars(args)
        )

    return args


def extract_winobias_entity(model_answer):
    """
    专门针对 WinoBias 数据集提取被指代的实体。
    模型回答格式: "The pronoun 'he/she' refers to the [entity]."
    """
    patterns = [
        r"refers to the\s+[\"']?(\w+)[\"']?",
        r"refers to\s+the\s+[\"']?(\w+)[\"']?",
        r"refer to the\s+[\"']?(\w+)[\"']?",
        r"refer to\s+the\s+[\"']?(\w+)[\"']?",
        r"refers to\s+[\"'](\w+)[\"']",
        r"is the\s+[\"']?(\w+)[\"']?",
        r"is\s+the\s+[\"']?(\w+)[\"']?",
        r"the answer is\s+[\"']?(\w+)[\"']?",
        r"the answer is the\s+[\"']?(\w+)[\"']?",
        r"it's the\s+[\"']?(\w+)[\"']?",
        r"must be the\s+[\"']?(\w+)[\"']?",
    ]

    for pattern in patterns:
        match = re.search(pattern, model_answer.lower())
        if match:
            entity = match.group(1)
            if entity not in ['he', 'she', 'his', 'her', 'him', 'they', 'them', 'the', 'a', 'an']:
                original_match = re.search(r'\b' + re.escape(entity) + r'\b', model_answer, re.IGNORECASE)
                if original_match:
                    return original_match.group(0)
                return entity

    return None


def extract_exact_answer(model, tokenizer, correctness, question, model_answer, correct_answer, model_name, dataset=None):

    # 对于 WinoBias 数据集，使用专门的实体提取逻辑
    if dataset and 'winobias' in dataset.lower():
        entity = extract_winobias_entity(str(model_answer))
        if entity:
            if entity.lower() in str(model_answer).lower():
                return entity, 1
        if correctness == 1 and correct_answer:
            if str(correct_answer).lower() in str(model_answer).lower():
                return correct_answer, 1
        return "NO ANSWER", 0

    if correctness == 1:
        found_ans_index = len(model_answer)
        found_ans = ""

        try:
            correct_answer_ = eval(correct_answer)
            if type(correct_answer_) == list:
                correct_answer = correct_answer_
        except:
            correct_answer = correct_answer

        if type(correct_answer) == list:
            for ans in correct_answer:
                ans_index = model_answer.lower().find(ans.lower())
                if ans_index != -1 and ans_index < found_ans_index:
                    found_ans = ans
                    found_ans_index = ans_index
        elif type(correct_answer) in [int, float]:
            found_ans_index = model_answer.lower().find(str(round(correct_answer)))
            found_ans = str(round(correct_answer))
            if found_ans_index == -1:
                found_ans_index = model_answer.lower().find(str(correct_answer))
                found_ans = str(correct_answer)
        else:
            found_ans_index = model_answer.lower().find(correct_answer.lower())
            found_ans = correct_answer

        if found_ans_index == -1:
            print("##")
            print(model_answer)
            print("##")
            print(correct_answer)
            print("ERROR!", question)
        exact_tokens = list(range(found_ans_index, found_ans_index + len(found_ans)))
        exact_answer = "".join([model_answer[i] for i in exact_tokens])
        valid = 1
    else:
        # For incorrect answers, use LLM with structured JSON prompt
        if isinstance(correct_answer, str):
            try:
                correct_answer_display = eval(correct_answer)
                if isinstance(correct_answer_display, list):
                    correct_answer_display = correct_answer_display[0] if correct_answer_display else correct_answer
            except:
                correct_answer_display = correct_answer
        else:
            correct_answer_display = str(correct_answer)

        prompt = f"""Question: {question}
Ground truth: {correct_answer_display}
Model answer: {model_answer}"""

        model_input = tokenize_with_system(prompt, tokenizer, model_name, EXTRACTION_SYSTEM_PROMPT).to(model.device)
        valid = 0
        retries = 0
        exact_answer = "NO ANSWER"

        print("=" * 80)
        print(f"INPUT PROMPT:\n{prompt[:500]}...")
        print("-" * 80)

        while valid == 0 and retries < 5:
            with torch.no_grad():
                model_output = generate(model_input, model, model_name, do_sample=(retries > 0),
                                       output_scores=False, max_new_tokens=800)
                raw_output = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])

            # 分离思考和回答
            think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                answer_part = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
                print(f"THINKING ({len(think_content)} chars): {think_content[:200]}...")
                print(f"ANSWER: {answer_part[:300]}")
            else:
                print(f"RAW OUTPUT: {raw_output[:300]}")

            parsed = parse_json_output(raw_output, model_name)

            if parsed and 'key_span_text' in parsed:
                exact_answer = parsed['key_span_text']
                print(f"PARSED JSON: key_span_text='{exact_answer}', matches={parsed.get('matches_ground_truth')}, conf={parsed.get('confidence')}")

                if exact_answer in ("NO_ANSWER", "NO ANSWER"):
                    exact_answer = "NO ANSWER"
                    valid = 1
                elif type(model_answer) == float:
                    exact_answer = "NO ANSWER"
                    valid = 0
                else:
                    def normalize_quotes(s):
                        return s.replace("'", '"').replace("\u2018", '"').replace("\u2019", '"').replace("\u201c", '"').replace("\u201d", '"')

                    normalized_answer = normalize_quotes(exact_answer.lower())
                    normalized_model = normalize_quotes(model_answer.lower())

                    if normalized_answer in normalized_model:
                        valid = 1
                        print(f"VALIDATION PASSED: '{exact_answer}' found in model_answer")
                    else:
                        print(f"VALIDATION FAILED [Retry {retries}]: '{exact_answer}' not in model_answer")
            else:
                print(f"  [Retry {retries}] Failed to parse JSON")
                cleaned = raw_output
                if 'qwen' in model_name.lower():
                    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
                elif 'llama' in model_name.lower():
                    cleaned = cleaned.replace("<|eot_id|>", "")
                elif 'mistral' in model_name.lower():
                    cleaned = cleaned.replace("</s>", "")

                cleaned = cleaned.strip().split('\n')[0].strip()
                if cleaned and len(cleaned) < 100:
                    exact_answer = cleaned
                    if exact_answer.lower() in model_answer.lower():
                        valid = 1

            retries += 1

    return exact_answer, valid


def main():
    args = parse_args()
    model, tokenizer = load_model_and_validate_gpu(args.extraction_model)

    if args.input_file:
        source_file = args.input_file
    else:
        source_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    resampling_file = f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.do_resampling}_textual_answers.pt"
    if args.do_resampling > 0:
        destination_file = f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.do_resampling}_exact_answers.pt"
    else:
        destination_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"

    model_answers = pd.read_csv(source_file)
    print(f"Total data length: {len(model_answers)}")

    if args.do_resampling > 0:
        all_resample_answers = torch.load(resampling_file)

    # 应用分片逻辑
    if args.num_shards > 1:
        total_len = len(model_answers)
        shard_size = (total_len + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_len)
        model_answers = model_answers.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"Shard {args.shard_id}/{args.num_shards}: processing rows {start_idx}-{end_idx} ({len(model_answers)} samples)")
        destination_file = destination_file.replace('.csv', f'_shard{args.shard_id}.csv')

    exact_answers = []
    valid_lst = []
    ctr = 0
    ctr_no_answer = 0

    if args.n_samples > 0:
        model_answers = resample(model_answers, n_samples=args.n_samples, stratify=model_answers['automatic_correctness'])

    for idx, row in tqdm(model_answers.iterrows(), total=len(model_answers)):
        print(f"###### sample {idx} #######")

        if 'raw_question' in row:
            question_col = 'raw_question'
        else:
            question_col = 'question'

        if args.do_resampling <= 0:
            if ('natural_questions' in source_file) or args.get_extraction_stats:
                automatic_correctness = 0
            else:
                automatic_correctness = row['automatic_correctness']

            if 'instruct' not in args.model.lower() and 'qwen' not in args.model.lower():
                model_answer = row['model_answer'].split("\n")[0]
            else:
                model_answer = row['model_answer']

            exact_answer, valid = extract_exact_answer(model, tokenizer,
                                                       automatic_correctness,
                                                       row[question_col], model_answer,
                                                       row['correct_answer'], args.extraction_model,
                                                       dataset=args.dataset)
            exact_answers.append(exact_answer)
            valid_lst.append(valid)
            if exact_answer == 'NO ANSWER':
                ctr_no_answer += 1
            if valid == 1:
                ctr += 1
        else:
            exact_answers_specific_index = []
            valid_lst_specific_index = []
            for resample_answers in all_resample_answers:
                assert(len(model_answers) == len(resample_answers))
                resample_answer = resample_answers[idx].split("\n")[0]

                automatic_correctness = compute_correctness([row.question], args.dataset, args.model, [row['correct_answer']], model, [resample_answer], tokenizer, None)['correctness'][0]

                exact_answer, valid = extract_exact_answer(model, tokenizer,
                                                           automatic_correctness,
                                                           row[question_col], resample_answer,
                                                           row['correct_answer'], args.model,
                                                           dataset=args.dataset)
                exact_answers_specific_index.append(exact_answer)
                valid_lst_specific_index.append(valid)
                if exact_answer == 'NO ANSWER':
                    ctr_no_answer += 1
                if valid == 1:
                    ctr += 1
            exact_answers.append(exact_answers_specific_index)
            valid_lst.append(valid_lst_specific_index)

    if args.do_resampling > 0:
        total_n_answers = len(model_answers) * len(all_resample_answers)
    else:
        total_n_answers = len(model_answers)

    wandb.summary['successful_extractions'] = ctr / total_n_answers
    wandb.summary['no_answer'] = ctr_no_answer / total_n_answers

    if not args.get_extraction_stats:
        if args.do_resampling <= 0:
            model_answers['exact_answer'] = exact_answers
            model_answers['valid_exact_answer'] = valid_lst
            model_answers.to_csv(destination_file)
        else:
            torch.save({
                "exact_answer": exact_answers,
                "valid_exact_answer": valid_lst
            }, destination_file)
    else:
        model_answers['exact_answer'] = exact_answers
        model_answers['valid_exact_answer'] = valid_lst

if __name__ == "__main__":
    main()
