import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import transformers
from baukit import TraceDict
from datasets import load_dataset
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

N_LAYERS_MISTRAL = 32
N_LAYER_LLAMA = 32
N_LAYER_QWEN3 = 36  # Qwen3-8B has 36 layers

LAYERS_TO_TRACE_MISTRAL = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYERS_MISTRAL)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYERS_MISTRAL)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYERS_MISTRAL)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYERS_MISTRAL)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYERS_MISTRAL)],
}

LAYERS_TO_TRACE_LLAMA = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYER_LLAMA)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_LLAMA)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_LLAMA)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_LLAMA)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_LLAMA)],
}

LAYERS_TO_TRACE_QWEN3 = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYER_QWEN3)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_QWEN3)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_QWEN3)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_QWEN3)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_QWEN3)],
}

LAYERS_TO_TRACE = {
    'mistralai/Mistral-7B-Instruct-v0.2': LAYERS_TO_TRACE_MISTRAL,
    'mistralai/Mistral-7B-v0.3': LAYERS_TO_TRACE_MISTRAL,
    'meta-llama/Meta-Llama-3-8B-Instruct': LAYERS_TO_TRACE_LLAMA,
    'meta-llama/Meta-Llama-3-8B': LAYERS_TO_TRACE_LLAMA,
    '/mnt/pcllzy/llama3-instruction-8b': LAYERS_TO_TRACE_LLAMA,
    '/home/models/llama3-8b-instruct': LAYERS_TO_TRACE_LLAMA,
    '/var/luzhenyan/Meta-Llama-3-8B-Instruct': LAYERS_TO_TRACE_LLAMA,
    '/var/wangyicheng/models/Qwen3-8B': LAYERS_TO_TRACE_QWEN3,
}

N_LAYERS = {
    'mistralai/Mistral-7B-Instruct-v0.2': N_LAYERS_MISTRAL,
    'mistralai/Mistral-7B-v0.3': N_LAYERS_MISTRAL,
    'meta-llama/Meta-Llama-3-8B-Instruct': N_LAYER_LLAMA,
    'meta-llama/Meta-Llama-3-8B': N_LAYER_LLAMA,
    '/mnt/pcllzy/llama3-instruction-8b': N_LAYER_LLAMA,
    '/home/models/llama3-8b-instruct': N_LAYER_LLAMA,
    '/var/luzhenyan/Meta-Llama-3-8B-Instruct': N_LAYER_LLAMA,
    '/var/wangyicheng/models/Qwen3-8B': N_LAYER_QWEN3,
}

HIDDEN_SIZE = {
    'tiiuae/falcon-40b-instruct': 8192,
    'mistralai/Mistral-7B-Instruct-v0.2': 4096,
    'mistralai/Mistral-7B-v0.3': 4096,
    'meta-llama/Meta-Llama-3-8B-Instruct': 8192,
    'meta-llama/Meta-Llama-3-8B': 8192,
    '/mnt/pcllzy/llama3-instruction-8b': 8192,
    '/home/models/llama3-8b-instruct': 8192,
    '/var/luzhenyan/Meta-Llama-3-8B-Instruct': 8192,
    '/var/wangyicheng/models/Qwen3-8B': 4096,
    'google/gemma-7b': 3072,
    'google/gemma-7b-it': 3072,
}

LIST_OF_DATASETS = ['triviaqa',
                    'imdb',
                    'winobias',
                    'hotpotqa',
                    'hotpotqa_with_context',
                    'math',
                    'movies',
                    'mnli',
                    'natural_questions_with_context',
                    'winogrande']

LIST_OF_TEST_DATASETS = [f"{x}_test" for x in LIST_OF_DATASETS]

LIST_OF_MODELS = ['mistralai/Mistral-7B-Instruct-v0.2',
                  'mistralai/Mistral-7B-v0.3',
                  'meta-llama/Meta-Llama-3-8B',
                  'meta-llama/Meta-Llama-3-8B-Instruct',
                  '/mnt/pcllzy/llama3-instruction-8b',
                  '/home/models/llama3-8b-instruct',
                  '/var/luzhenyan/Meta-Llama-3-8B-Instruct',
                  '/var/wangyicheng/models/Qwen3-8B',
                  ]

MODEL_FRIENDLY_NAMES = {
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral-7b-instruct',
    'mistralai/Mistral-7B-v0.3': 'mistral-7b',
    'meta-llama/Meta-Llama-3-8B': 'llama-3-8b',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3-8b-instruct',
    '/mnt/pcllzy/llama3-instruction-8b': 'llama-3-8b-instruct-local',
    '/home/models/llama3-8b-instruct': 'llama3-8b-instruct',
    '/var/luzhenyan/Meta-Llama-3-8B-Instruct': 'llama3-8b-instruct',
    '/var/wangyicheng/models/Qwen3-8B': 'qwen3-8b',
}

LIST_OF_PROBING_LOCATIONS = ['mlp', 'mlp_last_layer_only', 'mlp_last_layer_only_input', 'attention_output']


def encode(prompt, tokenizer, model_name):
    messages = [
        {"role": "user", "content": prompt}
    ]
    model_input = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
    return model_input


def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower() or 'qwen' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input


def find_prompt_end_in_full_ids(
    full_ids_1d: torch.Tensor,
    prompt_ids_1d: torch.Tensor,
    *,
    allow_bos_mismatch: bool = True,
) -> int:
    """
    在 `full_ids_1d`（通常是 Prompt+Answer 的全量序列）中，严格通过 ID 匹配找到 prompt 的结束位置（q_len）。

    设计目标：
    - 彻底规避”保存的 full_ids 包含 chat template，但重新 tokenize(question) 的模板略有差异”造成的错位
    - 对 BOS 差异做容错：full 有 BOS/ prompt 没 BOS，或相反
    - 对 chat template 差异做容错：prompt 有 chat template 但 full 没有，或相反

    返回：
      - q_len（int）：prompt 在 full_ids 中的结束 index（Python slicing 语义，full_ids[:q_len] 即 prompt 部分）
    “””
    full = full_ids_1d.detach().cpu()
    prompt = prompt_ids_1d.detach().cpu()

    if full.ndim != 1:
        full = full.view(-1)
    if prompt.ndim != 1:
        prompt = prompt.view(-1)

    # Chat template special tokens (Qwen3 style)
    # <|im_start|> = 151644, <|im_end|> = 151645
    CHAT_TEMPLATE_TOKENS = {151644, 151645, 151643}  # im_start, im_end, im_sep

    def strip_chat_template(ids):
        “””去掉 chat template tokens 和相邻的换行符”””
        if ids.numel() == 0:
            return ids
        start = 0
        while start < ids.numel():
            tok = ids[start].item()
            if tok in CHAT_TEMPLATE_TOKENS:
                start += 1
            elif tok == 198 and start > 0:  # \n after special token
                start += 1
            elif tok in {872, 78191}:  # 'user', 'assistant' role tokens
                start += 1
            else:
                break
        end = ids.numel()
        changed = True
        while changed and end > start:
            changed = False
            tok = ids[end - 1].item()
            if tok in CHAT_TEMPLATE_TOKENS:
                end -= 1
                changed = True
            elif tok == 198:  # \n
                end -= 1
                changed = True
        return ids[start:end]

    candidates = [prompt]
    if allow_bos_mismatch and prompt.numel() >= 2:
        candidates.append(prompt[1:])

    # 添加去掉 chat template 后的候选（Qwen3 兼容）
    prompt_stripped = strip_chat_template(prompt)
    if prompt_stripped.numel() > 0 and prompt_stripped.numel() != prompt.numel():
        candidates.append(prompt_stripped)
        if allow_bos_mismatch and prompt_stripped.numel() >= 2:
            candidates.append(prompt_stripped[1:])

    for cand in candidates:
        if cand.numel() == 0:
            continue
        clen = int(cand.numel())
        flen = int(full.numel())
        if clen > flen:
            continue
        for i in range(flen - clen + 1):
            if torch.equal(full[i : i + clen], cand):
                return int(i + clen)

    # Fallback：保持与旧逻辑相近（至少不越界）
    return int(prompt.numel())


def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None

    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output

def get_indices_of_exact_answer(tokenizer, input_output_ids, exact_answer, model_name, prompt=None, output_ids=None):

    if output_ids is not None:
        lower = input_output_ids.shape[0] - output_ids.shape[0]
    elif prompt is not None:
        prompt_len = tokenize(prompt, tokenizer, model_name).shape[1]
        lower = prompt_len
    else:
        lower = 1

    full_question_answer = tokenizer.decode(input_output_ids[lower:])
    exact_answer_index = full_question_answer.lower().find(exact_answer.lower().strip())

    if exact_answer_index == -1:
        # 找不到匹配时返回空列表，由上游决定跳过该样本
        return []
    true_exact_answer = full_question_answer[exact_answer_index:exact_answer_index + len(exact_answer)]
    assert true_exact_answer in full_question_answer

    higher = len(input_output_ids) - 1

    while true_exact_answer in tokenizer.decode(input_output_ids[lower:higher + 1]):
        higher -= 1
    higher += 1
    while true_exact_answer in tokenizer.decode(input_output_ids[lower:higher + 1]):
        lower += 1
    lower -= 1

    return list(range(lower, higher + 1))

def exact_answer_is_valid(exact_answer_valid, exact_answer):
    return (exact_answer_valid == 1) and (exact_answer != 'NO ANSWER') and (type(exact_answer) == str) and (
                len(exact_answer) > 0)

# A reusable dictionary in case we want to extract the exact answer from the same answer several times during a run
# for efficiency
exact_tokens_dict = {}
def get_token_index(token, tokenizer, question, model_name, full_answer_tokenized=None, exact_answer=None,
                    exact_answer_valid=None, use_dict=True):

    if (type(token) == str) and ('exact' in token):
        if exact_answer_is_valid(exact_answer_valid, exact_answer):
            if (not use_dict) or (question not in exact_tokens_dict):
                t = get_indices_of_exact_answer(tokenizer, full_answer_tokenized, exact_answer, model_name, prompt=question)
                exact_tokens_dict[question] = t
            else:
                t = exact_tokens_dict[question]

            # 找不到匹配则回退到问题末 token
            if t is None or len(t) == 0:
                return get_token_index('last_q_token', tokenizer, question, model_name, exact_answer, exact_answer_valid)

            if token == 'exact_answer_last_token':
                t = min(len(full_answer_tokenized) - 1, t[-1])
            elif token == 'exact_answer_first_token':
                t = t[0]
            elif token == 'exact_answer_before_first_token':
                t = t[0] - 1
            elif token == 'exact_answer_after_last_token':
                t = min(len(full_answer_tokenized) - 1, t[-1] + 1)
        else:
            t = get_token_index('last_q_token', tokenizer, question, model_name, exact_answer, exact_answer_valid) # default case. In the paper we're not supposed to get here.
    else:
        q_length = len(tokenize(question, tokenizer, model_name)[0])
        if token == 'last_q_token':
            t = q_length - 1
        elif token == 'first_answer_token':
            t = q_length
        elif token == 'second_answer_token':
            t = q_length + 1
        elif token == 'full_answer_last_token':
            # 回答段最后一个 token（整体输出末尾）
            t = len(full_answer_tokenized) - 1
        else:
            try:
                token = int(token)
            except ValueError:
                pass
            t = token
    return t


def get_embeddings_in_token(token, layer, extracted_embeddings, tokenizer, prompts, model_name,
                            full_answers_tokenized=None, exact_answers=None, valid_exact_answers=None,
                            use_dict=True):
    X = []
    for idx in range(len(prompts)):

        if (full_answers_tokenized is not None) and (exact_answers is not None) and (valid_exact_answers is not None):
            t = get_token_index(token, tokenizer, prompts[idx], model_name, full_answers_tokenized[idx],
                                exact_answers[idx], valid_exact_answers[idx], use_dict=use_dict)
        else:
            t = get_token_index(token, tokenizer, prompts[idx], model_name, use_dict=use_dict)

        if layer == 'all':
            X.append(extracted_embeddings[idx][:, t].float().numpy())
        else:
            X.append(extracted_embeddings[idx][layer][t].float().numpy())
    return X


def extract_internal_reps_single_sample(model, model_input, probe_at, model_name):

    model_input = model_input.to(model.device)
    layers_to_trace = get_probing_layer_names(probe_at, model_name)

    with torch.no_grad():
        with TraceDict(model, layers_to_trace, retain_input=True, clone=True) as ret:
            output = model(model_input.unsqueeze(dim=0), output_hidden_states=True)

    if 'attention' in probe_at:
        output_per_layer = get_attention_output(model, ret, layers_to_trace, probe_at)
    elif 'mlp' in probe_at:
        output_per_layer = get_mlp_output(ret, layers_to_trace, probe_at)
    else:
        raise TypeError("Probe type not supported")

    return output_per_layer


def get_mlp_output(ret, layers_to_trace, probe_at):
    mlp_output_per_layer = []
    mlp_input_per_layer = []
    for k in layers_to_trace:
        mlp_output_per_token = ret[k].output.squeeze().cpu()
        mlp_output_per_layer.append(mlp_output_per_token)
        mlp_input_per_token = ret[k].input.squeeze().cpu()
        mlp_input_per_layer.append(mlp_input_per_token)

    if 'input' in probe_at:
        return mlp_input_per_layer
    else:
        return mlp_output_per_layer


def get_attention_output(model, ret, layers_to_trace, probe_at):
    attention_output_per_layer = []
    for k in layers_to_trace:
        heads_per_token = ret[k].output.reshape(ret[k].input.shape[0],
                                                ret[k].input.shape[1],
                                                model.model.layers[0].self_attn.num_heads,
                                                model.model.layers[0].self_attn.head_dim).transpose(1, 2)
        attention_output = ret[k].output.squeeze().cpu()
        attention_output_per_layer.append(attention_output)


    return attention_output_per_layer


def extract_internal_reps_specific_layer_and_token(model, tokenizer, prompts, input_output_ids_lst,
                                                   probe_at, model_name, layer, token, exact_answers,
                                                   exact_answers_valid, use_dict_for_tokens=False):
    all_reps = []
    length = len(input_output_ids_lst)
    print(
        f"Extracting internal reps from layer {layer} and token {token} from {length} textual inputs...")

    for idx, (input_output_ids, prompt, exact_answer, exact_answer_valid) in tqdm(enumerate(zip(input_output_ids_lst, prompts, exact_answers, exact_answers_valid))):

        output = extract_internal_reps_single_sample(model, input_output_ids, probe_at, model_name)
        t = get_token_index(token, tokenizer, prompt, model_name, input_output_ids,
                            exact_answer, exact_answer_valid, use_dict=use_dict_for_tokens)
        rep = output[layer][t].float().numpy()
        all_reps.append(rep)

    return all_reps


def extract_internal_reps_all_layers_and_tokens(model, input_output_ids_lst, probe_at, model_name):
    all_outputs_per_layer = []

    length = len(input_output_ids_lst)
    print(f"Extracting internal reps from {length} textual inputs...")

    for input_output_ids in tqdm(input_output_ids_lst):
        output = extract_internal_reps_single_sample(model, input_output_ids, probe_at, model_name)

        all_outputs_per_layer.append(output)

    return all_outputs_per_layer


def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Started loading model")
    force_auto = os.environ.get("LLMSKNOW_FORCE_DEVICE_MAP_AUTO", "").strip() in {"1", "true", "True"}

    # Prefer GPU-only loading when a single CUDA device is visible (unless force_auto is set).
    # This avoids accidental CPU offload under `device_map='auto'`, which can silently degrade performance
    # and break scripts that assume full-GPU execution.
    if (not force_auto) and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=os.environ.get("LLMSKNOW_DEVICE_MAP", "auto"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # Warn if any weights are offloaded to CPU (common cause of unexpected slowdowns).
        if hasattr(model, "hf_device_map") and isinstance(getattr(model, "hf_device_map"), dict):
            if "cpu" in set(model.hf_device_map.values()):
                print("⚠️ Warning: model.hf_device_map contains 'cpu' (CPU offload enabled).")
    return model, tokenizer


def compute_metrics_probing(clf, X_valid, y_valid, pos_label=0, predicted_probas=None):
    if predicted_probas is None:
        baseline_acc = max(y_valid.mean(), (1-y_valid).mean())
        pred = clf.predict(X_valid)
        acc = (pred == y_valid).mean()
        acc_diff_from_baseline = acc - baseline_acc
        precision = precision_score(y_valid, pred, pos_label=pos_label)
        recall = recall_score(y_valid, pred, pos_label=pos_label)
        f1 = f1_score(y_valid, pred, pos_label=pos_label)
        predicted_probas = clf.predict_proba(X_valid)
        predicted_probas = predicted_probas[:, pos_label]
    else:
        baseline_acc = None
        acc = None
        acc_diff_from_baseline = None
        precision = None
        recall = None
        f1 = None

    fpr_for_auc, tpr_for_auc, thresholds = metrics.roc_curve(y_valid, predicted_probas, pos_label=pos_label)
    auc = metrics.auc(fpr_for_auc, tpr_for_auc)

    return {"acc_diff_from_baseline": acc_diff_from_baseline, "f1": f1, "precision": precision, "recall": recall,
            "auc": auc, "baseline_acc": baseline_acc, "acc": acc}


def probe_specific_layer_token(extracted_embeddings_train, extracted_embeddings_valid, layer, token, questions_train,
                               questions_valid, full_answer_tokenized_train, full_answer_tokenized_valid,
                               exact_answer_train, exact_answer_valid, validity_exact_answer_train,
                               validity_exact_answer_valid,
                               tokenizer, y_train, y_valid, seed, model_name,
                               use_dict_for_tokens=True):

    X_train = get_embeddings_in_token(token, layer, extracted_embeddings_train, tokenizer,
                                      questions_train, model_name, full_answer_tokenized_train, exact_answer_train,
                                      validity_exact_answer_train, use_dict=use_dict_for_tokens)
    X_valid = get_embeddings_in_token(token, layer, extracted_embeddings_valid, tokenizer,
                                      questions_valid, model_name, full_answer_tokenized_valid, exact_answer_valid,
                                      validity_exact_answer_valid,
                                      use_dict=use_dict_for_tokens)

    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    return compute_metrics_probing(clf, X_valid, y_valid, pos_label=0)


def compile_probing_indices(data, n_samples, seed, n_validation_samples=0):
    n_samples = eval(n_samples)
    indices = np.arange(len(data))

    if n_validation_samples > 0:
        n_validation_samples = min(n_validation_samples, round(0.2 * (len(indices))))
        indices, validation_data_indices = train_test_split(indices, test_size=n_validation_samples, random_state=seed)

    if n_samples != 'all' and type(n_samples) == int:
        np.random.shuffle(indices)
        indices = indices[:n_samples]  # should be consistent across runs same seed

    if n_validation_samples > 0:
        training_data_indices = indices
    else:
        training_data_indices, validation_data_indices = train_test_split(indices, test_size=0.2, random_state=seed)

    if 'exact_answer' in data:
        training_data_indices = training_data_indices[(data.iloc[training_data_indices]['valid_exact_answer'] == 1) & (data.iloc[training_data_indices]['exact_answer'] != 'NO ANSWER') & (data.iloc[training_data_indices]['exact_answer'].map(lambda x : type(x)) == str)]
        validation_data_indices = validation_data_indices[(data.iloc[validation_data_indices]['valid_exact_answer'] == 1) & (data.iloc[validation_data_indices]['exact_answer'] != 'NO ANSWER') & (data.iloc[validation_data_indices]['exact_answer'].map(lambda x : type(x)) == str)]

    return training_data_indices, validation_data_indices


def get_probing_layer_names(probe_at, model_name):
    if probe_at in ['mlp_last_layer_only', 'mlp_last_layer_only_input']:
        probe_at = 'mlp'
    # 支持动态模型路径：如果 model_name 不在预定义字典中，根据模型名推断
    if model_name in LAYERS_TO_TRACE:
        layers_to_trace = LAYERS_TO_TRACE[model_name][probe_at]
    elif 'qwen' in model_name.lower():
        layers_to_trace = LAYERS_TO_TRACE_QWEN3[probe_at]
    elif 'llama' in model_name.lower():
        layers_to_trace = LAYERS_TO_TRACE_LLAMA[probe_at]
    elif 'mistral' in model_name.lower():
        layers_to_trace = LAYERS_TO_TRACE_MISTRAL[probe_at]
    else:
        layers_to_trace = LAYERS_TO_TRACE_LLAMA[probe_at]
    return layers_to_trace


def prepare_for_probing(data, input_output_ids, training_data_indices, validation_data_indices):

    # small fixture to verify input is not too large which may cause memory overload
    training_data_indices = [i for i in training_data_indices if len(input_output_ids[i]) <= 10000]
    validation_data_indices = [i for i in validation_data_indices if len(input_output_ids[i]) <= 10000]

    data_train = data.iloc[training_data_indices].reset_index()
    data_valid = data.iloc[validation_data_indices].reset_index()


    y_train = data_train['automatic_correctness'].to_numpy()
    y_valid = data_valid['automatic_correctness'].to_numpy()

    input_output_ids_train = [input_output_ids[i] for i in training_data_indices]
    input_output_ids_valid = [input_output_ids[i] for i in validation_data_indices]

    if 'exact_answer' in data:
        exact_answer_train = data_train['exact_answer']
        exact_answer_valid = data_valid['exact_answer']
        validity_of_exact_answer_train = data_train['valid_exact_answer'].astype(int)
        validity_of_exact_answer_valid = data_valid['valid_exact_answer'].astype(int)
    else:
        exact_answer_train = None
        exact_answer_valid = None
        validity_of_exact_answer_train = None
        validity_of_exact_answer_valid = None

    questions_train = data.iloc[training_data_indices].reset_index()['question']
    questions_valid = data.iloc[validation_data_indices].reset_index()['question']

    return data_train, data_valid, input_output_ids_train, input_output_ids_valid, y_train, y_valid,\
            exact_answer_train, exact_answer_valid, validity_of_exact_answer_train, validity_of_exact_answer_valid, \
            questions_train, questions_valid
