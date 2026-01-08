import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    get_indices_of_exact_answer,
    load_model_and_validate_gpu,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--layer", type=str, default="last")
    parser.add_argument("--max_samples", type=int, default=200)
    return parser.parse_args()

def build_question_answer(row, model_name):
    q_col = "raw_question" if "raw_question" in row and not pd.isna(row["raw_question"]) else "question"
    question = str(row[q_col])
    if "instruct" in model_name.lower():
        answer = str(row["model_answer"])
    else:
        answer = str(row["model_answer"]).split("\n")[0]
    return question, answer

def forward_on_ids(model, input_ids_1d: torch.Tensor):
    input_ids = input_ids_1d.to(model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, output_attentions=False)
    logits = outputs.logits[0].detach().cpu()
    hidden_states = [h[0].detach().cpu() for h in outputs.hidden_states]
    return input_ids_1d.detach().cpu(), logits, hidden_states

def entropy_and_margin(logits_slice):
    probs = torch.softmax(logits_slice, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
    top2 = torch.topk(logits_slice, k=2, dim=-1).values
    margin = (top2[0] - top2[1]).item() if top2.numel() == 2 else 0.0
    p_top1 = torch.max(probs).item()
    return entropy, margin, p_top1

def main():
    args = parse_args()
    
    print(f"Loading probe from {args.probe_path}")
    probe_data = joblib.load(args.probe_path)
    clf = probe_data["clf"]
    pca = probe_data["pca"]

    friendly = MODEL_FRIENDLY_NAMES[args.model]
    source_file = f"../output/{friendly}-answers-{args.dataset}.csv"
    ids_file = f"../output/{friendly}-input_output_ids-{args.dataset}.pt"

    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    model, tokenizer = load_model_and_validate_gpu(args.model)
    layer_idx = -1 if args.layer == "last" else int(args.layer)

    df = pd.read_csv(source_file)
    df = df[(df["valid_exact_answer"] == 1) & (df["exact_answer"] != "NO ANSWER")]
    if args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)

    input_output_ids_all = torch.load(ids_file, map_location="cpu")

    X = []
    y_true = []
    
    print(f"Extracting features for evaluation on {args.dataset}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question, answer = build_question_answer(row, args.model)
        try:
            input_ids_1d = input_output_ids_all[int(idx)]
            input_ids, logits, hidden_states = forward_on_ids(model, input_ids_1d)
            span = get_indices_of_exact_answer(tokenizer, input_ids, row["exact_answer"], args.model, prompt=question)
        except Exception:
            continue

        if not span: continue

        # Evaluate on all tokens in the sequence (or just answer part)
        # Here we follow the training logic: evaluate on positives and some negatives
        for t_idx in range(len(input_ids)):
            label = 1 if t_idx in span else 0
            
            if t_idx == 0:
                entropy, margin, p_top1 = 0.0, 0.0, 0.0
            else:
                entropy, margin, p_top1 = entropy_and_margin(logits[t_idx - 1])
            
            hs = hidden_states[layer_idx][t_idx].to(torch.float32).numpy()
            pos_feat = np.array([t_idx, t_idx - span[0], span[-1] - t_idx], dtype=np.float32)
            scalar_feat = np.array([entropy, margin, p_top1], dtype=np.float32)
            feat = np.concatenate([hs, pos_feat, scalar_feat])
            
            X.append(feat)
            y_true.append(label)

    X = np.stack(X)
    if pca is not None:
        X = pca.transform(X)
    
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    print("\n=== Token-level Evaluation Results ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_prob):.4f}")

if __name__ == "__main__":
    main()



