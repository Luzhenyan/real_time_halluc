#!/usr/bin/env python3
"""
Visualize pos_probe scores for each sample.
- Plots: x=token step, y=confidence score
- Highlights GT span region
- Saves both plots and raw data (JSON)
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from tqdm import tqdm

sys.path.append("/mnt/pcllzy_2/LLMsKnow/src")
from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    tokenize,
    find_prompt_end_in_full_ids,
    get_indices_of_exact_answer,
    get_probing_layer_names,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="triviaqa_test")
    p.add_argument("--model", type=str, default="/mnt/pcllzy/llama3-instruction-8b")
    p.add_argument("--pos_probe_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=200, help="Number of samples to visualize (0=all)")
    p.add_argument("--balanced_eval", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="output/pos_score_vis")
    p.add_argument("--thr", type=float, default=0.8, help="Threshold line to draw")
    return p.parse_args()


def load_pos_probe(path, num_layers):
    d = joblib.load(path)
    probe_at = str(d.get("probe_at", "resid") or "resid")
    layer_raw = d.get("layer", None)
    layer = None
    if probe_at == "mlp":
        if layer_raw is None or str(layer_raw) == "last":
            layer = num_layers - 1
        else:
            layer = int(layer_raw)
    return {
        "clf": d["clf"],
        "pca": d.get("pca", None),
        "scaler": d.get("scaler", None),
        "probe_at": probe_at,
        "layer": layer,
    }


def forward_step(model, step_input, past, probe, mlp_layer_name):
    with torch.no_grad():
        if probe["probe_at"] == "mlp":
            with TraceDict(model, [mlp_layer_name], retain_input=False) as td:
                out = model(input_ids=step_input, past_key_values=past, use_cache=True, output_hidden_states=False)
            feat = td[mlp_layer_name].output[0, -1, :].cpu().float().numpy().reshape(1, -1)
        else:
            out = model(input_ids=step_input, past_key_values=past, use_cache=True, output_hidden_states=True)
            feat = out.hidden_states[-1][0, -1, :].cpu().float().numpy().reshape(1, -1)
        return out.past_key_values, feat


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model with auto device map for memory efficiency
    print("Loading model with device_map='auto'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    friendly = MODEL_FRIENDLY_NAMES[args.model]
    
    # Load data
    llms_know_root = "/mnt/pcllzy_2/LLMsKnow"
    df = pd.read_csv(f"{llms_know_root}/output/{friendly}-answers-{args.dataset}.csv").copy()
    df["_orig_idx"] = df.index.astype(int)
    
    if args.balanced_eval and args.max_samples > 0:
        df_pos = df[df["automatic_correctness"] == 1]
        df_neg = df[df["automatic_correctness"] == 0]
        half = max(1, args.max_samples // 2)
        n = min(half, len(df_pos), len(df_neg))
        df = pd.concat([df_pos.sample(n=n, random_state=42), df_neg.sample(n=n, random_state=42)]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"📊 Balanced subset: {n} correct + {n} incorrect = {2*n} samples")
    elif args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)
    
    ids_all = torch.load(f"{llms_know_root}/output/{friendly}-input_output_ids-{args.dataset}.pt", map_location="cpu")
    
    # Load probe
    probe = load_pos_probe(args.pos_probe_path, model.config.num_hidden_layers)
    mlp_layer_name = None
    if probe["probe_at"] == "mlp":
        names = get_probing_layer_names("mlp", model.config._name_or_path)
        mlp_layer_name = names[probe["layer"]]
        print(f"Using MLP layer={probe['layer']} ({mlp_layer_name})")
    
    all_samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        orig_idx = int(row["_orig_idx"])
        full_ids = ids_all[orig_idx]
        prompt = str(row["question"])
        exact_answer = str(row.get("exact_answer", ""))
        valid = int(row.get("valid_exact_answer", 0))
        correct = int(row.get("automatic_correctness", 0))
        
        if valid != 1 or not exact_answer or exact_answer == "NO ANSWER":
            continue
        
        prompt_ids = tokenize(prompt, tokenizer, args.model)[0]
        q_len = find_prompt_end_in_full_ids(full_ids, prompt_ids, allow_bos_mismatch=True)
        output_ids = full_ids[q_len:]
        
        span = get_indices_of_exact_answer(tokenizer, full_ids, exact_answer, args.model, output_ids=output_ids)
        if not span:
            continue
        
        # Convert to decode steps (1-indexed)
        span_steps = [t - q_len + 1 for t in span if t >= q_len]
        if not span_steps:
            continue
        gs, ge = min(span_steps), max(span_steps)
        
        # Teacher-forced decode
        current_input = full_ids[:q_len].to(model.device).unsqueeze(0)
        with torch.no_grad():
            pre = model(input_ids=current_input, use_cache=True, output_hidden_states=False)
        past = pre.past_key_values
        
        gold_out = full_ids[q_len:].to(model.device)
        max_steps = min(args.max_new_tokens, gold_out.shape[0])
        
        scores = []
        tokens = []
        for step in range(max_steps):
            step_input = gold_out[step:step+1].unsqueeze(0)
            past, feat = forward_step(model, step_input, past, probe, mlp_layer_name)
            
            if probe["scaler"] is not None:
                feat = probe["scaler"].transform(feat)
            if probe["pca"] is not None:
                feat = probe["pca"].transform(feat)
            
            score = float(probe["clf"].predict_proba(feat)[0, 1])
            scores.append(score)
            tokens.append(tokenizer.decode([gold_out[step].item()]))
        
        if not scores:
            continue
        
        sample_data = {
            "sample_idx": idx,
            "orig_idx": orig_idx,
            "correct": correct,
            "question": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "exact_answer": exact_answer,
            "gt_span": [gs, ge],  # 1-indexed decode steps
            "scores": scores,
            "tokens": tokens,
        }
        all_samples.append(sample_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 4))
        steps = list(range(1, len(scores) + 1))
        
        # Highlight GT span
        ax.axvspan(gs, ge, alpha=0.3, color='green', label=f'GT span [{gs}-{ge}]')
        
        # Plot scores
        ax.plot(steps, scores, 'b.-', linewidth=1, markersize=4, label='Pos score')
        
        # Threshold line
        ax.axhline(y=args.thr, color='r', linestyle='--', alpha=0.7, label=f'Threshold={args.thr}')
        
        ax.set_xlabel('Decode step (token position)')
        ax.set_ylabel('Confidence (P(key token))')
        ax.set_title(f'Sample {idx} | Correct={correct} | Answer: "{exact_answer[:30]}..."')
        ax.set_xlim(0.5, len(scores) + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add token labels on x-axis for first 20 tokens
        if len(tokens) <= 30:
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{s}\n{t[:6]}" for s, t in zip(steps, tokens)], fontsize=6, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/sample_{idx:03d}.png", dpi=150)
        plt.close()
    
    # Save all data to JSON
    with open(f"{args.output_dir}/all_samples.json", "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(all_samples)} samples to {args.output_dir}/")
    print(f"   - Plots: sample_XXX.png")
    print(f"   - Data: all_samples.json")
    
    # Summary stats
    if all_samples:
        all_scores = []
        gt_scores = []
        non_gt_scores = []
        for s in all_samples:
            gs, ge = s["gt_span"]
            for i, sc in enumerate(s["scores"], start=1):
                all_scores.append(sc)
                if gs <= i <= ge:
                    gt_scores.append(sc)
                else:
                    non_gt_scores.append(sc)
        
        print(f"\n📊 Score distribution:")
        print(f"   All tokens:     mean={np.mean(all_scores):.3f} median={np.median(all_scores):.3f}")
        print(f"   GT span tokens: mean={np.mean(gt_scores):.3f} median={np.median(gt_scores):.3f} (n={len(gt_scores)})")
        print(f"   Non-GT tokens:  mean={np.mean(non_gt_scores):.3f} median={np.median(non_gt_scores):.3f} (n={len(non_gt_scores)})")


if __name__ == "__main__":
    main()

