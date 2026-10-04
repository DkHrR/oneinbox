import argparse
import asyncio
import json
import os
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import tinker
from dotenv import load_dotenv

from classify import classify_email

# Documented rates from https://tinker-docs.thinkingmachines.ai/tinker/models/index.md
# Qwen3.5-4B: Prefill $0.33, Sample $1.005, Train $0.737
# Qwen3.6-35B-A3B: Prefill $0.54, Sample $1.335, Train $1.177
# Note: No separate rate is listed on Tinker docs for sampling a fine-tuned model; using the base-model rate.
MODEL_PRICING = {
    "Base Qwen3.5-4B (0-shot)": {"prompt_per_m": 0.33, "completion_per_m": 1.005},
    "Base Qwen3.5-4B (3-shot)": {"prompt_per_m": 0.33, "completion_per_m": 1.005},
    "Fine-tuned Qwen3.5-4B": {"prompt_per_m": 0.33, "completion_per_m": 1.005},
    "Base Qwen3.6-35B-A3B (0-shot)": {"prompt_per_m": 0.54, "completion_per_m": 1.335},
}

def load_data(path, limit=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            item = json.loads(line)
            item["id"] = idx
            data.append(item)
            if limit and len(data) >= limit:
                break
    return data

async def evaluate_single(sampling_client, tokenizer, item, few_shot_msgs, base_model_name):
    parsed, latency, valid_json, content, prompt_token_count, completion_token_count = await classify_email(
        sampling_client, tokenizer, item["user_message"], few_shot_msgs, base_model=bool(base_model_name)
    )
    
    return {
        "id": item["id"],
        "gold_label": item["label"],
        "pred_label": parsed.get("label", "invalid"),
        "gold_deadline": item.get("deadline"),
        "pred_deadline": parsed.get("deadline", None),
        "gold_why": item.get("why", ""),
        "pred_why": parsed.get("why", ""),
        "gold_summary": item.get("summary", ""),
        "pred_summary": parsed.get("summary", ""),
        "item_raw": item
    }, latency, valid_json, content, prompt_token_count, completion_token_count


async def run_eval(sampling_client, tokenizer, name, data, few_shot_msgs=None, base_model_name=None):
    semaphore = asyncio.Semaphore(10)
    async def bound_fetch(item):
        async with semaphore:
            return await evaluate_single(sampling_client, tokenizer, item, few_shot_msgs, base_model_name)

    tasks = [bound_fetch(item) for item in data]
    results_out = await asyncio.gather(*tasks)
    
    results = []
    latencies = []
    valid_json_count = 0
    raw_outputs = []
    errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for r, l, v, c, p_tokens, c_tokens in results_out:
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        if r is None:
            if "ERROR:" in c:
                errors += 1
                print(f"[{name}] {c}")
            continue
        if "ERROR:" in c:
            errors += 1
            print(f"[{name}] {c}")
            continue
        results.append(r)
        latencies.append(l)
        if v:
            valid_json_count += 1
        else:
            print(f"[{name}] INVALID JSON:\n{c}\n----------------------------------------")
        raw_outputs.append(c)
        
    for i, content in enumerate(raw_outputs[:3]):
        print(f"[{name}] RAW OUTPUT {i+1}:\n{content}\n")
        
    return results, latencies, valid_json_count, errors, total_prompt_tokens, total_completion_tokens

def compute_metrics(results):
    from collections import defaultdict
    labels = ["must_act", "worth_a_look", "fyi", "noise"]
    cm = defaultdict(int)
    correct = 0
    total = len(results)
    
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    
    deadline_correct = 0
    deadline_total = 0
    
    for r in results:
        g = r["gold_label"]
        p = r["pred_label"]
        cm[(g, p)] += 1
        
        if g == p:
            correct += 1
            label_tp[g] += 1
        else:
            label_fp[p] += 1
            label_fn[g] += 1
            
        if r["gold_deadline"] is not None:
            deadline_total += 1
            if r["gold_deadline"] == r["pred_deadline"]:
                deadline_correct += 1
                
    accuracy = correct / total if total else 0
    deadline_exact = deadline_correct / deadline_total if deadline_total else 0
    
    f1s = {}
    for l in labels:
        tp = label_tp[l]
        fp = label_fp[l]
        fn = label_fn[l]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s[l] = f1
        
    macro_f1 = sum(f1s.values()) / len(labels)
    
    cm_str = "Gold \\ Pred | " + " | ".join(labels) + "\n"
    cm_str += "--- | " + " | ".join(["---"]*len(labels)) + "\n"
    for g in labels:
        row = [str(cm[(g, p)]) for p in labels]
        cm_str += f"{g} | " + " | ".join(row) + "\n"
        
    return correct, total, accuracy, macro_f1, f1s, cm_str, deadline_correct, deadline_total, deadline_exact

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--ft-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    tinker_key = os.environ.get("TINKER_API_KEY")
    if not tinker_key:
        print("Missing TINKER_API_KEY")
        sys.exit(1)

    eval_path = "data/val.jsonl" if args.val else "data/test.jsonl"
    if not os.path.exists(eval_path):
        print(f"Error: {eval_path} missing.")
        sys.exit(1)
        
    test_data = load_data(eval_path, args.limit)
    train_data = load_data("data/train.jsonl")

    few_shot_msgs = []
    for item in train_data[:3]:
        few_shot_msgs.append({"role": "user", "content": item["user_message"]})
        ans = {"label": item["label"], "why": item["why"], "deadline": item.get("deadline"), "summary": item.get("summary", "")}
        few_shot_msgs.append({"role": "assistant", "content": json.dumps(ans)})
        
    ft_model = None
    if os.path.exists("checkpoint.txt"):
        with open("checkpoint.txt", "r") as f:
            ft_model = f.read().strip()

    models_to_eval = []
    if not args.ft_only:
        models_to_eval.extend([
            ("Base Qwen3.5-4B (0-shot)", "Qwen/Qwen3.5-4B", None, None),
            ("Base Qwen3.5-4B (3-shot)", "Qwen/Qwen3.5-4B", None, few_shot_msgs),
        ])
    if ft_model:
        models_to_eval.append(("Fine-tuned Qwen3.5-4B", None, ft_model, None))
    if not args.ft_only:
        models_to_eval.append(("Base Qwen3.6-35B-A3B (0-shot)", "Qwen/Qwen3.6-35B-A3B", None, None))

    eval_data = test_data
    num_eval_items = len(eval_data)
    
    # Pre-run cost estimation based on expected tokens and rates
    est_total_cost = 0.0
    for name, _, _, f_shots in models_to_eval:
        rates = MODEL_PRICING[name]
        p_tok = 750 if f_shots else 150
        c_tok = 45
        cost_per_item = (p_tok * rates["prompt_per_m"] + c_tok * rates["completion_per_m"]) / 1_000_000
        est_total_cost += cost_per_item * num_eval_items
        
    print(f"Estimated total cost for this eval run ({num_eval_items} items x {len(models_to_eval)} variants): ~${est_total_cost:.4f}")
    if est_total_cost > 0.15:
        print(f"Cost estimate (${est_total_cost:.4f}) exceeds $0.15 cap. Exiting.")
        sys.exit(1)

    service_client = tinker.ServiceClient()

    all_metrics = []
    all_predictions_to_save = []
    ft_wrong_emails = []
    
    for name, base_model, checkpoint, f_shots in models_to_eval:
        print(f"\nEvaluating {name}...")
        if checkpoint:
            sampling_client = service_client.create_sampling_client(model_path=checkpoint)
            tokenizer = sampling_client.get_tokenizer()
            is_base = False
        else:
            sampling_client = service_client.create_sampling_client(base_model=base_model)
            tokenizer = sampling_client.get_tokenizer()
            is_base = True
            
        results, latencies, valid_json, errors, tot_prompt, tot_comp = await run_eval(
            sampling_client, tokenizer, name, eval_data, f_shots, base_model_name=base_model if is_base else None
        )

        if len(results) == 0:
            print(f"No results for {name}.")
            continue
            
        correct, total, acc, macro_f1, f1s, cm_str, dead_corr, dead_tot, deadline_exact = compute_metrics(results)
        p50 = np.percentile(latencies, 50) if latencies else 0
        
        validity_rate = valid_json / len(eval_data)
        print(f"Validity rate for {name}: {valid_json}/{len(eval_data)} ({validity_rate:.2%}) (Errors: {errors}/{len(eval_data)})")
        
        # Calculate actual cost per 1,000 emails from recorded tokens
        rates = MODEL_PRICING[name]
        avg_prompt_tokens = tot_prompt / len(eval_data)
        avg_completion_tokens = tot_comp / len(eval_data)
        cost_per_email = (avg_prompt_tokens * rates["prompt_per_m"] + avg_completion_tokens * rates["completion_per_m"]) / 1_000_000
        cost_per_1k = cost_per_email * 1000
        cost_calc_explanation = (
            f"${cost_per_1k:.4f} per 1k emails [Avg {avg_prompt_tokens:.1f} prompt tokens @ ${rates['prompt_per_m']}/M + "
            f"avg {avg_completion_tokens:.1f} completion tokens @ ${rates['completion_per_m']}/M]"
        )

        all_metrics.append((
            name, correct, total, acc, macro_f1, f1s, valid_json, dead_corr, dead_tot, deadline_exact, p50, cm_str, errors, cost_per_1k, cost_calc_explanation
        ))

        # Record predictions
        for r in results:
            all_predictions_to_save.append({
                "id": r["id"],
                "variant": name,
                "gold_label": r["gold_label"],
                "predicted_label": r["pred_label"],
                "gold_deadline": r["gold_deadline"],
                "predicted_deadline": r["pred_deadline"]
            })
            if "Fine-tuned" in name:
                label_wrong = (r["gold_label"] != r["pred_label"])
                deadline_wrong = (r["gold_deadline"] != r["pred_deadline"])
                if label_wrong or deadline_wrong:
                    ft_wrong_emails.append(r)

    # Save predictions to data/eval_predictions.jsonl (or partial if --limit)
    pred_path = "data/eval_predictions_partial.jsonl" if args.limit is not None else "data/eval_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as pf:
        for p in all_predictions_to_save:
            pf.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(all_predictions_to_save)} predictions to {pred_path}.")

    # Print fine-tuned wrong emails
    print(f"\n=======================================================")
    print(f"FINE-TUNED MODEL DISCREPANCIES ({len(ft_wrong_emails)} total):")
    print(f"=======================================================")
    for idx, item in enumerate(ft_wrong_emails, 1):
        raw = item["item_raw"]
        print(f"\n--- WRONG EMAIL #{idx} (ID: {item['id']}) ---")
        print(f"From: {raw.get('from', '')}")
        print(f"Subject: {raw.get('subject', '')}")
        print(f"Received Date: {raw.get('received_date', '')}")
        print(f"Body: {raw.get('body', '')}")
        print(f"Gold: label='{item['gold_label']}', deadline='{item['gold_deadline']}', why='{item['gold_why']}'")
        print(f"Predicted: label='{item['pred_label']}', deadline='{item['pred_deadline']}', why='{item['pred_why']}'")
        print(f"Error Type: {'Label mismatch' if item['gold_label'] != item['pred_label'] else 'Deadline mismatch'}")

    out_results_path = "results_partial.md" if args.limit is not None else "results.md"
    with open(out_results_path, "w", encoding="utf-8") as out:
        out.write("# OneInbox Evaluation Results\n")
        out.write(f"**Dataset**: `{eval_path}` ({len(eval_data)} emails)\n")
        out.write(f"**Latency Benchmark**: Measured in parallel (concurrency 10 via asyncio semaphore)\n")
        out.write(f"**Pricing Source**: https://tinker-docs.thinkingmachines.ai/tinker/models/index.md (October 2026)\n")
        out.write(f"**Note on Fine-Tuned Pricing**: No separate rate is listed on Tinker docs for sampling a fine-tuned model, so the base-model rate ($0.33/M prompt, $1.005/M sample) is used.\n\n")

        for name, correct, total, acc, macro_f1, f1s, valid_json, dead_corr, dead_tot, deadline_exact, p50, cm_str, errors, cost_per_1k, cost_calc_explanation in all_metrics:
            out.write(f"## {name}\n")
            out.write(f"- Label Accuracy: {correct}/{total} ({acc:.2%})\n")
            out.write(f"- Macro-F1: {macro_f1:.4f}\n")
            for k, v in f1s.items():
                out.write(f"  - {k}: {v:.4f}\n")
            out.write(f"- JSON Validity Rate: {valid_json}/{total} ({valid_json/total:.2%})\n")
            out.write(f"- Deadline Exact-Match: {dead_corr}/{dead_tot} ({deadline_exact:.2%}) [{dead_tot} emails have a non-null gold deadline]\n")
            out.write(f"- Infrastructure Errors: {errors}/{total}\n")
            out.write(f"- p50 Latency: {p50:.2f}s per email (measured in parallel, concurrency 10)\n")
            out.write(f"- Cost per 1,000 emails: ${cost_per_1k:.4f}\n")
            out.write(f"  - Cost Computation: {cost_calc_explanation}\n\n")
            out.write("### Confusion Matrix\n")
            out.write(cm_str + "\n\n")

if __name__ == "__main__":
    asyncio.run(main())
