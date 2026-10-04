---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- email
- classification
- lora
- peft
- tinker
---

# OneInbox Qwen3.5-4B LoRA Adapter

This repository contains LoRA adapter weights fine-tuned from **Qwen/Qwen3.5-4B** using **Tinker**. The adapter powers [OneInbox](https://github.com/DkHrR/oneinbox), an intelligent, privacy-first daily email digest assistant that classifies incoming messages into actionable priority tiers and extracts strict deadlines.

## Training Details

- **Base Model**: `Qwen/Qwen3.5-4B`
- **Training Framework**: Tinker Supervised LoRA Training
- **Training Set**: 540 diverse synthetic emails curated to reflect realistic email structures, edge cases, urgent requests, and newsletters
- **Adapter Configuration**: Rank $r=32$, Alpha $\alpha=32$, targeting `all-linear` modules

## Classification Labels & Schema

The model classifies an email into one of four labels:
- `must_act`: Immediate action required (bills, account suspension notices, security alerts, calendar invitations with impending deadlines).
- `worth_a_look`: High-value personal opportunities, relevant technical releases, developer meetups, or beta invites.
- `fyi`: Informational notifications requiring no action (receipts, change password confirmations, closed issue updates).
- `noise`: Marketing blasts, promotional discounts, routine automated newsletters, and social clutter.

### Output JSON Format
```json
{
  "label": "<must_act | worth_a_look | fyi | noise>",
  "why": "<under 10 words explanation>",
  "deadline": "<YYYY-MM-DD or null>",
  "summary": "<concise one-line summary>"
}
```

## Evaluation Benchmark

Evaluated on `data/val.jsonl` (60 held-out synthetic emails, 28 containing non-null gold deadlines). Latency was benchmarked in parallel with concurrency 10.

| Variant | Label Accuracy | Macro-F1 | JSON Validity | Deadline Exact-Match | p50 Latency | Cost / 1k Emails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Qwen3.5-4B (0-shot)** | 52/60 (86.67%) | 0.8645 | 60/60 (100.00%) | 23/28 (82.14%) | 2.58s | $0.1569 |
| **Base Qwen3.5-4B (3-shot)** | 56/60 (93.33%) | 0.9328 | 60/60 (100.00%) | 25/28 (89.29%) | 2.55s | $0.2776 |
| **Fine-tuned Qwen3.5-4B (Ours)** | **58/60 (96.67%)** | **0.9666** | **60/60 (100.00%)** | **25/28 (89.29%)** | 2.85s | **$0.1412** |
| **Qwen3.6-35B-A3B (0-shot)** | 51/60 (85.00%) | 0.8486 | 60/60 (100.00%) | 25/28 (89.29%) | 2.15s | $0.2286 |

### Fine-Tuned Per-Label Breakdown
- **must_act**: Precision 1.0000 | Recall 1.0000 | F1: **1.0000**
- **worth_a_look**: Precision 0.9333 | Recall 0.9333 | F1: **0.9333**
- **fyi**: Precision 1.0000 | Recall 0.9333 | F1: **0.9655**
- **noise**: Precision 0.9375 | Recall 1.0000 | F1: **0.9677**

## Honest Limitations

1. **Synthetic Benchmark & Generator-Made Labels**: The 540 training emails and 60 validation emails are synthetically generated, and the test set is synthetic with generator-made labels. While designed with realistic edge cases, urgent deadlines, and varied phrasing, performance on real-world email distributions with noisy formatting or multipart HTML may vary.
2. **Untested on Real Inboxes**: IMAP mode is untested on a real inbox due to privacy considerations.
3. **Feedback Behavior**: Accurate/Wrong feedback is logged to Backboard but does not change filtering (only deterministic user preference rules like `always_show` and `ignore` modify routing).
4. **Digest Memory Limit**: The digest snapshot is limited to about 31 emails by Backboard's 4,000-character memory capacity.
5. **Inference Pricing Assumption**: The official Tinker pricing documentation lists base Qwen3.5-4B rates ($0.33/M prompt tokens, $1.005/M completion tokens) but does not provide a distinct rate for sampling fine-tuned checkpoints. The reported cost per 1,000 emails ($0.1412) assumes base model pricing.
