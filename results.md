# OneInbox Evaluation Results
**Dataset**: `data/val.jsonl` (60 emails)
**Latency Benchmark**: Measured in parallel (concurrency 10 via asyncio semaphore)
**Pricing Source**: https://tinker-docs.thinkingmachines.ai/tinker/models/index.md (October 2026)
**Note on Fine-Tuned Pricing**: No separate rate is listed on Tinker docs for sampling a fine-tuned model, so the base-model rate ($0.33/M prompt, $1.005/M sample) is used.

## Base Qwen3.5-4B (0-shot)
- Label Accuracy: 52/60 (86.67%)
- Macro-F1: 0.8645
  - must_act: 0.8824
  - worth_a_look: 0.9333
  - fyi: 0.8148
  - noise: 0.8276
- JSON Validity Rate: 60/60 (100.00%)
- Deadline Exact-Match: 23/28 (82.14%) [28 emails have a non-null gold deadline]
- Infrastructure Errors: 0/60
- p50 Latency: 2.58s per email (measured in parallel, concurrency 10)
- Cost per 1,000 emails: $0.1569
  - Cost Computation: $0.1569 per 1k emails [Avg 283.4 prompt tokens @ $0.33/M + avg 63.1 completion tokens @ $1.005/M]

### Confusion Matrix
Gold \ Pred | must_act | worth_a_look | fyi | noise
--- | --- | --- | --- | ---
must_act | 15 | 0 | 0 | 0
worth_a_look | 1 | 14 | 0 | 0
fyi | 2 | 0 | 11 | 2
noise | 1 | 1 | 1 | 12


## Base Qwen3.5-4B (3-shot)
- Label Accuracy: 56/60 (93.33%)
- Macro-F1: 0.9328
  - must_act: 0.9677
  - worth_a_look: 0.9655
  - fyi: 0.8889
  - noise: 0.9091
- JSON Validity Rate: 60/60 (100.00%)
- Deadline Exact-Match: 25/28 (89.29%) [28 emails have a non-null gold deadline]
- Infrastructure Errors: 0/60
- p50 Latency: 2.55s per email (measured in parallel, concurrency 10)
- Cost per 1,000 emails: $0.2776
  - Cost Computation: $0.2776 per 1k emails [Avg 703.5 prompt tokens @ $0.33/M + avg 45.3 completion tokens @ $1.005/M]

### Confusion Matrix
Gold \ Pred | must_act | worth_a_look | fyi | noise
--- | --- | --- | --- | ---
must_act | 15 | 0 | 0 | 0
worth_a_look | 0 | 14 | 0 | 1
fyi | 1 | 0 | 12 | 2
noise | 0 | 0 | 0 | 15


## Fine-tuned Qwen3.5-4B
- Label Accuracy: 58/60 (96.67%)
- Macro-F1: 0.9666
  - must_act: 1.0000
  - worth_a_look: 0.9333
  - fyi: 0.9655
  - noise: 0.9677
- JSON Validity Rate: 60/60 (100.00%)
- Deadline Exact-Match: 25/28 (89.29%) [28 emails have a non-null gold deadline]
- Infrastructure Errors: 0/60
- p50 Latency: 2.85s per email (measured in parallel, concurrency 10)
- Cost per 1,000 emails: $0.1412
  - Cost Computation: $0.1412 per 1k emails [Avg 279.4 prompt tokens @ $0.33/M + avg 48.7 completion tokens @ $1.005/M]

### Confusion Matrix
Gold \ Pred | must_act | worth_a_look | fyi | noise
--- | --- | --- | --- | ---
must_act | 15 | 0 | 0 | 0
worth_a_look | 0 | 14 | 0 | 1
fyi | 0 | 1 | 14 | 0
noise | 0 | 0 | 0 | 15


## Base Qwen3.6-35B-A3B (0-shot)
- Label Accuracy: 51/60 (85.00%)
- Macro-F1: 0.8486
  - must_act: 0.9375
  - worth_a_look: 0.8462
  - fyi: 0.8000
  - noise: 0.8108
- JSON Validity Rate: 60/60 (100.00%)
- Deadline Exact-Match: 25/28 (89.29%) [28 emails have a non-null gold deadline]
- Infrastructure Errors: 0/60
- p50 Latency: 2.15s per email (measured in parallel, concurrency 10)
- Cost per 1,000 emails: $0.2286
  - Cost Computation: $0.2286 per 1k emails [Avg 283.4 prompt tokens @ $0.54/M + avg 56.6 completion tokens @ $1.335/M]

### Confusion Matrix
Gold \ Pred | must_act | worth_a_look | fyi | noise
--- | --- | --- | --- | ---
must_act | 15 | 0 | 0 | 0
worth_a_look | 0 | 11 | 0 | 4
fyi | 2 | 0 | 10 | 3
noise | 0 | 0 | 0 | 15


