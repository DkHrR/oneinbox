# OneInbox

OneInbox is an experimental email classification project that uses Large Language Models to automatically categorize incoming emails into actionable labels.

## Setup
1. Create a `.env` file with `TINKER_API_KEY` and `BACKBOARD_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`

## How to Train
We fine-tune Qwen3.5-4B using Tinker:
```bash
python train.py --yes
```

## How to Evaluate
You can evaluate zero-shot, few-shot, and fine-tuned variants:
```bash
python eval.py --val
```

## Note on Evaluation Data
**Honest Note:** The validation set (`data/val.jsonl`) consists entirely of synthetic emails. None of this has been evaluated on a real inbox yet. The performance metrics reflect the model's ability to classify the generated data distribution and might not generalize cleanly to actual noisy inbox data.
