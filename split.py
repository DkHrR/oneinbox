"""
split.py — split data/synthetic.jsonl into train (540) and val (60), stratified by label.
No real email data. Run: python split.py
"""
import json
import os
import random

SYNTHETIC_FILE = "data/synthetic.jsonl"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"

LABELS = ["must_act", "worth_a_look", "fyi", "noise"]
TOTAL_TARGET = 600
TRAIN_TARGET = 540  # 90%
VAL_TARGET = 60     # 10%
PER_LABEL_TRAIN = TRAIN_TARGET // len(LABELS)  # 135
PER_LABEL_VAL = VAL_TARGET // len(LABELS)       # 15

random.seed(42)

# Load synthetic emails
emails = []
with open(SYNTHETIC_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            emails.append(json.loads(line))

print(f"Loaded {len(emails)} synthetic emails.")

# Group by label
by_label = {lbl: [] for lbl in LABELS}
for e in emails:
    lbl = e.get("label")
    if lbl in by_label:
        by_label[lbl].append(e)

# Check we have enough
for lbl in LABELS:
    n = len(by_label[lbl])
    needed = PER_LABEL_TRAIN + PER_LABEL_VAL
    if n < needed:
        raise ValueError(f"Not enough emails for label '{lbl}': have {n}, need {needed}")

# Split per label
train_emails = []
val_emails = []

for lbl in LABELS:
    subset = by_label[lbl][:]
    random.shuffle(subset)
    val_emails.extend(subset[:PER_LABEL_VAL])
    train_emails.extend(subset[PER_LABEL_VAL:PER_LABEL_VAL + PER_LABEL_TRAIN])

# Shuffle final sets
random.shuffle(train_emails)
random.shuffle(val_emails)

# Sanity: assert no overlap
train_ids = set(json.dumps(e, sort_keys=True) for e in train_emails)
val_ids = set(json.dumps(e, sort_keys=True) for e in val_emails)
overlap = train_ids & val_ids
assert len(overlap) == 0, f"FATAL: {len(overlap)} emails appear in both train and val!"

os.makedirs("data", exist_ok=True)

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for e in train_emails:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

with open(VAL_FILE, "w", encoding="utf-8") as f:
    for e in val_emails:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

# Summary
from collections import Counter
train_labels = Counter(e["label"] for e in train_emails)
val_labels = Counter(e["label"] for e in val_emails)

print(f"\nSplit complete. No overlap confirmed.")
print(f"\nTrain ({len(train_emails)} emails):")
for lbl in LABELS:
    print(f"  {lbl:<13}: {train_labels[lbl]}")

print(f"\nVal ({len(val_emails)} emails):")
for lbl in LABELS:
    print(f"  {lbl:<13}: {val_labels[lbl]}")

print(f"\nFiles written:")
print(f"  {TRAIN_FILE}")
print(f"  {VAL_FILE}")
