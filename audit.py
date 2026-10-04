"""audit.py — full audit of data/synthetic.jsonl"""
import json, re, random, sys
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ── Date phrase detector (mirrors gen_synthetic.py exactly) ──────────────────
MONTH_NAMES = {
    'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,
    'apr':4,'april':4,'may':5,'jun':6,'june':6,'jul':7,'july':7,
    'aug':8,'august':8,'sep':9,'sept':9,'september':9,'oct':10,
    'october':10,'nov':11,'november':11,'dec':12,'december':12
}

def has_date_phrase(text: str) -> bool:
    lower = text.lower()
    m1 = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\b', lower)
    if m1 and m1.group(2) in MONTH_NAMES:
        return True
    m2 = re.search(r'\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b', lower)
    if m2 and m2.group(1) in MONTH_NAMES:
        return True
    if re.search(r'\b\d{1,2}[/-]\d{1,2}\b', lower): return True
    if re.search(r'\b(today|tonight|by eod|tomorrow)\b', lower): return True
    if re.search(r'\bwithin \d+ hours?\b', lower): return True
    if re.search(r'\b(?:within|in)\s+\d+\s*days?\b', lower): return True
    if re.search(r'\bend of (?:this )?week\b', lower): return True
    if re.search(r'\b(?:by|on|this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', lower): return True
    return False

# ── Hinglish detector — test it first ────────────────────────────────────────
# Words that exist in Hinglish but rarely in standard English in these contexts
HINGLISH_PATTERN = re.compile(
    r'\b(kar|karo|karna|karta|karti|kal|aaj|nahi|nahin|hai|hain|hoga|hogi|'
    r'kiya|kuch|bhi|abhi|bahut|accha|yaar|bhai|dena|dedo|lena|lelo|tab|tak|'
    r'wala|wale|wali|taaki|matlab|seedha|jaldi|suno|dekho|batao|lagta|lagti|'
    r'chahiye|milega|milegi|karo|karein|submit kar|review kar|bhejo|dikhao)\b',
    re.IGNORECASE
)

# Self-test: must match known Hinglish
_test_cases = [
    ("Kal tak submit kar dena please", True),
    ("PR review kar lo bhai", True),
    ("Please submit your project by tomorrow", False),
    ("Taaki further charges na lagein", True),
    ("Your invoice is ready for review", False),
]
print("=== HINGLISH DETECTOR SELF-TEST ===")
all_ok = True
for text, expected in _test_cases:
    result = bool(HINGLISH_PATTERN.search(text))
    status = "OK" if result == expected else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  [{status}] expected={expected} got={result}  | '{text}'")
if not all_ok:
    raise RuntimeError("Hinglish detector self-test FAILED — fix the pattern before proceeding.")
print("  => All self-tests passed.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
emails = []
with open('data/synthetic.jsonl', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            emails.append(json.loads(line))

print(f"Total emails loaded: {len(emails)}")

# ── Label counts ──────────────────────────────────────────────────────────────
label_counts = Counter(e['label'] for e in emails)
print(f"Label counts: {dict(label_counts)}")

# ── URGENT subjects ───────────────────────────────────────────────────────────
urgent = [e for e in emails if re.match(r'^(urgent|urgnt)[\s:!\-]', e.get('subject',''), re.IGNORECASE)]
print(f"URGENT subjects: {len(urgent)} ({len(urgent)/len(emails)*100:.1f}%)")

# ── Hinglish count ────────────────────────────────────────────────────────────
hinglish = [e for e in emails if HINGLISH_PATTERN.search(e.get('body_snippet',''))]
print(f"Hinglish emails: {len(hinglish)} ({len(hinglish)/len(emails)*100:.1f}%)")
if hinglish:
    print("  Sample Hinglish bodies:")
    for e in hinglish[:3]:
        matched = HINGLISH_PATTERN.search(e['body_snippet'])
        print(f"    match='{matched.group(0)}'  body='{e['body_snippet'][:100]}'")

# ── Deadline-null rate per label ──────────────────────────────────────────────
print("\nDeadline-null rate per label:")
for lbl in ['must_act', 'worth_a_look', 'fyi', 'noise']:
    sub = [e for e in emails if e['label'] == lbl]
    null_ct = sum(1 for e in sub if not e.get('deadline'))
    print(f"  {lbl:<13}: {null_ct}/{len(sub)} null ({null_ct/len(sub)*100:.1f}%)")

# ── Deadline consistency issues ───────────────────────────────────────────────
dl_issues = []
for e in emails:
    body = e.get('body_snippet', '')
    dl = e.get('deadline')
    body_has = has_date_phrase(body)
    lbl = e['label']
    if dl and lbl in ['fyi', 'noise']:
        dl_issues.append(('fyi_noise_has_deadline', lbl, e['subject'][:60], dl, body[:80]))
    if dl and not body_has and lbl in ['must_act', 'worth_a_look']:
        dl_issues.append(('deadline_but_no_date_phrase', lbl, e['subject'][:60], dl, body[:80]))
    if not dl and body_has and lbl in ['must_act', 'worth_a_look']:
        dl_issues.append(('date_phrase_but_no_deadline', lbl, e['subject'][:60], dl, body[:80]))

print(f"\nDeadline consistency issues: {len(dl_issues)}")
for issue in dl_issues[:10]:
    kind, lbl, subj, dl, body_preview = issue
    print(f"  [{kind}] [{lbl}] subject='{subj}' deadline={dl}")
    print(f"    body_preview: '{body_preview}'")

# ── Top 20 sender names ───────────────────────────────────────────────────────
print("\nTop 20 sender names:")
for s, c in Counter(e['from'] for e in emails).most_common(20):
    print(f"  [{c}x] {s}")

# ── Top 10 sender domains ─────────────────────────────────────────────────────
domains = Counter()
for e in emails:
    m = re.search(r'@([^>]+)', e['from'])
    if m:
        domains[m.group(1).strip()] += 1
print("\nTop 10 sender domains:")
for d, c in domains.most_common(10):
    print(f"  [{c}x] {d}")

# ── 8 random FULL records ─────────────────────────────────────────────────────
random.seed(99)
sample = random.sample(emails, 8)
print("\n" + "="*70)
print("8 RANDOM FULL RECORDS")
print("="*70)
for i, e in enumerate(sample):
    print(f"\nSample {i+1} [{e['label']}]")
    print(f"  received_date : {e.get('received_date')}")
    print(f"  from          : {e['from']}")
    print(f"  subject       : {e['subject']}")
    print(f"  body          : {e['body_snippet']}")
    print(f"  label         : {e['label']}")
    print(f"  why           : {e['why']}")
    print(f"  deadline      : {e.get('deadline')}")
    print(f"  summary       : {e['summary']}")
    print("-"*60)
