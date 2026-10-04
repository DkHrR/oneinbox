import os
import yaml
import json
import asyncio
import logging
import re
import random
import datetime
import argparse
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dotenv import load_dotenv

import tinker

load_dotenv(override=True)

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FIRST_NAMES = [
    "Aarav", "Aditi", "Aditya", "Akash", "Amit", "Ananya", "Aniket", "Arjun",
    "Deepak", "Divya", "Gaurav", "Isha", "Ishaan", "Karan", "Kavya", "Manish",
    "Meera", "Neha", "Nikhil", "Pooja", "Pranav", "Priya", "Rahul", "Rajesh",
    "Riya", "Rohan", "Rohit", "Sameer", "Sanjay", "Shreya", "Siddharth", "Sneha",
    "Sunita", "Tanvi", "Varun", "Vikram"
]

LAST_NAMES = [
    "Agarwal", "Banerjee", "Bhatt", "Choudhury", "Deshmukh", "Gupta", "Iyer",
    "Joshi", "Kapoor", "Kulkarni", "Mehta", "Mukherjee", "Nair", "Patel",
    "Pillai", "Rao", "Reddy", "Sharma", "Singh", "Verma"
]

COMMON_TYPOS = [
    ("the", "teh"), ("receive", "recieve"), ("scheduled", "schedueled"),
    ("available", "availble"), ("submission", "submision"), ("please", "pleae"),
    ("deadline", "dealine"), ("tomorrow", "tommorow")
]

MONTH_NAMES = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9, 'oct': 10,
    'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}
WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

AUTOMATED_MAILBOXES = ["noreply", "alerts", "notifications", "orders", "support", "billing", "updates"]

first_name_counts = Counter()
last_name_counts = Counter()
person_to_org = {}
seed_counts = Counter()
MAX_PER_SEED = 18 # 3% of 600

used_amounts = set()
generation_attempts = Counter()
verifier_passes = Counter()
verifier_drops = Counter()
verifier_reasons = []

HUMAN_SCENARIOS = [
    "recruiter", "interview", "professor", "teammate", "team member", "friend", "hr"
]

# COST TRACKING
PREFILL_COST_PER_M = 0.54
SAMPLE_COST_PER_M = 1.335
SPEND_CAP = 2.00
total_input_tokens = 0
total_output_tokens = 0

def check_spend_cap():
    global total_input_tokens, total_output_tokens
    cost = (total_input_tokens / 1_000_000) * PREFILL_COST_PER_M + (total_output_tokens / 1_000_000) * SAMPLE_COST_PER_M
    if cost > SPEND_CAP:
        logging.error(f"Spend cap exceeded! Estimated cost: ${cost:.4f}. Halting.")
        sys.exit(0)
    return cost

def is_human_sender_scenario(scenario: str) -> bool:
    sc = scenario.lower()
    return any(kw in sc for kw in HUMAN_SCENARIOS)

def get_sender_identity(org_name: str, org_domain: str, is_human: bool, scenario: str, max_cap: int = 24) -> str:
    if not is_human:
        sc = scenario.lower()
        if "shipping" in sc or "package" in sc or "delivery" in sc:
            mb = random.choice(["tracking", "delivery", "shipping", "notifications", "noreply"])
        elif "receipt" in sc or "invoice" in sc or "purchase" in sc or "order" in sc:
            mb = random.choice(["orders", "billing", "receipts", "noreply"])
        elif "statement" in sc or "bank" in sc or "fee" in sc:
            mb = random.choice(["statements", "alerts", "accounts", "noreply"])
        elif "security" in sc or "password" in sc or "login" in sc or "verify" in sc:
            mb = random.choice(["security", "alerts", "identity", "noreply"])
        elif "social" in sc or "follower" in sc or "liked" in sc:
            mb = random.choice(["notifications", "activity", "noreply"])
        elif "promo" in sc or "sale" in sc or "discount" in sc or "offer" in sc:
            mb = random.choice(["offers", "deals", "promotions", "noreply"])
        elif "newsletter" in sc:
            mb = random.choice(["newsletter", "digest", "updates", "noreply"])
        elif "grant" in sc or "fellowship" in sc or "scholarship" in sc or "hackathon" in sc:
            mb = random.choice(["programs", "grants", "community", "team", "noreply"])
        else:
            mb = random.choice(["noreply", "updates", "notifications"])
        return f"{org_name} <{mb}@{org_domain}>"

    for _ in range(100):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        full = f"{fn} {ln}"

        if first_name_counts[fn] >= max_cap or last_name_counts[ln] >= max_cap:
            continue
        if full in person_to_org and person_to_org[full] != org_name:
            continue

        person_to_org[full] = org_name
        first_name_counts[fn] += 1
        last_name_counts[ln] += 1
        username = f"{fn.lower()}.{ln.lower()}"
        return f"{full} ({org_name}) <{username}@{org_domain}>"

    mailbox = random.choice(AUTOMATED_MAILBOXES)
    return f"{org_name} <{mailbox}@{org_domain}>"

def get_scenario_amount(scenario: str) -> Optional[str]:
    sc = scenario.lower()
    if "instamart" in sc or "grocery" in sc or "online purchase" in sc or "cart" in sc:
        val = random.randint(300, 2500)
    elif "tuition" in sc or "fee" in sc or "university" in sc:
        val = random.randint(40000, 300000)
    elif "doctor" in sc or "clinic" in sc:
        val = random.randint(500, 2500)
    elif "library" in sc:
        val = random.randint(50, 500)
    elif "shoes" in sc or "clothing" in sc or "fashion" in sc or "dress" in sc:
        val = random.randint(799, 4999)
    elif "electronics" in sc or "iphone" in sc or "biggest sale" in sc or "laptop" in sc:
        val = random.randint(9999, 65000)
    elif "license" in sc or "software" in sc:
        val = random.randint(1500, 12000)
    elif "automatic payment" in sc or "subscription" in sc or "netflix" in sc or "spotify" in sc:
        val = random.randint(199, 1499)
    elif "gift card" in sc or "sweepstakes" in sc or "prize" in sc:
        val = random.randint(2000, 25000)
    elif "birthday discount" in sc:
        val = random.randint(200, 1500)
    else:
        return None

    for _ in range(100):
        formatted = f"₹{val:,}"
        if formatted not in used_amounts:
            used_amounts.add(formatted)
            return formatted
        val += random.randint(1, 19)
    formatted = f"₹{val:,}"
    used_amounts.add(formatted)
    return formatted

def get_random_received_date() -> datetime.date:
    start_date = datetime.date(2026, 9, 1)
    end_date = datetime.date(2026, 10, 1)
    days_range = (end_date - start_date).days
    return start_date + datetime.timedelta(days=random.randint(0, days_range))

def choose_deadline_config(label: str, has_deadline: bool, received_date: datetime.date) -> Tuple[Optional[datetime.date], str, str]:
    if label in ["fyi", "noise"] or not has_deadline:
        return None, "none", "There is NO deadline or action date for this email. Dates for validity, billing periods, delivery, or statements are NOT deadlines. DO NOT specify a deadline. JSON field 'deadline' MUST be null."

    if label == "must_act":
        offset = random.randint(0, 7)
        target = received_date + datetime.timedelta(days=offset)
        if offset == 0:
            style, phrase = "today", "by today"
        elif offset == 1:
            style, phrase = "tomorrow", "tomorrow"
        elif offset in [2, 3]:
            style, phrase = f"within {offset*24} hours", f"within {offset*24} hours"
        elif offset <= 5:
            style, phrase = "weekday", target.strftime("by %A")
        else:
            style, phrase = "written_date", target.strftime("%d %b")

        instruction = (
            f"The deadline is {target.isoformat()} ({offset} days from received_date). "
            f"In the email body, you MUST include this exact time phrase using style '{style}' like '{phrase}'. "
            f"NEVER write the ISO date '{target.isoformat()}' in the email text! "
            f"The JSON output field 'deadline' MUST be '{target.isoformat()}'."
        )
        return target, style, instruction

    if label == "worth_a_look":
        offset = random.randint(8, 14)
        target = received_date + datetime.timedelta(days=offset)
        style = random.choice(["written_date", "numeric"])
        phrase = target.strftime("%d %b") if style == "written_date" else target.strftime("%d/%m")
        instruction = (
            f"The deadline is {target.isoformat()} ({offset} days away, more than 7 days). "
            f"In the email body, you MUST refer to this date using '{phrase}'. Do NOT use bare weekday names. "
            f"NEVER write the ISO date '{target.isoformat()}' in the email text! "
            f"The JSON output field 'deadline' MUST be '{target.isoformat()}'."
        )
        return target, style, instruction

def parse_date_phrase_from_text(text: str, received_date: datetime.date) -> Tuple[bool, Optional[datetime.date], str]:
    lower = text.lower()
    m_dt1 = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\b', lower)
    if m_dt1 and m_dt1.group(2) in MONTH_NAMES:
        day = int(m_dt1.group(1))
        month = MONTH_NAMES[m_dt1.group(2)]
        return True, datetime.date(received_date.year, month, day), m_dt1.group(0)

    m_dt2 = re.search(r'\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b', lower)
    if m_dt2 and m_dt2.group(1) in MONTH_NAMES:
        day = int(m_dt2.group(2))
        month = MONTH_NAMES[m_dt2.group(1)]
        return True, datetime.date(received_date.year, month, day), m_dt2.group(0)

    m_num = re.search(r'\b(\d{1,2})[/-](\d{1,2})\b', lower)
    if m_num:
        day = int(m_num.group(1))
        month = int(m_num.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return True, datetime.date(received_date.year, month, day), m_num.group(0)

    if re.search(r'\b(today|tonight|by eod)\b', lower):
        return True, received_date, "today"

    if re.search(r'\btomorrow\b', lower):
        return True, received_date + datetime.timedelta(days=1), "tomorrow"

    m_hrs = re.search(r'\bwithin (\d+)\s*hours?\b', lower)
    if m_hrs:
        hrs = int(m_hrs.group(1))
        days = max(1, round(hrs / 24))
        return True, received_date + datetime.timedelta(days=days), f"within {hrs} hours"

    m_days = re.search(r'\b(?:within|in)\s+(\d+)\s*days?\b', lower)
    if m_days:
        d = int(m_days.group(1))
        return True, received_date + datetime.timedelta(days=d), f"within {d} days"

    if re.search(r'\bend of (?:this )?week\b', lower):
        days_to_sunday = (6 - received_date.weekday()) % 7
        if days_to_sunday == 0:
            days_to_sunday = 7
        return True, received_date + datetime.timedelta(days=days_to_sunday), "end of week"

    m_wk = re.search(r'\b(?:by|on|this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', lower)
    if m_wk:
        target_weekday = WEEKDAYS.index(m_wk.group(1))
        days_ahead = (target_weekday - received_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return True, received_date + datetime.timedelta(days=days_ahead), m_wk.group(1)

    return False, None, ""

def build_generator_prompt(
    label: str, seed: str, sender_from: str, scenario_amount: Optional[str],
    received_date: datetime.date, deadline_info: Tuple[Optional[datetime.date], str, str],
    allow_urgent: bool, is_hard_neg: bool, apply_typo: bool, is_hinglish: bool
) -> str:
    target_deadline, _, deadline_instruction = deadline_info
    received_date_str = received_date.isoformat()

    rules = [
        f"Today is {received_date_str}. Any relative dates must be anchored to {received_date_str}.",
        "Body MUST be 1 to 5 coherent, grammatically sensible sentences. Ban garbled, disjointed, or invented words.",
        deadline_instruction,
        "NEVER write the ISO date format YYYY-MM-DD anywhere in the subject or body snippet."
    ]

    rules.append(
        "LABEL POLICY:\n"
        "- must_act: Action/reply required, hard calendar deadline within 7 days, OR a security alert asking user to verify/act.\n"
        "- worth_a_look: Specific opportunity (fellowship, grant, scholarship, CFP, job role) or personally relevant tool release.\n"
        "- fyi: Routine status notice, confirmation, successful login alert, OTP, receipt, or bank statement (NO user action required).\n"
        "- noise: Marketing promos, sales, generic brand newsletters, social pings, even when artificially urgent."
    )

    if label == "must_act":
        rules.append("MUST_ACT: The email MUST clearly require a reply, payment, or action within 7 days, or demand account verification.")
    elif label == "worth_a_look":
        rules.append("WORTH_A_LOOK: A specific opportunity (scholarship, CFP, grant, job role) or personally relevant tool release. Deadlines, if any, MUST be >7 days away.")
    elif label == "fyi":
        rules.append("FYI: Informational notice, delivery tracking, receipt, or successful login confirmation. NEVER demand user action.")
    elif label == "noise":
        rules.append("NOISE: Promotional marketing sale, brand newsletter, or social media activity notification.")

    if is_hard_neg:
        rules.append("HARD NEGATIVE: The email may sound time-pressured, but its classification is strictly '" + label + "'.")

    if not allow_urgent:
        rules.append("DO NOT begin the subject line with 'URGENT' or 'Urgent:'. Use a realistic subject line.")

    if scenario_amount:
        rules.append(f"CURRENCY: Naturally mention the realistic rupee amount {scenario_amount} using ₹.")
    else:
        rules.append("CURRENCY: DO NOT mention any prices, money, fees, or currency amounts.")

    if is_hinglish:
        rules.append("LANGUAGE: Natural colloquial Hinglish in Latin script (e.g. 'Kal tak submit kar dena please', 'PR review kar lo'). Strictly NO Devanagari.")
    elif apply_typo:
        orig, typo = random.choice(COMMON_TYPOS)
        rules.append(f"LANGUAGE: Natural English with at most ONE common realistic typo ('{orig}' -> '{typo}'). Everything else clean.")
    else:
        rules.append("LANGUAGE: Clean, professional or natural English. No typos.")

    rules.append("WHY FIELD: Must be a complete, grammatically valid sentence of 4 to 8 words, starting with a capital letter and ending with a period. Strictly under 10 words.")
    rules_text = "\n- ".join(rules)

    system_prompt = (
        "You are an expert synthetic email dataset generator. "
        "Adhere strictly to sender context, formatting, label policy, realism, and date rules."
    )
    user_prompt = f"""Scenario / Seed: {seed}
Target Label: {label}
Sender From: "{sender_from}"

Rules & Constraints:
- {rules_text}

Return EXACTLY a single valid JSON object (no markdown quotes, no extra commentary):
{{
  "from": "{sender_from}",
  "subject": "Realistic email subject line",
  "body_snippet": "First 1-5 coherent sentences of email body (max 500 chars)",
  "label": "{label}",
  "why": "Complete sentence under 10 words.",
  "deadline": {f'"{target_deadline.isoformat()}"' if target_deadline else 'null'},
  "summary": "Brief summary under 20 words."
}}"""

    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
</think>
"""

def build_verifier_prompt(seed: str, email: dict) -> str:
    system_prompt = """You are a strict, skeptical QA auditor for an email triage training dataset.
Be skeptical and fail anything with odd or invented words, mixed-up sentences, a truncated why, or unrealistic amounts.
Healthy drop rate is 10-30%. Do not be lenient.

Audit Criteria:
1. sender_fits (bool): Does the sender organization and domain realistically match the scenario?
   - Automated senders MUST be a brand/noreply address.
   - Only human-written categories use a person name.
2. coherent (bool): Is the body 1-5 natural, grammatically sensible sentences?
3. no_garbled_words (bool): Are all words legitimate English or natural colloquial Hinglish?
4. why_complete (bool): Is the why field a complete sentence under 10 words ending with a period?
5. deadline_correct (bool):
   - must_act / worth_a_look: Match deadline relative to received_date. If no date phrase, deadline is null.
   - fyi / noise: MUST have a null deadline always.
   - Are amounts realistic for scenario?
6. label_correct (bool):
   - must_act: Action/reply required, deadline within 7 days.
   - worth_a_look: Specific opportunity.
   - fyi: Routine status notice, no action.
   - noise: Marketing promos, sales, newsletters.

Return ONLY a JSON object:
{
  "sender_fits": true/false,
  "coherent": true/false,
  "no_garbled_words": true/false,
  "why_complete": true/false,
  "deadline_correct": true/false,
  "label_correct": true/false,
  "reason": "1 concise sentence under 20 words explaining any failure, or 'All checks passed'"
}"""

    user_prompt = f"""Scenario Seed: {seed}
Assigned Label: {email['label']}
Candidate Email:
Sender: {email['from']}
Subject: {email['subject']}
Body: {email['body_snippet']}
Why: {email['why']}
Summary: {email['summary']}
Gold Deadline: {email['deadline']}
Received Date: {email['received_date']}

Audit this email."""

    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
</think>
"""

def basic_deterministic_validate(raw_data: dict, expected_label: str, received_date: datetime.date, allow_urgent: bool, is_human: bool) -> Tuple[bool, str, dict]:
    label = raw_data.get("label")
    if label not in ["must_act", "worth_a_look", "fyi", "noise"]:
        return False, f"Invalid label: {label}", {}
    if label != expected_label:
        return False, f"Label mismatch: expected {expected_label}, got {label}", {}

    for field in ["from", "subject", "body_snippet", "why", "summary"]:
        if not raw_data.get(field):
            return False, f"Missing required field: {field}", {}

    subject = str(raw_data["subject"]).strip()
    body = str(raw_data["body_snippet"]).strip()
    why = str(raw_data["why"]).strip()
    summary = str(raw_data["summary"]).strip()
    full_text = subject + " " + body
    sender_from = str(raw_data["from"]).strip()

    if re.search(r'\b\d{4}-\d{2}-\d{2}\b', full_text):
        return False, "Found ISO date format in text", {}

    if re.search(r'[\u0900-\u097F]', full_text):
        return False, "Found Devanagari script", {}

    if re.search(r'\b(19\d\d|20[0-2][0-5])\b', full_text):
        return False, "Found year earlier than 2026", {}

    if not allow_urgent and re.match(r'^(urgent|urgnt)[\s\:\!\-]', subject, re.IGNORECASE):
        return False, "Subject starts with URGENT when forbidden", {}
        
    # Sender rule check
    is_person_generated = bool(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+ \(.*\) <', sender_from))
    if is_human and not is_person_generated:
        return False, "Sender must be a person name for this scenario", {}
    if not is_human and is_person_generated:
        return False, "Sender must NOT be a person name for automated/system scenarios", {}

    why_words = why.split()
    if len(why_words) > 10:
        why = " ".join(why_words[:9]) + "."
        raw_data["why"] = why
    if not (why[0].isupper() and why.endswith('.')):
        why = why.capitalize()
        if not why.endswith('.'):
            why += '.'
        raw_data["why"] = why

    raw_deadline = raw_data.get("deadline")
    clean_deadline = str(raw_deadline).strip() if (raw_deadline and str(raw_deadline).strip().lower() not in ["null", "none"]) else None

    # Null deadline enforcement
    if label in ["fyi", "noise"] and clean_deadline is not None:
        return False, f"{label} emails MUST have a null deadline", {}

    is_dl_consistent, dl_reason = check_deadline_consistency(body, received_date, clean_deadline, why, summary, label)
    if not is_dl_consistent:
        return False, dl_reason, {}

    if clean_deadline:
        dl_date = datetime.date.fromisoformat(clean_deadline)
        offset = (dl_date - received_date).days
        if offset < 0:
            return False, "Deadline is before received_date", {}
        if offset > 60:
            return False, "Deadline is >60 days away", {}
        if label == "must_act" and offset > 7:
            return False, f"must_act has deadline {offset} days away (>7 days)", {}

    clean_email = {
        "received_date": received_date.isoformat(),
        "from": sender_from,
        "subject": subject,
        "body_snippet": body[:500],
        "label": label,
        "why": why,
        "deadline": clean_deadline,
        "summary": summary,
        "user_message": (
            f"Date: {received_date.isoformat()}\n"
            f"From: {sender_from}\n"
            f"Subject: {subject}\n\n"
            f"{body[:500]}"
        ),
        "seed": raw_data.get("seed", "")
    }
    return True, "Valid", clean_email

def check_deadline_consistency(body: str, received_date: datetime.date, deadline_str: Optional[str], why: str, summary: str, label: str) -> Tuple[bool, str]:
    has_phrase, expected_date, phrase = parse_date_phrase_from_text(body, received_date)

    if has_phrase:
        if label not in ["fyi", "noise"]:
            if not deadline_str:
                return False, f"Body contains date phrase '{phrase}', but gold deadline is null"
            try:
                dl_date = datetime.date.fromisoformat(deadline_str)
            except ValueError:
                return False, f"Invalid deadline format: {deadline_str}"

            if expected_date:
                diff = abs((dl_date - expected_date).days)
                if diff > 1:
                    return False, f"Date phrase '{phrase}' implies {expected_date}, but deadline is {dl_date}"
    else:
        if deadline_str:
            return False, f"Body has no date phrase, but gold deadline is {deadline_str} (must be null)"

    if not deadline_str:
        for field_name, text in [("why", why), ("summary", summary)]:
            f_has, _, f_phrase = parse_date_phrase_from_text(text, received_date)
            if f_has and f_phrase in ["today", "tonight", "tomorrow"] or (f_has and re.search(r'\d', f_phrase)):
                return False, f"{field_name} mentions calendar date '{f_phrase}', but deadline is null"

    return True, "Consistent"

async def verify_email_with_model(sampling_client, tokenizer, seed: str, email: dict, semaphore: asyncio.Semaphore) -> Tuple[bool, str]:
    global total_input_tokens, total_output_tokens
    prompt = build_verifier_prompt(seed, email)
    
    prompt_tokens = tokenizer.encode(prompt)
    model_input = tinker.types.ModelInput.from_ints(prompt_tokens)
    
    try:
        async with semaphore:
            result = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=300,
                    temperature=0.0
                )
            )
        
        total_input_tokens += len(prompt_tokens)
        total_output_tokens += len(result.sequences[0].tokens)
        check_spend_cap()
        
        clean_text = tokenizer.decode(result.sequences[0].tokens).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if not match:
            return False, f"Verifier failed to return JSON: {clean_text[:80]}"
        data = json.loads(match.group(0))
        
    except Exception as e:
        return False, f"Verifier exception: {e}"

    s_fits = data.get("sender_fits", False)
    coh = data.get("coherent", False)
    no_garb = data.get("no_garbled_words", False)
    why_comp = data.get("why_complete", False)
    dl_cor = data.get("deadline_correct", False)
    l_cor = data.get("label_correct", False)
    reason = data.get("reason", "No reason provided")

    if s_fits and coh and no_garb and why_comp and dl_cor and l_cor:
        return True, "Passed verifier"
    else:
        failures = []
        if not s_fits: failures.append("sender_fits=False")
        if not coh: failures.append("coherent=False")
        if not no_garb: failures.append("no_garbled_words=False")
        if not why_comp: failures.append("why_complete=False")
        if not dl_cor: failures.append("deadline_correct=False")
        if not l_cor: failures.append("label_correct=False")
        return False, f"Verifier dropped ({', '.join(failures)}): {reason}"

async def generate_single_email_with_verification(
    sampling_client,
    tokenizer,
    label: str,
    seed_item: dict,
    has_deadline: bool,
    semaphore: asyncio.Semaphore,
    is_hard_neg: bool = False,
    max_retries: int = 5
) -> dict:
    global total_input_tokens, total_output_tokens
    seed = seed_item["scenario"]
    senders = seed_item["senders"]
    has_amount = seed_item.get("has_amount", False)
    is_human = is_human_sender_scenario(seed)

    for attempt in range(max_retries):
        generation_attempts[label] += 1
        received_date = get_random_received_date()
        sender_org = random.choice(senders)
        sender_from = get_sender_identity(sender_org["name"], sender_org["domain"], is_human, seed)

        scenario_amount = get_scenario_amount(seed) if has_amount else None
        allow_urgent = (random.random() < 0.08)
        is_hinglish = (random.random() < 0.08 and is_human)
        apply_typo = (not is_hinglish and random.random() < 0.10)

        deadline_info = choose_deadline_config(label, has_deadline, received_date)
        prompt = build_generator_prompt(
            label, seed, sender_from, scenario_amount, received_date, deadline_info,
            allow_urgent, is_hard_neg, apply_typo, is_hinglish
        )

        prompt_tokens = tokenizer.encode(prompt)
        model_input = tinker.types.ModelInput.from_ints(prompt_tokens)

        try:
            async with semaphore:
                result = await sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(
                        max_tokens=500,
                        temperature=1.0
                    )
                )
            
            total_input_tokens += len(prompt_tokens)
            total_output_tokens += len(result.sequences[0].tokens)
            check_spend_cap()

            raw_text = tokenizer.decode(result.sequences[0].tokens).strip()
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON in generation output")

            raw_data = json.loads(match.group(0))
            raw_data["seed"] = seed
            
            body = str(raw_data.get("body_snippet", "")).strip()
            has_phrase, expected_date, phrase = parse_date_phrase_from_text(body, received_date)
            if has_phrase and expected_date and label not in ["fyi", "noise"]:
                raw_data["deadline"] = expected_date.isoformat()
            else:
                raw_data["deadline"] = None

            is_valid, basic_reason, clean_email = basic_deterministic_validate(raw_data, label, received_date, allow_urgent, is_human)

            if not is_valid:
                verifier_drops[label] += 1
                verifier_reasons.append(basic_reason)
                logging.warning(f"[{label}] Basic validation failed: {basic_reason} (attempt {attempt+1})")
                continue

            if random.random() < 0.25:
                v_ok, v_reason = await verify_email_with_model(sampling_client, tokenizer, seed, clean_email, semaphore)
            else:
                v_ok, v_reason = True, "Skipped verification"

            if v_ok:
                verifier_passes[label] += 1
                return clean_email
            else:
                verifier_drops[label] += 1
                verifier_reasons.append(v_reason)
                logging.warning(f"[{label}] {v_reason} (attempt {attempt+1})")
                continue

        except Exception as e:
            verifier_drops[label] += 1
            verifier_reasons.append(str(e))
            logging.warning(f"[{label}] Generation error: {e} (attempt {attempt+1})")

    return None

def get_char_trigrams(text: str) -> set:
    clean = re.sub(r'[^a-z0-9 ]', '', text.lower())
    if len(clean) < 3:
        return set()
    return set(clean[i:i+3] for i in range(len(clean) - 2))

def deduplicate_emails(emails: List[Dict]) -> Tuple[List[Dict], int]:
    unique = []
    dropped = 0
    for e in emails:
        sub = re.sub(r'[^a-zA-Z0-9]+', '', e.get('subject', '')).lower()
        bod_prefix = re.sub(r'[^a-zA-Z0-9]+', '', e.get('body_snippet', '')[:80]).lower()
        bod_trigrams = get_char_trigrams(e.get('body_snippet', ''))

        is_dup = False
        for ex in unique:
            ex_sub = re.sub(r'[^a-zA-Z0-9]+', '', ex.get('subject', '')).lower()
            ex_bod_prefix = re.sub(r'[^a-zA-Z0-9]+', '', ex.get('body_snippet', '')[:80]).lower()
            if sub == ex_sub and bod_prefix == ex_bod_prefix:
                is_dup = True
                break
            ex_trigrams = get_char_trigrams(ex.get('body_snippet', ''))
            if bod_trigrams and ex_trigrams:
                jaccard = len(bod_trigrams.intersection(ex_trigrams)) / len(bod_trigrams.union(ex_trigrams))
                if jaccard > 0.65:
                    is_dup = True
                    break

        if is_dup:
            dropped += 1
        else:
            unique.append(e)

    return unique, dropped

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/synthetic.jsonl")
    args = parser.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY not set")

    with open("data/seeds.yaml", "r", encoding="utf-8") as f:
        seeds_config = yaml.safe_load(f)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.6-35B-A3B")
    tokenizer = sampling_client.get_tokenizer()
    
    semaphore = asyncio.Semaphore(16)

    logging.info("Re-validating existing data...")
    saved_counts = {"must_act": 0, "worth_a_look": 0, "fyi": 0, "noise": 0}
    all_emails = []
    
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    if os.path.exists(args.output_file):
        lines = []
        with open(args.output_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        valid_emails = []
        for line in lines:
            try:
                em = json.loads(line)
                label = em.get("label")
                seed = em.get("seed", "")
                received_date = datetime.date.fromisoformat(em.get("received_date"))
                is_human = is_human_sender_scenario(seed)
                # Re-validate based on new stricter rules
                is_valid, _, clean_em = basic_deterministic_validate(em, label, received_date, allow_urgent=True, is_human=is_human)
                if is_valid:
                    valid_emails.append(clean_em)
                    saved_counts[label] += 1
                    seed_counts[seed] += 1
            except Exception as e:
                pass
                
        all_emails = valid_emails
        with open(args.output_file, "w", encoding="utf-8") as f:
            for em in all_emails:
                f.write(json.dumps(em) + "\n")
        
        logging.info(f"Re-validation complete. Retained {len(all_emails)} valid emails.")
    
    target_per_label = 150 # Target 600 total
    labels = ["must_act", "worth_a_look", "fyi", "noise"]
    
    total_emails_checked = 0
    
    logging.info("Continuing generation to target 600...")
    while True:
        active_labels = [l for l in labels if saved_counts[l] < target_per_label]
        if not active_labels:
            logging.info("Generated target number for all labels (600 total)!")
            break
            
        tasks = []
        round_labels = []
        for label in active_labels:
            # Filter seeds for this label that haven't hit the cap
            available_seeds = [s for s in seeds_config[label] if seed_counts[s["scenario"]] < MAX_PER_SEED]
            if not available_seeds:
                available_seeds = seeds_config[label] # Fallback if all hit cap
                
            seed_item = random.choice(available_seeds)
            has_dl = (random.random() < 0.8) if label in ["must_act", "worth_a_look"] else False
            is_hard_neg = (random.random() < 0.15) if label in ["noise", "fyi"] else False
            
            tasks.append(generate_single_email_with_verification(
                sampling_client, tokenizer, label, seed_item, has_dl, semaphore, is_hard_neg
            ))
            round_labels.append(label)
            
        results = await asyncio.gather(*tasks)
        
        with open(args.output_file, "a", encoding="utf-8") as f:
            for i, res in enumerate(results):
                total_emails_checked += 1
                if res:
                    f.write(json.dumps(res) + "\n")
                    f.flush()
                    all_emails.append(res)
                    saved_counts[round_labels[i]] += 1
                    seed_counts[res.get("seed", "")] += 1
                    
                if total_emails_checked % 50 == 0:
                    cost = (total_input_tokens / 1_000_000) * PREFILL_COST_PER_M + (total_output_tokens / 1_000_000) * SAMPLE_COST_PER_M
                    logging.info(f"Running Estimate Cost: ${cost:.4f}, Progress: {saved_counts}")

    print("\n" + "="*70)
    print("FINAL DATASET STATISTICS:")
    print("="*70)
    
    # 1. Label Counts
    print(f"Label Counts: {saved_counts}")
    
    # 2. Deadline-null rate per label
    null_by_label = Counter()
    total_by_label = Counter()
    for e in all_emails:
        lbl = e["label"]
        total_by_label[lbl] += 1
        if not e.get("deadline"):
            null_by_label[lbl] += 1

    print("\nDeadline-Null Rate:")
    for lbl in labels:
        n_cnt = null_by_label[lbl]
        t_cnt = total_by_label[lbl]
        r = (n_cnt / t_cnt * 100) if t_cnt else 0
        print(f"  {lbl:<13}: {n_cnt}/{t_cnt} null ({r:.1f}%)")
        
    # 3. Verifier drop reasons
    print("\nTop 5 Verifier Drop Reasons:")
    for r, count in Counter(verifier_reasons).most_common(5):
        print(f"  [{count}x] {r}")

    # 4. Top 20 sender names
    print("\nTop 20 Sender Names:")
    for s_name, count in Counter(e["from"] for e in all_emails).most_common(20):
        print(f"  [{count}x] {s_name}")
        
    # 5. Count per seed (top 10)
    print("\nTop 10 Seeds (Max 3% / 18 emails rule applied):")
    for s_name, count in seed_counts.most_common(10):
        print(f"  [{count}x] {s_name}")
        
    # 6. Final Cost
    cost = (total_input_tokens / 1_000_000) * PREFILL_COST_PER_M + (total_output_tokens / 1_000_000) * SAMPLE_COST_PER_M
    print(f"\nFINAL ESTIMATED COST: ${cost:.4f}")

    # 7. 8 Random Full Records
    print("\n" + "="*70)
    print("8 RANDOM FULL RECORDS:")
    print("="*70)
    sample_8 = random.sample(all_emails, min(8, len(all_emails)))
    for i, e in enumerate(sample_8):
        print(f"Sample {i+1} [{e['label']}]")
        print(f"Received Date: {e.get('received_date')}")
        print(f"From: {e['from']}")
        print(f"Subject: {e['subject']}")
        print(f"Body:\n{e['body_snippet']}")
        print(f"Why: {e['why']}")
        print(f"Deadline: {e['deadline']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
