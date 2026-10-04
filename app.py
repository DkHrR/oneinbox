import os
import sys
import json
import time
import hmac
import hashlib
import asyncio
import logging
import argparse
import email
import email.header
import imaplib
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote

from dotenv import load_dotenv
import requests

from backboard_client import (
    get_or_create_assistant,
    list_memories,
    add_memory,
    parse_rules,
    save_digest_to_backboard,
    get_digest_from_backboard
)
from redact import redact_text

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("oneinbox")

CHECKPOINT_FILE = "checkpoint.txt"
SAMPLE_DATA_FILE = "data/sample/emails.jsonl"
LOCAL_DIGEST_FILE = "latest_digest.json"
DIGEST_PREVIEW_FILE = "digest_preview.md"

# ── HMAC Signatures ────────────────────────────────────────────────────────────

def get_feedback_secret() -> str:
    return os.environ.get("FEEDBACK_SECRET", "default-insecure-secret-change-me")

def get_digest_token() -> str:
    return os.environ.get("DIGEST_TOKEN", "default-digest-token")

def make_feedback_sig(secret: str, params: dict) -> str:
    keys = sorted(k for k in params.keys() if k != "sig")
    query = "&".join(f"{k}={params[k]}" for k in keys)
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

def verify_feedback_sig(secret: str, params: dict, sig: str) -> bool:
    if not sig:
        return False
    expected = make_feedback_sig(secret, params)
    return hmac.compare_digest(expected, sig)

def build_feedback_url(base_url: str, secret: str, action: str, **kwargs) -> str:
    params = {"action": action, **kwargs}
    sig = make_feedback_sig(secret, params)
    params["sig"] = sig
    return f"{base_url.rstrip('/')}/feedback?{urlencode(params)}"

# ── Email Loading (Demo & IMAP) ────────────────────────────────────────────────

def load_demo_emails(path: str = SAMPLE_DATA_FILE) -> list[dict]:
    emails = []
    if not os.path.exists(path):
        logger.error(f"Sample data file {path} not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["id"] = idx
            emails.append(item)
    return emails

def decode_header_value(val) -> str:
    parts = email.header.decode_header(val or "")
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result).strip()

def extract_body_from_msg(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")
    return body.strip()

def fetch_imap_emails(limit: int = 50) -> list[dict]:
    host = os.environ.get("IMAP_HOST")
    user = os.environ.get("IMAP_USER")
    password = os.environ.get("IMAP_PASS")

    if not host or not user or not password:
        logger.error("IMAP credentials not configured in environment (IMAP_HOST, IMAP_USER, IMAP_PASS).")
        return []

    emails = []
    try:
        mail = imaplib.IMAP4_SSL(host)
        mail.login(user, password)
        # Read-only selection
        mail.select("INBOX", readonly=True)

        since_date = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
        status, data = mail.search(None, f'(SINCE "{since_date}")')
        if status != "OK":
            logger.warning("IMAP search returned non-OK status.")
            return []

        msg_ids = data[0].split()
        if not msg_ids:
            logger.info("No emails found in the last 24h via IMAP.")
            return []

        # Take the most recent up to limit
        selected_ids = msg_ids[-limit:]
        for idx, mid in enumerate(reversed(selected_ids)):
            res, mdata = mail.fetch(mid, "(RFC822)")
            if res != "OK":
                continue
            raw_msg = email.message_from_bytes(mdata[0][1])
            subject = decode_header_value(raw_msg.get("Subject", "(No Subject)"))
            sender = decode_header_value(raw_msg.get("From", "Unknown"))
            date_str = decode_header_value(raw_msg.get("Date", ""))
            
            raw_body = extract_body_from_msg(raw_msg)
            # Redact PII BEFORE model sees it
            redacted_body = redact_text(raw_body[:500])
            user_msg = f"Date: {date_str}\nFrom: {sender}\nSubject: {subject}\n\n{redacted_body}"

            emails.append({
                "id": idx,
                "received_date": date_str,
                "from": sender,
                "subject": subject,
                "body": redacted_body,
                "user_message": user_msg
            })
        mail.logout()
    except Exception as e:
        logger.error(f"IMAP fetch encountered error: {e}")
    return emails

# ── Classification Pipeline ────────────────────────────────────────────────────

def get_checkpoint_path() -> str:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "tinker://f29da6f5-1860-582f-a71f-2fd87bd36cd8:train:0/sampler_weights/final"

async def process_emails_async(emails: list[dict], rules: tuple[set[str], set[str]]) -> list[dict]:
    import tinker
    from classify import classify_email
    checkpoint = get_checkpoint_path()
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    tokenizer = sampling_client.get_tokenizer()

    always_show, ignore = rules

    semaphore = asyncio.Semaphore(10)

    async def classify_one(item):
        async with semaphore:
            parsed, latency, valid, raw_resp, p_tok, c_tok = await classify_email(
                sampling_client, tokenizer, item["user_message"], base_model=False
            )
            
            label = parsed.get("label", "noise")
            why = parsed.get("why", "")
            deadline = parsed.get("deadline", None)
            summary = parsed.get("summary", "")

            # Apply deterministic Backboard memory rules
            sender_lower = item.get("from", "").lower()
            
            # Check ignore rules first
            for ig in ignore:
                if ig in sender_lower:
                    label = "ignored"
                    why = f"Rule applied: ignore sender={ig}"
                    break
            
            # Check always_show rules
            for al in always_show:
                if al in sender_lower:
                    label = "pinned"
                    why = f"Rule applied: always_show sender={al}"
                    break

            return {
                "id": item["id"],
                "from": item["from"],
                "subject": item["subject"],
                "received_date": item.get("received_date", ""),
                "deadline": deadline,
                "label": label,
                "why": why,
                "summary": summary
            }

    tasks = [classify_one(em) for em in emails]
    return await asyncio.gather(*tasks)

def run_classification_pipeline(emails: list[dict], rules: tuple[set[str], set[str]]) -> list[dict]:
    return asyncio.run(process_emails_async(emails, rules))

# ── Preview & Persistence ──────────────────────────────────────────────────────

def generate_digest_preview_md(classified_items: list[dict], rules: tuple[set[str], set[str]]) -> str:
    always_show, ignore = rules
    must_act = [x for x in classified_items if x["label"] == "must_act"]
    pinned = [x for x in classified_items if x["label"] == "pinned"]
    worth_a_look = [x for x in classified_items if x["label"] == "worth_a_look"]
    fyi = [x for x in classified_items if x["label"] == "fyi"]
    noise = [x for x in classified_items if x["label"] == "noise"]
    ignored = [x for x in classified_items if x["label"] == "ignored"]

    md = []
    md.append("# OneInbox Digest Preview")
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Total Processed**: {len(classified_items)} emails\n")

    md.append("## Active Preference Rules")
    md.append(f"- **Always Show**: {', '.join(sorted(always_show)) if always_show else '(none)'}")
    md.append(f"- **Ignore**: {', '.join(sorted(ignore)) if ignore else '(none)'}\n")

    md.append("## Categorized Digest")

    md.append(f"### 🚨 Must Act ({len(must_act)})")
    if not must_act:
        md.append("_None_")
    for item in must_act:
        deadline_badge = f" [Deadline: {item['deadline']}]" if item.get('deadline') else ""
        md.append(f"- **{item['from']}**: {item['subject']}{deadline_badge}")
        md.append(f"  - *Summary*: {item.get('summary', '')}")
        md.append(f"  - *Reason*: {item.get('why', '')}")

    md.append(f"\n### 💡 Worth a Look ({len(worth_a_look)})")
    if not worth_a_look:
        md.append("_None_")
    for item in worth_a_look:
        md.append(f"- **{item['from']}**: {item['subject']}")
        md.append(f"  - *Summary*: {item.get('summary', '')}")
        md.append(f"  - *Reason*: {item.get('why', '')}")

    md.append(f"\n### ℹ️ FYI ({len(fyi)})")
    if not fyi:
        md.append("_None_")
    for item in fyi:
        md.append(f"- **{item['from']}**: {item['subject']}")
        md.append(f"  - *Summary*: {item.get('summary', '')}")

    md.append(f"\n### 📌 Pinned ({len(pinned)})")
    if not pinned:
        md.append("_None_")
    for item in pinned:
        md.append(f"- **{item['from']}**: {item['subject']}")
        md.append(f"  - *Summary*: {item.get('summary', '')}")

    md.append(f"\n### 🔇 Noise ({len(noise)})")
    md.append(f"- {len(noise)} promotional/marketing emails filtered.")
    md.append(f"- {len(ignored)} ignored based on rules.")

    return "\n".join(md)

# ── Web Application (Flask) ────────────────────────────────────────────────────

def create_app():
    from flask import Flask, request, render_template_string, abort, redirect, jsonify
    
    global_digest_store = {}

    app = Flask(__name__)
    secret = get_feedback_secret()
    token = get_digest_token()
    base_url = os.environ.get("APP_BASE_URL", "http://localhost:5000")
    backboard_key = os.environ.get("BACKBOARD_API_KEY", "")
    assistant_id = os.environ.get("BACKBOARD_ASSISTANT_ID", "")

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OneInbox Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #000000;
            color: #ffffff;
            -webkit-font-smoothing: antialiased;
        }

        /* Minimal scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }

        .glass-panel {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }

        .dashboard-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .dashboard-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .btn-action {
            font-size: 0.75rem;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.8);
            transition: all 0.2s;
            text-decoration: none;
            text-align: center;
        }
        .btn-action:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            border-radius: 8px;
            color: rgba(255,255,255,0.6);
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
            text-decoration: none;
        }
        .nav-item:hover, .nav-item.active {
            background: rgba(255,255,255,0.05);
            color: #fff;
        }
    </style>
</head>
<body class="flex h-screen overflow-hidden selection:bg-white/20">

    <!-- Left Sidebar -->
    <aside class="w-64 border-r border-white/10 glass-panel flex flex-col h-full hidden md:flex shrink-0">
        <div class="p-6">
            <h1 class="text-xl font-semibold tracking-tight text-white mb-1">OneInbox<span class="text-blue-500">.</span></h1>
            <div class="text-xs text-gray-500">Neural Digest</div>
        </div>
        
        <nav class="flex-1 px-4 space-y-2 overflow-y-auto">
            <div class="text-[10px] uppercase tracking-widest text-gray-500 font-semibold px-4 mb-2 mt-4">Overview</div>
            <a href="#" class="nav-item active">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                Dashboard
            </a>
            <div class="text-[10px] uppercase tracking-widest text-gray-500 font-semibold px-4 mb-2 mt-8">Categories</div>
            <a href="#action-required" class="nav-item">
                <span class="dot bg-red-500 shadow-[0_0_8px_#ef4444]"></span> Action Required
                <span class="ml-auto bg-white/10 text-white/60 px-2 py-0.5 rounded-md text-xs">{{ digest.must_act|length }}</span>
            </a>
            <a href="#worth-look" class="nav-item">
                <span class="dot bg-blue-500"></span> Worth a Look
                <span class="ml-auto bg-white/10 text-white/60 px-2 py-0.5 rounded-md text-xs">{{ digest.worth_a_look|length }}</span>
            </a>
            <a href="#fyi" class="nav-item">
                <span class="dot bg-emerald-500"></span> FYI
                <span class="ml-auto bg-white/10 text-white/60 px-2 py-0.5 rounded-md text-xs">{{ digest.fyi|length }}</span>
            </a>
        </nav>

        <div class="p-6 border-t border-white/5">
            <div class="text-xs text-gray-500">Status</div>
            <div class="flex items-center gap-2 text-sm font-medium text-emerald-400 mt-1">
                <span class="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]"></span> Backboard Active
            </div>
        </div>
    </aside>

    <!-- Main Content Area -->
    <main class="flex-1 flex flex-col h-full overflow-hidden bg-black relative">
        
        <!-- Ambient Glow for Dashboard -->
        <div class="absolute top-0 right-0 w-[800px] h-[500px] bg-blue-500/10 blur-[120px] rounded-full pointer-events-none -z-10 transform translate-x-1/2 -translate-y-1/2"></div>
        <div class="absolute bottom-0 left-0 w-[600px] h-[400px] bg-red-500/5 blur-[100px] rounded-full pointer-events-none -z-10 transform -translate-x-1/2 translate-y-1/2"></div>

        <!-- Top Header -->
        <header class="h-20 border-b border-white/10 glass-panel flex items-center justify-between px-8 shrink-0 z-10">
            <div>
                <div class="text-xs font-medium text-gray-400">Generated: {{ digest.generated_at }}</div>
            </div>
            <div class="flex items-center gap-4">
                <div class="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm font-medium text-gray-300">
                    {{ digest.total_count }} Emails Analyzed
                </div>
            </div>
        </header>

        <!-- Scrollable Grid Area -->
        <div class="flex-1 overflow-y-auto p-8 scroll-smooth z-10">
            
            {% if digest.mode == 'Demo (Sample Emails)' %}
            <div class="mb-8 border border-orange-500/30 bg-orange-500/10 rounded-xl p-4 text-orange-400 text-sm font-medium">
                Demo Mode Active: Displaying synthetic archived sample emails.
            </div>
            {% endif %}

            <!-- Welcome Banner -->
            <div class="mb-12 p-8 rounded-2xl border border-white/10 bg-gradient-to-r from-white/5 to-transparent">
                <h2 class="text-3xl font-semibold mb-2">Good morning.</h2>
                <p class="text-gray-400 max-w-2xl">Your inbox has been processed. You have <strong class="text-white">{{ digest.must_act|length }} tasks</strong> requiring your attention and <strong class="text-white">{{ digest.worth_a_look|length }} updates</strong> worth reading today.</p>
            </div>

            {% if digest.must_act %}
            <div id="action-required" class="mb-12">
                <div class="flex items-center gap-3 mb-6">
                    <span class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_12px_#ef4444]"></span>
                    <h3 class="text-xl font-semibold text-white">Action Required</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-red-500/20 to-transparent ml-4"></div>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                    {% for item in digest.must_act %}
                    <div class="dashboard-card relative overflow-hidden group">
                        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500/80 to-red-500/20"></div>
                        <div class="p-6 flex-1 flex flex-col">
                            <div class="text-xs font-semibold text-red-400 uppercase tracking-wider mb-2 flex justify-between items-start">
                                <span>{{ item.from }}</span>
                                {% if item.deadline %}
                                <span class="bg-red-500/20 text-red-400 px-2 py-1 rounded border border-red-500/30">{{ item.deadline }}</span>
                                {% endif %}
                            </div>
                            <h4 class="text-lg font-semibold text-white mb-3">{{ item.subject }}</h4>
                            <p class="text-sm text-gray-400 leading-relaxed mb-4 flex-1">{{ item.summary }}</p>
                            <div class="text-xs text-gray-500 bg-black/50 p-3 rounded-lg border border-white/5 mb-6">
                                <span class="text-gray-400 font-medium">AI Reason:</span> {{ item.why }}
                            </div>
                            
                            <div class="grid grid-cols-3 gap-2 mt-auto">
                                <a href="{{ item.ignore_link }}" class="btn-action col-span-3 mb-2 hover:!bg-white/10">Ignore Sender</a>
                                <a href="{{ item.good_link }}" class="btn-action col-span-1.5 hover:!bg-emerald-500/20 hover:!text-emerald-400 !border !border-transparent hover:!border-emerald-500/30">Accurate</a>
                                <a href="{{ item.wrong_link }}" class="btn-action col-span-1.5 hover:!bg-red-500/20 hover:!text-red-400 !border !border-transparent hover:!border-red-500/30">Wrong</a>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if digest.worth_a_look %}
            <div id="worth-look" class="mb-12">
                <div class="flex items-center gap-3 mb-6">
                    <span class="w-3 h-3 rounded-full bg-blue-500 shadow-[0_0_12px_#3b82f6]"></span>
                    <h3 class="text-xl font-semibold text-white">Worth a Look</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-blue-500/20 to-transparent ml-4"></div>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                    {% for item in digest.worth_a_look %}
                    <div class="dashboard-card">
                        <div class="p-6 flex-1 flex flex-col">
                            <div class="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-2">{{ item.from }}</div>
                            <h4 class="text-lg font-semibold text-white mb-3">{{ item.subject }}</h4>
                            <p class="text-sm text-gray-400 leading-relaxed mb-4 flex-1">{{ item.summary }}</p>
                            
                            <div class="grid grid-cols-2 gap-2 mt-auto">
                                <a href="{{ item.always_link }}" class="btn-action hover:!bg-blue-500/20 hover:!text-blue-400">Always Show</a>
                                <a href="{{ item.ignore_link }}" class="btn-action hover:!bg-red-500/20 hover:!text-red-400">Ignore</a>
                                <a href="{{ item.good_link }}" class="btn-action col-span-2 mt-2 hover:!bg-emerald-500/20 hover:!text-emerald-400">Accurate Match</a>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if digest.fyi %}
            <div id="fyi" class="mb-12">
                <div class="flex items-center gap-3 mb-6">
                    <span class="w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_12px_#10b981]"></span>
                    <h3 class="text-xl font-semibold text-white">For Your Information</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-emerald-500/20 to-transparent ml-4"></div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {% for item in digest.fyi %}
                    <div class="dashboard-card p-5">
                        <div class="text-[10px] font-semibold text-emerald-500 uppercase tracking-wider mb-1 truncate">{{ item.from }}</div>
                        <h4 class="text-base font-semibold text-white mb-2 line-clamp-1">{{ item.subject }}</h4>
                        <p class="text-xs text-gray-400 leading-relaxed mb-4 line-clamp-3 flex-1">{{ item.summary }}</p>
                        
                        <div class="flex gap-2 mt-auto border-t border-white/5 pt-3">
                            <a href="{{ item.always_link }}" class="text-[10px] font-medium text-gray-400 hover:text-white flex-1 text-center bg-white/5 py-1.5 rounded">Always</a>
                            <a href="{{ item.ignore_link }}" class="text-[10px] font-medium text-gray-400 hover:text-red-400 flex-1 text-center bg-white/5 py-1.5 rounded">Ignore</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

        </div>
    </main>
</body>
</html>"""

    CONFIRMATION_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preference Recorded</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            padding: 16px;
        }
        .box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 28px;
            max-width: 440px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .icon { font-size: 2.5rem; margin-bottom: 12px; }
        h2 { margin: 0 0 10px 0; font-size: 1.3rem; }
        p { color: #64748b; font-size: 0.95rem; line-height: 1.5; margin-bottom: 20px; }
        a {
            display: inline-block;
            background: #4f46e5;
            color: #ffffff;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
        }
    </style>
</head>
<body>
<div class="box">
    <div class="icon">✨</div>
    <h2>Preference Saved</h2>
    <p>{{ message }}</p>
    <a href="{{ digest_url }}">Return to Digest</a>
</div>
</body>
</html>"""

    @app.route("/")
    def index():
        return redirect(f"/digest?token={token}")

    @app.route("/ingest", methods=["POST"])
    def ingest():
        ingest_token = os.environ.get("INGEST_TOKEN")
        auth_header = request.headers.get("Authorization", "")
        if f"Bearer {ingest_token}" != auth_header:
            abort(401, "Invalid INGEST_TOKEN")
        data = request.json
        if not data:
            abort(400, "Missing JSON payload")
            
        global_digest_store["latest"] = data
        try:
            with open(LOCAL_DIGEST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
            
        if backboard_key and assistant_id:
            try:
                save_digest_to_backboard(backboard_key, assistant_id, data)
            except Exception as e:
                logger.error(f"Failed to persist digest snapshot to Backboard: {e}")
                abort(500, f"Failed to persist digest snapshot to Backboard: {e}")
            
        return jsonify({"status": "OK"}), 200

    @app.route("/digest")
    def digest():
        req_token = request.args.get("token")
        if not req_token or req_token != token:
            abort(403, "Invalid or missing DIGEST_TOKEN")

        digest_data = global_digest_store.get("latest")
        if not digest_data and os.path.exists(LOCAL_DIGEST_FILE):
            try:
                with open(LOCAL_DIGEST_FILE, "r", encoding="utf-8") as f:
                    digest_data = json.load(f)
            except Exception:
                pass
        
        if not digest_data and backboard_key and assistant_id:
            digest_data = get_digest_from_backboard(backboard_key, assistant_id)

        if not digest_data:
            return "No digest yet", 200

        # Enhance items with signed feedback links
        items = digest_data.get("items", [])
        must_act = []
        worth_a_look = []
        fyi = []
        noise_count = 0

        pinned = []
        ignored_count = 0
        from datetime import datetime
        is_demo = digest_data.get("mode") == "Demo (Sample Emails)"
        
        for it in items:
            lbl = it.get("label", "noise")
            sender = it.get("from", "")
            item_id = it.get("id", 0)

            # Fix deadlines for demo mode
            dl = it.get("deadline")
            if dl and is_demo:
                try:
                    dt = datetime.strptime(dl, "%Y-%m-%d")
                    if dt < datetime.now():
                        it["deadline"] = f"{dl} (sample date)"
                except:
                    pass

            # Generate signed action URLs
            it["always_link"] = build_feedback_url(base_url, secret, "always_show", sender=sender)
            it["ignore_link"] = build_feedback_url(base_url, secret, "ignore", sender=sender)
            it["good_link"] = build_feedback_url(base_url, secret, "good", sender=sender, label=lbl)
            it["wrong_link"] = build_feedback_url(base_url, secret, "wrong", sender=sender, label=lbl)

            if lbl == "must_act":
                must_act.append(it)
            elif lbl == "worth_a_look":
                worth_a_look.append(it)
            elif lbl == "fyi":
                fyi.append(it)
            elif lbl == "pinned":
                pinned.append(it)
            elif lbl == "ignored":
                ignored_count += 1
            else:
                noise_count += 1

        render_data = {
            "generated_at": digest_data.get("generated_at", ""),
            "total_count": len(items),
            "must_act": must_act,
            "worth_a_look": worth_a_look,
            "fyi": fyi,
            "noise_count": noise_count,
            "pinned": pinned,
            "ignored_count": ignored_count,
            "mode": digest_data.get("mode")
        }

        return render_template_string(HTML_TEMPLATE, digest=render_data)

    @app.route("/feedback")
    def feedback():
        params = dict(request.args)
        sig = params.pop("sig", None)
        action = params.get("action")

        if not sig or not verify_feedback_sig(secret, params, sig):
            abort(403, "Invalid HMAC signature for feedback action")

        msg = "Feedback received."
        if action == "always_show":
            sender = params.get("sender", "").strip()
            rule_text = f"RULE: always_show sender={sender.lower()}"
            if backboard_key and assistant_id:
                add_memory(backboard_key, assistant_id, rule_text)
            msg = f"Rule saved: emails from '{sender}' will now always be prioritized into Action Required on the next digest."

        elif action == "ignore":
            sender = params.get("sender", "").strip()
            rule_text = f"RULE: ignore sender={sender.lower()}"
            if backboard_key and assistant_id:
                add_memory(backboard_key, assistant_id, rule_text)
            msg = f"Rule saved: emails from '{sender}' will now be treated as noise on the next digest."

        elif action in ("good", "wrong"):
            sender = params.get("sender", "unknown")
            label = params.get("label", "unknown")
            feedback_text = f"FEEDBACK: {action} sender={sender} label={label}"
            if backboard_key and assistant_id:
                add_memory(backboard_key, assistant_id, feedback_text)
            msg = f"Feedback recorded ({action})."

        digest_url = f"/digest?token={token}"
        return render_template_string(CONFIRMATION_TEMPLATE, message=msg, digest_url=digest_url)

    return app

def post_digest_to_ingest(digest_data: dict, base_url: str, ingest_token: str, max_retries: int = None) -> bool:
    if max_retries is None:
        max_retries = int(os.environ.get("INGEST_RETRIES", "5"))
    ingest_url = f"{base_url.rstrip('/')}/ingest"

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Posting digest to {ingest_url} (attempt {attempt}/{max_retries})...")
            resp = requests.post(
                ingest_url,
                json=digest_data,
                headers={"Authorization": f"Bearer {ingest_token}"},
                timeout=60
            )
            if resp.status_code == 200:
                logger.info("Successfully posted digest to /ingest.")
                return True
            else:
                logger.error(f"POST /ingest attempt {attempt} returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"POST /ingest attempt {attempt} failed: {e}")

        if attempt < max_retries:
            logger.info("Waiting 20 seconds before retry...")
            time.sleep(20)

    logger.error(f"All {max_retries} attempts to POST /ingest failed.")
    return False

# ── Main Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OneInbox Daily Email Intelligence Digest")
    parser.add_argument("--dry-run", action="store_true", help="Print digest summary counts and save digest_preview.md.")
    parser.add_argument("--imap", action="store_true", help="Fetch emails from real inbox over IMAP (read-only, last 24h).")
    parser.add_argument("--server", action="store_true", help="Start the web digest server.")
    parser.add_argument("--port", type=int, default=5000, help="Web server port.")
    args = parser.parse_args()

    if args.server:
        app = create_app()
        logger.info(f"Starting OneInbox web server on port {args.port}...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    # Check Tinker API key
    if not os.environ.get("TINKER_API_KEY"):
        logger.error("TINKER_API_KEY missing from environment.")
        sys.exit(1)

    # 1. Backboard setup & Memory Rules
    bb_key = os.environ.get("BACKBOARD_API_KEY", "")
    asst_id = os.environ.get("BACKBOARD_ASSISTANT_ID", "")
    rules = (set(), set())
    if bb_key:
        asst_id = get_or_create_assistant(bb_key, asst_id)
        mems = list_memories(bb_key, asst_id)
        rules = parse_rules(mems)
        always_show, ignore = rules
        logger.info(f"Loaded {len(mems)} memories from Backboard (Active rules: {len(always_show)} always_show, {len(ignore)} ignore).")

    # 2. Email Fetching
    if args.imap:
        logger.info("Running in IMAP mode (read-only, last 24 hours)...")
        emails = fetch_imap_emails(limit=50)
        mode_str = "IMAP (Live Inbox, last 24h)"
    else:
        logger.info("Running in DEMO mode (loading sample from data/sample/emails.jsonl)...")
        emails = load_demo_emails()
        mode_str = "Demo (Sample Emails)"

    if not emails:
        logger.warning("No emails to process.")
        return

    # 3. Model Classification with fine-tuned checkpoint
    logger.info(f"Classifying {len(emails)} emails with fine-tuned model...")
    classified_items = run_classification_pipeline(emails, rules)

    # 4. Post Latest Digest to Web Service
    digest_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode_str,
        "total_count": len(classified_items),
        "items": classified_items
    }
    if not args.dry_run:
        base_url = os.environ.get("APP_BASE_URL", "http://localhost:5000")
        ingest_token = os.environ.get("INGEST_TOKEN", "")
        if not post_digest_to_ingest(digest_data, base_url, ingest_token):
            sys.exit(1)

    # 5. Digest Preview & Summary Counts
    preview_md = generate_digest_preview_md(classified_items, rules)
    with open(DIGEST_PREVIEW_FILE, "w", encoding="utf-8") as f:
        f.write(preview_md)
    logger.info(f"Saved digest preview to {DIGEST_PREVIEW_FILE}.")

    must_act = [x for x in classified_items if x["label"] == "must_act"]
    pinned = [x for x in classified_items if x["label"] == "pinned"]
    worth_a_look = [x for x in classified_items if x["label"] == "worth_a_look"]
    fyi = [x for x in classified_items if x["label"] == "fyi"]
    noise = [x for x in classified_items if x["label"] == "noise"]
    ignored = [x for x in classified_items if x["label"] == "ignored"]
    print(f"OneInbox: {len(must_act)} need action | {len(pinned)} pinned | {len(worth_a_look)} worth a look | {len(fyi)} FYI | {len(noise)} noise | {len(ignored)} ignored")

if __name__ == "__main__":
    main()
