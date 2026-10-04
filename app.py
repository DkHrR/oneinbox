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
    <title>OneInbox | Colorful Morning</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #fdfbf7;
            --sidebar: #0f172a; /* Deep Slate */
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-main);
            color: #1e293b;
            -webkit-font-smoothing: antialiased;
        }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }

        /* Colorful Gradient Cards */
        .card-urgent {
            background: linear-gradient(135deg, #fff1f2 0%, #ffedd5 100%);
            border: 1px solid #fecdd3;
            box-shadow: 0 4px 15px rgba(225, 29, 72, 0.05);
        }
        .card-urgent:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 12px 25px rgba(225, 29, 72, 0.15);
            border-color: #fda4af;
        }

        .card-look {
            background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
            border: 1px solid #bfdbfe;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.05);
        }
        .card-look:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 12px 25px rgba(59, 130, 246, 0.15);
            border-color: #93c5fd;
        }

        .card-fyi {
            background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
            border: 1px solid #a7f3d0;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.05);
        }
        .card-fyi:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 12px 25px rgba(16, 185, 129, 0.15);
            border-color: #6ee7b7;
        }

        .app-card {
            border-radius: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        /* Fun Buttons */
        .btn {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 10px 16px;
            border-radius: 12px;
            transition: all 0.2s;
            text-align: center;
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(4px);
        }
        
        .btn-urgent { color: #be123c; border: 1px solid #fda4af; }
        .btn-urgent:hover { background: #be123c; color: #fff; }
        
        .btn-look { color: #4338ca; border: 1px solid #a5b4fc; }
        .btn-look:hover { background: #4338ca; color: #fff; }

        .btn-fyi { color: #047857; border: 1px solid #6ee7b7; }
        .btn-fyi:hover { background: #047857; color: #fff; }

        /* Sidebar Items */
        .sidebar-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            border-radius: 12px;
            color: #94a3b8;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .sidebar-item:hover {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        
        /* Glass Header */
        .glass-header {
            background: rgba(253, 251, 247, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <!-- Premium Dark Sidebar -->
    <aside class="w-72 bg-[#0f172a] text-white flex flex-col h-full hidden md:flex shrink-0 shadow-2xl z-20 relative">
        <!-- Colorful accent line at top -->
        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-rose-400 via-blue-400 to-emerald-400"></div>
        
        <div class="p-8 pb-4">
            <h1 class="text-3xl font-bold tracking-tight mb-1 bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">OneInbox.</h1>
            <div class="text-xs text-slate-400 font-medium tracking-wide uppercase">Intelligence Digest</div>
        </div>
        
        <nav class="flex-1 px-4 space-y-2 overflow-y-auto mt-8">
            <div class="text-[10px] uppercase tracking-widest text-slate-500 font-bold px-4 mb-3">Today's Categories</div>
            
            <a href="#action-required" class="sidebar-item group">
                <div class="w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center group-hover:bg-rose-500 group-hover:text-white transition-colors">
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
                Action Required
                <span class="ml-auto text-[11px] bg-rose-500 text-white font-bold px-2.5 py-1 rounded-full shadow-[0_0_10px_rgba(244,63,94,0.5)]">{{ digest.must_act|length }}</span>
            </a>
            
            <a href="#worth-look" class="sidebar-item group">
                <div class="w-8 h-8 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center group-hover:bg-blue-500 group-hover:text-white transition-colors">
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"></path></svg>
                </div>
                Worth a Look
                <span class="ml-auto text-[11px] bg-slate-800 text-slate-300 font-bold px-2.5 py-1 rounded-full">{{ digest.worth_a_look|length }}</span>
            </a>
            
            <a href="#fyi" class="sidebar-item group">
                <div class="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center group-hover:bg-emerald-500 group-hover:text-white transition-colors">
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
                For Your Info
                <span class="ml-auto text-[11px] bg-slate-800 text-slate-300 font-bold px-2.5 py-1 rounded-full">{{ digest.fyi|length }}</span>
            </a>
        </nav>

        <div class="p-8 border-t border-slate-800">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                    <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                </div>
                <div>
                    <div class="text-sm font-semibold text-white">Backboard</div>
                    <div class="text-[10px] text-emerald-400 font-medium">Memory Sync Active</div>
                </div>
            </div>
        </div>
    </aside>

    <!-- Main Content Area -->
    <main class="flex-1 flex flex-col h-full overflow-hidden relative">
        
        <!-- Glass Header -->
        <header class="h-20 glass-header flex items-center justify-between px-8 shrink-0 z-10 sticky top-0">
            <div>
                <h2 class="text-xl font-bold text-slate-800">Good morning.</h2>
                <div class="text-xs font-medium text-slate-500 mt-0.5">Generated {{ digest.generated_at }}</div>
            </div>
            <div class="px-4 py-2 rounded-xl bg-white border border-slate-200 shadow-sm text-sm font-bold text-slate-700 flex items-center gap-2">
                <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                {{ digest.total_count }} Emails Analyzed
            </div>
        </header>

        <!-- Scrollable Grid Area -->
        <div class="flex-1 overflow-y-auto p-8 lg:p-12 scroll-smooth">
            
            {% if digest.mode == 'Demo (Sample Emails)' %}
            <div class="mb-10 bg-gradient-to-r from-amber-100 to-orange-100 border border-amber-200 rounded-2xl p-4 text-amber-800 text-sm font-bold flex items-center justify-center shadow-sm">
                🚧 Demo Mode: Viewing synthetic archived sample emails.
            </div>
            {% endif %}

            {% if digest.must_act %}
            <div id="action-required" class="mb-16">
                <div class="flex items-center gap-4 mb-8">
                    <h3 class="text-2xl font-bold text-rose-600">Action Required</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-rose-200 to-transparent"></div>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                    {% for item in digest.must_act %}
                    <div class="app-card card-urgent">
                        <div class="p-6 lg:p-8 flex-1 flex flex-col">
                            <div class="flex justify-between items-start mb-4">
                                <div class="text-[11px] font-bold text-rose-500 uppercase tracking-widest bg-white/50 px-3 py-1 rounded-full shadow-sm">{{ item.from }}</div>
                            </div>
                            <h4 class="text-xl font-bold text-slate-900 mb-3 leading-snug">{{ item.subject }}</h4>
                            
                            {% if item.deadline %}
                            <div class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500 text-white text-xs font-bold mb-4 shadow-sm w-max">
                                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                Due: {{ item.deadline }}
                            </div>
                            {% endif %}
                            
                            <p class="text-sm font-medium text-slate-700 leading-relaxed mb-5 flex-1">{{ item.summary }}</p>
                            
                            <div class="text-[11px] font-medium text-rose-800 bg-white/60 p-3 rounded-xl border border-white/50 mb-6 shadow-sm">
                                <strong class="text-rose-900 font-bold">Why:</strong> {{ item.why }}
                            </div>
                            
                            <div class="grid grid-cols-2 gap-3 mt-auto">
                                <a href="{{ item.ignore_link }}" class="btn btn-urgent col-span-2">Silence Sender</a>
                                <a href="{{ item.good_link }}" class="btn btn-urgent !bg-white">Accurate ✓</a>
                                <a href="{{ item.wrong_link }}" class="btn btn-urgent !bg-white">Wrong ✕</a>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if digest.worth_a_look %}
            <div id="worth-look" class="mb-16">
                <div class="flex items-center gap-4 mb-8">
                    <h3 class="text-2xl font-bold text-blue-600">Worth a Look</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-blue-200 to-transparent"></div>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                    {% for item in digest.worth_a_look %}
                    <div class="app-card card-look">
                        <div class="p-6 lg:p-8 flex-1 flex flex-col">
                            <div class="text-[11px] font-bold text-blue-500 uppercase tracking-widest bg-white/50 w-max px-3 py-1 rounded-full shadow-sm mb-4">{{ item.from }}</div>
                            <h4 class="text-xl font-bold text-slate-900 mb-3 leading-snug">{{ item.subject }}</h4>
                            <p class="text-sm font-medium text-slate-700 leading-relaxed mb-5 flex-1">{{ item.summary }}</p>
                            
                            <div class="grid grid-cols-2 gap-3 mt-auto pt-6 border-t border-blue-200/50">
                                <a href="{{ item.always_link }}" class="btn btn-look !bg-white">Always Show</a>
                                <a href="{{ item.ignore_link }}" class="btn btn-look">Ignore</a>
                                <a href="{{ item.good_link }}" class="btn btn-look col-span-2 !bg-blue-600 !text-white hover:!bg-blue-700">Perfect Classification ✓</a>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if digest.fyi %}
            <div id="fyi" class="mb-16">
                <div class="flex items-center gap-4 mb-8">
                    <h3 class="text-2xl font-bold text-emerald-600">For Your Information</h3>
                    <div class="h-px flex-1 bg-gradient-to-r from-emerald-200 to-transparent"></div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                    {% for item in digest.fyi %}
                    <div class="app-card card-fyi p-5 lg:p-6">
                        <div class="text-[10px] font-bold text-emerald-600 uppercase tracking-widest mb-3 truncate">{{ item.from }}</div>
                        <h4 class="text-base font-bold text-slate-900 mb-2 line-clamp-1">{{ item.subject }}</h4>
                        <p class="text-xs font-medium text-slate-600 leading-relaxed mb-5 flex-1 line-clamp-3">{{ item.summary }}</p>
                        
                        <div class="flex gap-2 mt-auto">
                            <a href="{{ item.always_link }}" class="btn btn-fyi flex-1 text-[10px] !py-2 !px-2">Always</a>
                            <a href="{{ item.ignore_link }}" class="btn btn-fyi flex-1 text-[10px] !py-2 !px-2 !bg-white hover:!bg-emerald-600 hover:!text-white">Ignore</a>
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
