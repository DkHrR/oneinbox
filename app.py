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
    <title>OneInbox Digest</title>
    <style>
        :root {
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: #e2e8f0;
            --must-act: #ef4444;
            --worth: #6366f1;
            --fyi: #0ea5e9;
            --noise: #94a3b8;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #090d16;
                --card-bg: #131b2e;
                --text: #f1f5f9;
                --muted: #94a3b8;
                --border: #1e293b;
            }
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 16px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 680px;
        }
        header {
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        h1 { margin: 0 0 6px 0; font-size: 1.5rem; }
        .meta { color: var(--muted); font-size: 0.85rem; }
        .counts { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        .badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 9999px;
            display: inline-block;
        }
        .badge-must-act { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-worth { background: rgba(99, 102, 241, 0.15); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.3); }
        .badge-fyi { background: rgba(14, 165, 233, 0.15); color: #0ea5e9; border: 1px solid rgba(14, 165, 233, 0.3); }
        .badge-noise { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .card.must-act-card { border-left: 4px solid var(--must-act); }
        .card.worth-card { border-left: 4px solid var(--worth); }
        .card.fyi-card { border-left: 4px solid var(--fyi); }
        .sender { font-weight: 600; font-size: 0.95rem; margin-bottom: 2px; }
        .subject { font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }
        .summary { font-size: 0.9rem; line-height: 1.4; margin-bottom: 8px; }
        .why { font-size: 0.8rem; color: var(--muted); margin-bottom: 10px; font-style: italic; }
        .deadline { color: #ef4444; font-weight: 600; font-size: 0.82rem; margin-bottom: 8px; }
        .actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }
        .btn {
            font-size: 0.75rem;
            text-decoration: none;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid var(--border);
            color: var(--text);
            background: var(--bg);
            transition: all 0.15s;
        }
        .btn:hover { opacity: 0.8; }
        .btn-rule { border-color: rgba(99, 102, 241, 0.4); color: var(--worth); }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Inbox Intelligence Digest</h1>
        {% if digest.mode == 'Demo (Sample Emails)' %}
        <div style="background: #fef3c7; color: #92400e; padding: 8px; border-radius: 6px; font-weight: 600; margin-bottom: 12px; font-size: 0.9rem;">
            Demo: archived sample emails dated Sept 2026
        </div>
        {% endif %}
        <div class="meta">Generated: {{ digest.generated_at }} | {{ digest.total_count }} emails analyzed</div>
        <div class="counts">
            <span class="badge badge-must-act">{{ digest.must_act|length }} Action Required</span>
            <span class="badge badge-worth">{{ digest.worth_a_look|length }} Worth a Look</span>
            <span class="badge badge-fyi">{{ digest.fyi|length }} FYI</span>
            <span class="badge badge-noise">{{ digest.noise_count }} Filtered Noise</span>
            {% if digest.pinned %}<span class="badge badge-fyi">{{ digest.pinned|length }} Pinned</span>{% endif %}
            {% if digest.ignored_count %}<span class="badge badge-noise">{{ digest.ignored_count }} Ignored</span>{% endif %}
        </div>
    </header>

    {% if digest.pinned %}
    <h2>📌 Pinned</h2>
    {% for item in digest.pinned %}
    <div class="card fyi-card">
        <div class="sender">{{ item.from }}</div>
        <div class="subject">{{ item.subject }}</div>
        <div class="summary">{{ item.summary }}</div>
    </div>
    {% endfor %}
    {% endif %}

    {% if digest.must_act %}
    <h2>🚨 Action Required</h2>
    {% for item in digest.must_act %}
    <div class="card must-act-card">
        <div class="sender">{{ item.from }}</div>
        <div class="subject">{{ item.subject }}</div>
        {% if item.deadline %}<div class="deadline">⏰ Deadline: {{ item.deadline }}</div>{% endif %}
        <div class="summary">{{ item.summary }}</div>
        <div class="why">{{ item.why }}</div>
        <div class="actions">
            <a class="btn btn-rule" href="{{ item.ignore_link }}">Ignore Sender</a>
            <a class="btn" href="{{ item.good_link }}">Accurate ✓</a>
            <a class="btn" href="{{ item.wrong_link }}">Wrong ✗</a>
        </div>
    </div>
    {% endfor %}
    {% endif %}

    {% if digest.worth_a_look %}
    <h2>💡 Worth a Look</h2>
    {% for item in digest.worth_a_look %}
    <div class="card worth-card">
        <div class="sender">{{ item.from }}</div>
        <div class="subject">{{ item.subject }}</div>
        <div class="summary">{{ item.summary }}</div>
        <div class="why">{{ item.why }}</div>
        <div class="actions">
            <a class="btn btn-rule" href="{{ item.always_link }}">Always Show</a>
            <a class="btn btn-rule" href="{{ item.ignore_link }}">Ignore</a>
            <a class="btn" href="{{ item.good_link }}">Accurate ✓</a>
            <a class="btn" href="{{ item.wrong_link }}">Wrong ✗</a>
        </div>
    </div>
    {% endfor %}
    {% endif %}

    {% if digest.fyi %}
    <h2>ℹ️ For Your Information</h2>
    {% for item in digest.fyi %}
    <div class="card fyi-card">
        <div class="sender">{{ item.from }}</div>
        <div class="subject">{{ item.subject }}</div>
        <div class="summary">{{ item.summary }}</div>
        <div class="actions">
            <a class="btn btn-rule" href="{{ item.always_link }}">Always Show</a>
            <a class="btn btn-rule" href="{{ item.ignore_link }}">Ignore</a>
            <a class="btn" href="{{ item.good_link }}">Accurate ✓</a>
        </div>
    </div>
    {% endfor %}
    {% endif %}

    <div style="margin-top: 30px; text-align: center; color: var(--muted); font-size: 0.8rem;">
        OneInbox • Protected with HMAC tokens • Backboard Memory Enabled
    </div>
</div>
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
        ingest_url = f"{base_url.rstrip('/')}/ingest"
        
        try:
            resp = requests.post(ingest_url, json=digest_data, headers={"Authorization": f"Bearer {ingest_token}"}, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Failed to POST /ingest. Status: {resp.status_code}. Response: {resp.text}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to POST /ingest: {e}")
            sys.exit(1)

    # 5. Digest Preview & Summary Counts
    preview_md = generate_digest_preview_md(classified_items, rules)
    with open(DIGEST_PREVIEW_FILE, "w", encoding="utf-8") as f:
        f.write(preview_md)
    logger.info(f"Saved digest preview to {DIGEST_PREVIEW_FILE}.")

    if args.dry_run:
        must_act = [x for x in classified_items if x["label"] == "must_act"]
        pinned = [x for x in classified_items if x["label"] == "pinned"]
        worth_a_look = [x for x in classified_items if x["label"] == "worth_a_look"]
        fyi = [x for x in classified_items if x["label"] == "fyi"]
        noise = [x for x in classified_items if x["label"] == "noise"]
        ignored = [x for x in classified_items if x["label"] == "ignored"]
        print(f"OneInbox: {len(must_act)} need action | {len(pinned)} pinned | {len(worth_a_look)} worth a look | {len(fyi)} FYI | {len(noise)} noise | {len(ignored)} ignored")

if __name__ == "__main__":
    main()
