# OneInbox

OneInbox is an intelligent, privacy-first daily email digest assistant powered by a fine-tuned Qwen3.5-4B model and persistent memory via Backboard.

It categorizes incoming emails into four actionable tiers:
- 🚨 **must_act**: Urgent action required (bills, critical deadlines, confirmations, security notices)
- 💡 **worth_a_look**: High-value opportunities, meetups, beta releases, or personal notes
- ℹ️ **fyi**: Non-actionable informational updates (receipts, change password notices, closed tickets)
- 🔇 **noise**: Marketing blasts, promotions, newsletters, and social clutter

---

## Architecture & How It Works

1. **Email Ingestion**:
   - **Demo Mode** (`--dry-run` or default): Reads sample emails from `data/sample/emails.jsonl` (labels stripped).
   - **IMAP Mode** (`--imap`): Connects to your email provider over SSL in **read-only** mode for the last 24 hours.
2. **PII Redaction**:
   - Strips phone numbers, OTPs, account numbers, and URLs from email snippets prior to model inference (`redact.py`).
3. **Model Inference**:
   - Runs native sampling on Tinker using our fine-tuned Qwen3.5-4B model (`checkpoint.txt`).
4. **Preference Rules via Backboard Memory**:
   - User preferences (`RULE: always_show sender=...` and `RULE: ignore sender=...`) are retrieved directly from Backboard memories.
   - Rules are parsed and applied deterministically in code—never relying on probabilistic semantic search for classification decisions.
5. **Mobile Digest Web Service & HMAC Feedback**:
   - A lightweight Flask service serving `/digest?token=...` protected by `DIGEST_TOKEN`.
   - Each item includes one-click feedback buttons (`Always Show Sender`, `Ignore Sender`, `Accurate ✓`, `Wrong ✗`).
   - Links are signed with an HMAC-SHA256 signature using `FEEDBACK_SECRET`. When tapped, the web service verifies the signature, appends the preference rule to Backboard memory, and confirms the update. The next digest automatically honors the updated rule.

---

## Environment Variables

Configure these in `.env` (or in the Render dashboard for cloud deployment):

| Variable | Description |
| :--- | :--- |
| `TINKER_API_KEY` | API key for Tinker training and inference. |
| `BACKBOARD_API_KEY` | API key for Backboard assistant memory storage. |
| `BACKBOARD_ASSISTANT_ID` | Assistant ID for OneInbox on Backboard. |
| `DIGEST_TOKEN` | Secret URL token required to access `/digest?token=...`. |
| `FEEDBACK_SECRET` | HMAC secret key used to sign and verify feedback links. |
| `APP_BASE_URL` | Base URL of the web service (e.g. `http://localhost:5000` or `https://oneinbox.onrender.com`). |
| `IMAP_HOST` | *(Optional for IMAP mode)* e.g. `imap.gmail.com`. |
| `IMAP_USER` | *(Optional for IMAP mode)* Your email address. |
| `IMAP_PASS` | *(Optional for IMAP mode)* Your app password. |

---

## Setup & Running Locally

### 1. Installation
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Daily Digest in Demo Mode
```bash
# Preview digest summary counts
python app.py --dry-run
```

### 3. Run Web Service Locally
```bash
python app.py --server --port 5000
```
Open `http://localhost:5000/digest?token=<YOUR_DIGEST_TOKEN>` on your browser or mobile phone.

### 4. Run IMAP Mode (Read-Only)
```bash
python app.py --imap --dry-run
```

---

## Deployment on Render (`render.yaml`)

`render.yaml` defines:
1. **Web Service** (`oneinbox-web`): Runs Gunicorn serving the mobile digest and HMAC feedback endpoints.
2. **Cron Job** (`oneinbox-daily-digest`): Runs `python app.py` daily at 01:30 UTC (07:00 IST).

### Storage Design: Shared State across Render Containers
- On Render's free tier, Web Services and Cron Jobs run in isolated ephemeral containers and cannot share a local filesystem.
- Furthermore, Render free web services spin down after 15 minutes of inactivity.
- **Why Backboard?**: Backboard acts as the central, persistent memory and state layer. When the cron job finishes, it persists rules and digest metadata directly to Backboard. When the web service wakes up, it fetches the state from Backboard. This utilizes Render's $50 free credit for compute, while Backboard's free tier covers our memory needs, avoiding paid Render persistent disks.

---

## Privacy Notes

- **No Third-Party Notification Service**: No third-party notification service is used. The digest is read exclusively on a token-protected page (`/digest?token=...`). A push notification could be added later.
- **Never Logged**: Raw email bodies, full headers, and API keys are never printed to stdout/stderr or written to version control.
- **Strict Read-Only IMAP**: IMAP connections use `mail.select("INBOX", readonly=True)`, preventing any modification, flagging, or deletion of user emails.
- **Pre-Model Redaction**: PII (phone numbers, OTP codes, card/account numbers, and external URLs) is stripped before prompt assembly.
- **Secrets Protected**: `.env`, `latest_digest.json`, and `digest_preview.md` are added to `.gitignore`.

---

## Honest Limitations

- **Synthetic Validation Dataset**: The training set and validation benchmark (`data/val.jsonl`) consist of synthetic emails designed to capture distribution shifts, deadlines, and urgency phrasing. The 96.67% accuracy was measured against this synthetic distribution.
- **IMAP Pipeline Untested on Live Inboxes**: While the IMAP pipeline implements standard RFC822 parsing, PII redaction, and read-only fetching, it has not yet been benchmarked on a live production inbox with complex HTML multipart layouts.
- **Fine-Tuned Pricing Model**: The Tinker documentation rate card (`https://tinker-docs.thinkingmachines.ai/tinker/models/index.md`) lists base model rates for Qwen3.5-4B ($0.33/M prompt, $1.005/M sample) but does not list a separate rate for sampling a fine-tuned LoRA checkpoint. Evaluation and usage costs for the fine-tuned model are computed assuming the base model rate.
