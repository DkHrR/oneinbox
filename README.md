# OneInbox

OneInbox is an intelligent, privacy-first daily email digest assistant powered by a fine-tuned Qwen3.5-4B model and persistent memory via Backboard.

- 📄 **Sample Digest for Judges**: See [docs/sample_digest.md](docs/sample_digest.md) for a complete run output.
- 🤗 **Hugging Face Model**: [`dKhRr/oneinbox-qwen3.5-4b-lora`](https://huggingface.co/dKhRr/oneinbox-qwen3.5-4b-lora)

It categorizes incoming emails into four actionable tiers:
- 🚨 **must_act**: Needs a reply or action, or has a deadline within 7 days (bills, deadlines, security alerts that ask you to verify)
- 💡 **worth_a_look**: High-value opportunities, meetups, beta releases, or personal notes
- ℹ️ **fyi**: Receipts, confirmations, and status or security notices that need no action
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
   - A lightweight Flask service serving `/digest?token=...` protected by `DIGEST_TOKEN`. The page shows timestamps in UTC.
   - Each item includes one-click feedback buttons (`Always show`, `Silence sender`, `Accurate`, `Wrong`).
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
| `INGEST_TOKEN` | Secret authorization token used by the daily digest workflow (`.github/workflows/daily_digest.yml`) to POST the digest to the web service. |
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
> **Note**: The live demo uses the author's private Tinker checkpoint. Judges can read [docs/sample_digest.md](docs/sample_digest.md), and the adapter weights are on Hugging Face for self-hosting.

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

## Deployment on Render (`render.yaml`) & GitHub Actions

The system runs on a 100% free-tier split architecture:
1. **Web Service on Render** (`oneinbox-web`): Defined in `render.yaml`, runs Gunicorn on Render's free web plan to serve the mobile digest and HMAC feedback endpoints. Render hosts only the web service.
2. **Daily Digest on GitHub Actions** (`.github/workflows/daily_digest.yml`): Runs `python app.py` daily at 01:30 UTC (07:00 IST), not a Render cron job. GitHub scheduled runs can be delayed by a few minutes, and the workflow can also be started manually with "Run workflow".

### Storage Design: Shared State across Services
- The web service and GitHub Actions runner operate in separate environments and cannot share a local filesystem.
- Furthermore, Render free web services spin down after 15 minutes of inactivity.
- **Why Backboard?**: Backboard acts as the central, persistent memory and state layer. When the daily digest workflow completes on GitHub Actions, it posts the digest to the web service `/ingest` endpoint and persists rules and digest metadata directly to Backboard. When the web service wakes up, it fetches the state from Backboard, keeping the entire architecture 100% free without paid persistent disks.

---

## Privacy Notes

- **No Third-Party Notification Service**: No third-party notification service is used. The digest is read exclusively on a token-protected page (`/digest?token=...`). A push notification could be added later.
- **Digest Snapshot Storage**: The digest snapshot (sender, subject, label, deadline, summary, reason, never email bodies) is stored in Backboard memory so the web service can read it.
- **Never Logged**: Raw email bodies, full headers, and API keys are never printed to stdout/stderr or written to version control.
- **Strict Read-Only IMAP**: IMAP connections use `mail.select("INBOX", readonly=True)`, preventing any modification, flagging, or deletion of user emails.
- **Pre-Model Redaction**: PII (phone numbers, OTP codes, card/account numbers, and external URLs) is stripped before prompt assembly.
- **Secrets Protected**: `.env`, `latest_digest.json`, and `digest_preview.md` are added to `.gitignore`.

---

## Known limits

- **Backboard Memory Capacity**: Backboard memories have a character limit of 4,000 characters. With compressed JSON snapshots averaging ~128 stored characters per email (measured from the real 25-email run at 3,194 characters), approximately **31 real emails** fit under the 4,000 limit before snapshot capacity is reached. The digest is limited to about 31 emails by Backboard's memory size.
- **Synthetic Benchmark & Generator-Made Labels**: The training set and validation benchmark (`data/val.jsonl`) consist of synthetic emails designed to capture distribution shifts, deadlines, and urgency phrasing. The test set is synthetic with generator-made labels.
- **IMAP Mode Untested on Real Inboxes**: While the IMAP pipeline implements standard RFC822 parsing, PII redaction, and read-only fetching, IMAP mode is untested on a real inbox with complex HTML multipart layouts.
- **Feedback Scope**: Accurate/Wrong feedback is logged but does not change filtering (only preference rules like `always_show` and `ignore` change routing).
- **Fine-Tuned Pricing Model**: The Tinker documentation rate card (`https://tinker-docs.thinkingmachines.ai/tinker/models/index.md`) lists base model rates for Qwen3.5-4B ($0.33/M prompt, $1.005/M sample) but does not list a separate rate for sampling a fine-tuned LoRA checkpoint. Evaluation and usage costs for the fine-tuned model are computed assuming the base model rate.
