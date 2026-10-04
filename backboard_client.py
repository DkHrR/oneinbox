import os
import json
import logging
import requests

logger = logging.getLogger(__name__)
BASE_URL = "https://app.backboard.io/api"

def get_headers(api_key: str) -> dict:
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

def get_or_create_assistant(api_key: str, assistant_id: str = None) -> str:
    headers = get_headers(api_key)
    if assistant_id:
        # Verify it exists
        try:
            res = requests.get(f"{BASE_URL}/assistants/{assistant_id}", headers=headers, timeout=10)
            if res.status_code == 200:
                return assistant_id
        except Exception:
            pass

    try:
        res = requests.get(f"{BASE_URL}/assistants", headers=headers, timeout=10)
        if res.status_code == 200:
            assistants = res.json()
            for a in assistants:
                if a.get("name") == "OneInbox":
                    return a.get("assistant_id")
    except Exception as e:
        logger.warning(f"Failed to list assistants: {e}")

    # Create new
    payload = {
        "name": "OneInbox",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "memory": "Auto"
    }
    res = requests.post(f"{BASE_URL}/assistants", json=payload, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()
    return data.get("assistant_id")

def list_memories(api_key: str, assistant_id: str) -> list[dict]:
    headers = get_headers(api_key)
    res = requests.get(f"{BASE_URL}/assistants/{assistant_id}/memories", headers=headers, timeout=10)
    if res.status_code == 200:
        data = res.json()
        return data.get("memories", [])
    return []

def add_memory(api_key: str, assistant_id: str, content: str) -> dict:
    headers = get_headers(api_key)
    payload = {"content": content}
    res = requests.post(f"{BASE_URL}/assistants/{assistant_id}/memories", json=payload, headers=headers, timeout=10)
    res.raise_for_status()
    return res.json()

def parse_rules(memories: list[dict]) -> tuple[set[str], set[str]]:
    always_show = set()
    ignore = set()
    for mem in memories:
        content = mem.get("content", "").strip()
        if content.startswith("RULE: always_show sender="):
            val = content.split("sender=", 1)[1].strip().lower()
            if val:
                always_show.add(val)
        elif content.startswith("RULE: ignore sender="):
            val = content.split("sender=", 1)[1].strip().lower()
            if val:
                ignore.add(val)
    return always_show, ignore

def save_digest_to_backboard(api_key: str, assistant_id: str, digest_data: dict):
    # Store snapshot as a memory for cross-container retrieval on Render free tier
    try:
        content = "DIGEST_SNAPSHOT: " + json.dumps(digest_data)
        add_memory(api_key, assistant_id, content)
    except Exception as e:
        logger.warning(f"Could not persist digest snapshot to Backboard: {e}")

def get_digest_from_backboard(api_key: str, assistant_id: str) -> dict | None:
    try:
        mems = list_memories(api_key, assistant_id)
        # Find all DIGEST_SNAPSHOT memories
        snapshots = []
        for mem in mems:
            content = mem.get("content", "")
            if content.startswith("DIGEST_SNAPSHOT: "):
                snapshots.append(mem)
        
        if not snapshots:
            return None
            
        # Pick the newest by created_at
        # Assuming mem has a 'created_at' field (ISO8601 string or timestamp)
        newest = max(snapshots, key=lambda x: x.get("created_at", ""))
        raw = newest.get("content", "")[len("DIGEST_SNAPSHOT: "):]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Could not retrieve digest snapshot from Backboard: {e}")
    return None
