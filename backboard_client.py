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
    raw_json = json.dumps(digest_data)
    if len(raw_json) > 3500:
        import zlib, base64
        content = "DIGEST_SNAPSHOT_Z:" + base64.b64encode(zlib.compress(raw_json.encode("utf-8"))).decode("ascii")
    else:
        content = "DIGEST_SNAPSHOT: " + raw_json

    if len(content) > 4000:
        raise ValueError(f"Digest snapshot size ({len(content)} chars) exceeds Backboard memory limit of 4000 characters. Cannot persist digest.")

    add_memory(api_key, assistant_id, content)
    
    # Cleanup old snapshots
    try:
        mems = list_memories(api_key, assistant_id)
        snapshot_mems = [m for m in mems if m.get("content", "").startswith("DIGEST_SNAPSHOT: ") or m.get("content", "").startswith("DIGEST_SNAPSHOT_Z:")]
        # Sort descending by created_at, keep first
        snapshot_mems.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        for old_mem in snapshot_mems[1:]:
            try:
                requests.delete(f"{BASE_URL}/assistants/{assistant_id}/memories/{old_mem['id']}", headers=get_headers(api_key), timeout=10)
            except Exception as e:
                logger.warning(f"Failed to delete old snapshot {old_mem['id']}: {e}")
    except Exception as e:
        logger.warning(f"Failed during old snapshot cleanup: {e}")

def get_digest_from_backboard(api_key: str, assistant_id: str) -> dict | None:
    try:
        mems = list_memories(api_key, assistant_id)
        # Find all DIGEST_SNAPSHOT memories
        snapshots = []
        for mem in mems:
            content = mem.get("content", "")
            if content.startswith("DIGEST_SNAPSHOT: ") or content.startswith("DIGEST_SNAPSHOT_Z:"):
                snapshots.append(mem)
        
        if not snapshots:
            return None
            
        # Pick the newest by created_at
        newest = max(snapshots, key=lambda x: x.get("created_at", ""))
        content = newest.get("content", "")
        if content.startswith("DIGEST_SNAPSHOT_Z:"):
            import zlib, base64
            compressed_str = content[len("DIGEST_SNAPSHOT_Z:"):]
            raw = zlib.decompress(base64.b64decode(compressed_str)).decode("utf-8")
            return json.loads(raw)
        else:
            raw = content[len("DIGEST_SNAPSHOT: "):]
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Could not retrieve digest snapshot from Backboard: {e}")
    return None
