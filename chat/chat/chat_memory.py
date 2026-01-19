import json
import os
from datetime import datetime

# In-memory storage (session-based)
CHAT_STORE = {}

CHAT_LOG_DIR = "chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

def create_session(session_id: str):
    CHAT_STORE[session_id] = []

def save_message(session_id: str, role: str, content: str):
    message = {
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "content": content
    }

    CHAT_STORE.setdefault(session_id, []).append(message)

    # Also persist to file (audit trail)
    file_path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(CHAT_STORE[session_id], f, indent=2)

def get_history(session_id: str):
    return CHAT_STORE.get(session_id, [])
