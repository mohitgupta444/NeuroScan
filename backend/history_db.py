"""
SQLite persistence for scans and their chat history.

Every /predict call creates one "scan" row (storing the original image,
the segmentation mask, and the full JSON result). Every /explain and /chat
turn is appended to that scan's chat_messages, so reopening the app later
still shows the full conversation for a given scan.

This uses Python's built-in sqlite3 -- no extra dependency needed.
"""

import base64
import io
import json
import os
import sqlite3
import time
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "neuroscan.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            original_image_b64 TEXT,
            result_json TEXT NOT NULL,
            top_class TEXT,
            risk_band TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
    """)
    conn.commit()
    conn.close()


def create_scan(original_image_bytes, result_json):
    """Store a new scan. Returns the generated scan_id."""
    scan_id = uuid.uuid4().hex[:12]
    original_b64 = base64.b64encode(original_image_bytes).decode("utf-8")

    top = result_json.get("classification", [{}])[0]
    risk = result_json.get("risk", {})

    conn = get_conn()
    conn.execute(
        "INSERT INTO scans (id, created_at, original_image_b64, result_json, top_class, risk_band) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            scan_id,
            time.time(),
            original_b64,
            json.dumps(result_json),
            top.get("name", "unknown"),
            risk.get("band", "unknown"),
        ),
    )
    conn.commit()
    conn.close()
    return scan_id


def get_scan(scan_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "original_image_b64": row["original_image_b64"],
        "result": json.loads(row["result_json"]),
        "top_class": row["top_class"],
        "risk_band": row["risk_band"],
    }


def list_scans(limit=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, created_at, top_class, risk_band FROM scans "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(scan_id, role, content):
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_messages (scan_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (scan_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()


def get_messages(scan_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM chat_messages "
        "WHERE scan_id = ? ORDER BY created_at ASC",
        (scan_id,),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]


def delete_scan(scan_id):
    conn = get_conn()
    conn.execute("DELETE FROM chat_messages WHERE scan_id = ?", (scan_id,))
    conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    conn.commit()
    conn.close()


# Initialize the database file/tables as soon as this module is imported.
init_db()
