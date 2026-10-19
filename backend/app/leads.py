"""Tiny SQLite store for quotation leads (sample data only, never real customer data)."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .timeutil import now_ist


def data_dir() -> Path:
    """backend/data by default; override with WAVENXD_DATA_DIR (the tests do)."""
    path = Path(os.getenv("WAVENXD_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "leads.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            company TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            application TEXT,
            requirement_notes TEXT,
            recommended_product_id TEXT,
            followup_at TEXT,
            email_status TEXT
        )"""
    )
    return conn


def save_lead(
    *,
    name: str,
    company: str,
    email: str,
    phone: str,
    application: str,
    requirement_notes: str,
    recommended_product_id: str,
    followup_at: datetime,
) -> int:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            """INSERT INTO leads (created_at, name, company, email, phone, application,
                                  requirement_notes, recommended_product_id, followup_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now_ist().isoformat(timespec="seconds"),
                name, company, email, phone, application, requirement_notes,
                recommended_product_id, followup_at.isoformat(timespec="minutes"),
            ),
        )
        return int(cur.lastrowid)


def update_email_status(lead_id: int, status: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE leads SET email_status = ? WHERE id = ?", (status, lead_id))


def list_leads() -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM leads ORDER BY id")]