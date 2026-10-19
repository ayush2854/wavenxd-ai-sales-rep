"""Quotation-request email with a calendar invite attached.

Delivery outcomes (the assistant may only say "emailed" for "sent"):
  "sent"          delivered through SMTP
  "saved"         SMTP is not configured, so the .eml file was saved to data/outbox
  "failed_saved"  SMTP is configured but failed, so the .eml file was saved to data/outbox
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from .timeutil import now_ist

log = logging.getLogger("wavenxd.email")

CALL_MINUTES = 30


def _esc(text: str) -> str:
    """Escape text for an iCalendar property value."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_ics(start: datetime, summary: str, description: str, uid: Optional[str] = None) -> bytes:
    """A minimal .ics event. Times are converted to UTC, so no timezone block is needed."""
    stamp = "%Y%m%dT%H%M%SZ"
    utc = timezone.utc
    end = start + timedelta(minutes=CALL_MINUTES)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//WaveNxD//AI Sales Representative//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid or uuid.uuid4()}@wavenxd-sales-assistant",
        f"DTSTAMP:{datetime.now(utc).strftime(stamp)}",
        f"DTSTART:{start.astimezone(utc).strftime(stamp)}",
        f"DTEND:{end.astimezone(utc).strftime(stamp)}",
        f"SUMMARY:{_esc(summary)}",
        f"DESCRIPTION:{_esc(description)}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def describe_product(catalog: dict[str, Any], product_id: str) -> Optional[tuple[str, list[str]]]:
    """Title and spec lines for a catalog id, taken only from catalog.json."""
    for n in catalog["nozzles"]:
        if n["id"] == product_id:
            feed = f"{'up to ' if n.get('feed_rate_is_max') else ''}{n['feed_rate_ml_min']:g} ml/min"
            visc = f"{n['viscosity_max_cp']:g} cP" if n.get("viscosity_note") else f"below {n['viscosity_max_cp']:g} cP"
            return n["name"], [
                f"Frequency: {n['frequency_khz']} kHz",
                f"Droplet size: {n['droplet_size_um']} µm",
                f"Feed rate: {feed}",
                f"Viscosity: {visc}",
                f"Listed applications: {', '.join(n['applications'])}",
            ]
    for s in catalog["accessories_and_systems"]:
        if s["id"] == product_id:
            lines = [s["details"]]
            if s.get("applications"):
                lines.append(f"Listed applications: {', '.join(s['applications'])}")
            return s["name"], lines
    return None


def build_quotation_email(
    *,
    to_email: str,
    name: str,
    company: str,
    application: str,
    requirement_notes: str,
    product_title: str,
    product_lines: list[str],
    when_text: str,
    start: datetime,
) -> EmailMessage:
    sender_name = os.getenv("SENDER_NAME", "WaveNxD Sales Assistant")
    sender_email = os.getenv("SENDER_EMAIL") or os.getenv("SMTP_USER") or "noreply@example.com"

    msg = EmailMessage()
    msg["Subject"] = f"WaveNxD quotation request: {product_title}"
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to_email
    team = os.getenv("SALES_TEAM_EMAIL")
    if team:
        msg["Bcc"] = team  # the client does not see this address

    spec_block = "\n".join(f"  - {line}" for line in product_lines)
    body = (
        f"Dear {name},\n\n"
        "Thank you for speaking with the AI sales assistant of WaveNxD Technologies. "
        "This email summarises your quotation request.\n\n"
        f"Company: {company}\n"
        f"Application: {application}\n"
        f"Your requirements: {requirement_notes or 'As discussed in the chat'}\n\n"
        f"Suggested product: {product_title}\n{spec_block}\n\n"
        f"Follow-up call: {when_text}. A calendar invite is attached.\n\n"
        "The detailed quotation, including pricing and delivery time, will be confirmed by the "
        "WaveNxD team. The assistant does not quote prices or delivery times.\n\n"
        f"Regards,\n{sender_name}\nWaveNxD Technologies Pvt. Ltd., Kolhapur\n"
    )
    msg.set_content(body)

    ics = build_ics(
        start,
        summary="WaveNxD follow-up call",
        description=f"Follow-up call with {name} ({company}) about: {product_title}",
    )
    msg.add_attachment(ics, maintype="text", subtype="calendar", filename="wavenxd-followup-call.ics")
    return msg


def _save_eml(msg: EmailMessage, outbox: Path) -> Path:
    outbox.mkdir(parents=True, exist_ok=True)
    safe_to = re.sub(r"[^A-Za-z0-9]+", "_", str(msg["To"]))
    path = outbox / f"{now_ist().strftime('%Y%m%d-%H%M%S')}-{safe_to}.eml"
    path.write_bytes(msg.as_bytes())
    return path


def deliver(msg: EmailMessage, outbox: Path) -> str:
    host, user, password = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    if not (host and user and password):
        path = _save_eml(msg, outbox)
        log.info("SMTP not configured; saved email to %s", path)
        return "saved"
    port = int(os.getenv("SMTP_PORT", "587"))
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(msg)
        log.info("Quotation email sent to %s", msg["To"])
        return "sent"
    except Exception as exc:
        path = _save_eml(msg, outbox)
        log.error("SMTP delivery failed (%s); saved email to %s", exc, path)
        return "failed_saved"