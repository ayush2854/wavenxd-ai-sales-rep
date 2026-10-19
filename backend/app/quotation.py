"""Validate a quotation request, store the lead, book the follow-up and send the email.

One function does all four steps so the assistant makes a single, hard-to-get-wrong tool call.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from . import emailer, leads
from .recommender import load_catalog
from .timeutil import IST, now_ist

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_phone(phone: str) -> Optional[str]:
    digits = re.sub(r"[\s\-().]", "", phone or "")
    if digits.startswith("+"):
        digits = digits[1:]
    return digits if digits.isdigit() and 10 <= len(digits) <= 13 else None


def submit_quotation_request(
    name: str,
    company: str,
    email: str,
    phone: str,
    application: str,
    recommended_product_id: str,
    followup_date: str,
    followup_time: str,
    requirement_notes: str = "",
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    catalog = load_catalog()
    now = now or now_ist()
    name, company = (name or "").strip(), (company or "").strip()
    email, application = (email or "").strip(), (application or "").strip()
    requirement_notes = (requirement_notes or "").strip()

    errors: list[str] = []
    if len(name) < 2:
        errors.append("name is missing")
    if len(company) < 2:
        errors.append("company is missing")
    if not EMAIL_RE.match(email):
        errors.append(f"'{email}' does not look like a valid email address; ask the client to confirm it")
    phone_clean = _clean_phone(phone)
    if phone_clean is None:
        errors.append(f"'{phone}' does not look like a valid phone number (10 to 13 digits); ask the client to confirm it")
    product = emailer.describe_product(catalog, (recommended_product_id or "").strip())
    if product is None:
        errors.append(
            f"'{recommended_product_id}' is not a catalog id; use best.id (or a systems_to_consider id) "
            "exactly as returned by recommend_nozzle"
        )

    start: Optional[datetime] = None
    try:
        start = datetime.strptime(f"{followup_date} {followup_time}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
    except ValueError:
        errors.append("followup_date must be YYYY-MM-DD and followup_time must be HH:MM in 24-hour format")
    else:
        if start <= now:
            errors.append("the follow-up date and time is in the past; ask the client for a future slot")

    if errors:
        return {"status": "error", "errors": errors}

    assert product is not None and start is not None
    product_title, product_lines = product
    when_text = start.strftime("%A, %d %B %Y at %I:%M %p") + " IST"

    lead_id = leads.save_lead(
        name=name,
        company=company,
        email=email,
        phone=phone_clean,
        application=application,
        requirement_notes=requirement_notes,
        recommended_product_id=recommended_product_id.strip(),
        followup_at=start,
    )
    msg = emailer.build_quotation_email(
        to_email=email,
        name=name,
        company=company,
        application=application,
        requirement_notes=requirement_notes,
        product_title=product_title,
        product_lines=product_lines,
        when_text=when_text,
        start=start,
    )
    email_status = emailer.deliver(msg, leads.data_dir() / "outbox")
    leads.update_email_status(lead_id, email_status)

    return {
        "status": "ok",
        "lead_id": lead_id,
        "email_status": email_status,
        "followup_call": when_text,
        "summary": {
            "name": name,
            "company": company,
            "email": email,
            "application": application,
            "requirement_notes": requirement_notes,
            "recommended_product": product_title,
        },
    }