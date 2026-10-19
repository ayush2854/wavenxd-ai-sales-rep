"""Quotation, lead storage, email and calendar invite tests (no network, no real email)."""
import email as email_lib
from datetime import datetime
from email import policy

import pytest

from app import emailer, leads, quotation
from app.agent import submit_quotation_request as tool_wrapper
from app.prompts import build_system_prompt
from app.recommender import load_catalog
from app.timeutil import IST

NOW = datetime(2026, 10, 20, 10, 0, tzinfo=IST)  # Tuesday

GOOD = dict(
    name="Test Client",
    company="Test Labs Pvt Ltd",
    email="test.client@example.com",
    phone="+91 98765 43210",
    application="stent coating",
    recommended_product_id="std_torch_180khz",
    followup_date="2026-10-21",
    followup_time="16:00",
    requirement_notes="10 cP, 3 ml/min, 10 um droplets",
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVENXD_DATA_DIR", str(tmp_path))
    for var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PORT", "SALES_TEAM_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def submit(**overrides):
    return quotation.submit_quotation_request(**{**GOOD, **overrides}, now=NOW)


# ---------- happy path ----------
def test_valid_request_is_stored_and_saved_when_smtp_not_configured(isolated):
    result = submit()
    assert result["status"] == "ok"
    assert result["email_status"] == "saved"
    assert result["followup_call"] == "Wednesday, 21 October 2026 at 04:00 PM IST"
    rows = leads.list_leads()
    assert len(rows) == 1
    assert rows[0]["company"] == "Test Labs Pvt Ltd"
    assert rows[0]["phone"] == "919876543210"
    assert rows[0]["email_status"] == "saved"
    assert len(list((isolated / "outbox").glob("*.eml"))) == 1


def test_saved_email_has_catalog_specs_ics_invite_and_no_prices(isolated):
    submit()
    eml = next((isolated / "outbox").glob("*.eml"))
    msg = email_lib.message_from_bytes(eml.read_bytes(), policy=policy.default)
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "180 kHz" in body and "10 µm" in body
    assert "pricing and delivery time, will be confirmed" in body
    assert "₹" not in body and "Rs." not in body
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_content_type() == "text/calendar"
    ics = attachments[0].get_content().replace("\r\n", "\n") if isinstance(attachments[0].get_content(), str) else attachments[0].get_payload(decode=True).decode()
    assert "DTSTART:20261021T103000Z" in ics  # 16:00 IST is 10:30 UTC


def test_ics_uses_crlf_and_escapes_text():
    ics = emailer.build_ics(NOW, "Call", "a, b; c").decode()
    assert ics.startswith("BEGIN:VCALENDAR\r\n")
    assert "DESCRIPTION:a\\, b\\; c" in ics


# ---------- validation ----------
@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"email": "not-an-email"}, "valid email"),
        ({"phone": "12345"}, "valid phone"),
        ({"name": ""}, "name is missing"),
        ({"company": ""}, "company is missing"),
        ({"recommended_product_id": "made_up_nozzle"}, "not a catalog id"),
        ({"followup_date": "21/10/2026"}, "YYYY-MM-DD"),
        ({"followup_date": "2026-10-19"}, "in the past"),
    ],
)
def test_invalid_input_returns_errors_and_stores_nothing(overrides, fragment):
    result = submit(**overrides)
    assert result["status"] == "error"
    assert any(fragment in e for e in result["errors"])
    assert leads.list_leads() == []


def test_micro_dispensing_system_can_be_quoted(isolated):
    assert submit(recommended_product_id="micro_dispensing")["status"] == "ok"


# ---------- SMTP behaviour ----------
class FakeSMTP:
    sent = []
    fail = False

    def __init__(self, host, port, timeout=None):
        if FakeSMTP.fail:
            raise OSError("connection refused")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


@pytest.fixture()
def smtp(monkeypatch):
    FakeSMTP.sent, FakeSMTP.fail = [], False
    monkeypatch.setattr(emailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("SALES_TEAM_EMAIL", "team@example.com")
    return FakeSMTP


def test_email_is_sent_through_smtp_when_configured(smtp):
    result = submit()
    assert result["email_status"] == "sent"
    assert len(smtp.sent) == 1
    assert smtp.sent[0]["To"] == "test.client@example.com"
    assert smtp.sent[0]["Bcc"] == "team@example.com"


def test_smtp_failure_is_reported_honestly_and_email_is_saved(smtp, isolated):
    smtp.fail = True
    result = submit()
    assert result["status"] == "ok"
    assert result["email_status"] == "failed_saved"
    assert len(list((isolated / "outbox").glob("*.eml"))) == 1


# ---------- wiring ----------
def test_tool_wrapper_reaches_the_quotation_logic():
    result = tool_wrapper(**{**GOOD, "followup_date": "2030-01-15"})
    assert result["status"] == "ok"


def test_prompt_carries_todays_date_and_the_quotation_rules():
    prompt = build_system_prompt(load_catalog(), now=NOW)
    assert "Tuesday, 20 October 2026" in prompt
    assert "submit_quotation_request" in prompt
    assert "never say it was sent" in prompt