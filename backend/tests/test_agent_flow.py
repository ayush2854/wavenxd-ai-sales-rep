"""Controller tests with a scripted fake Gemini: no network, no key.

The fake answers two kinds of calls: extraction (JSON mode) and reply (plain text).
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app import agent as agent_module
from app import leads
from app.agent import FALLBACK_REPLY, SalesAgent
from app.extractor import extraction_request, parse_extraction
from app.state import ConversationState
from app.timeutil import IST

STENT = {"application": "stent coating", "viscosity_cp": 10, "flow_ml_min": 3, "droplet_size_um": 10}
STENT_MSG = "I want to coat stents, 10 cP, 3 ml/min, 10 micron droplets"
CONTACT = {
    "wants_quotation": True,
    "name": "Test Client",
    "company": "Test Labs Pvt Ltd",
    "email": "test.client@example.com",
    "phone": "7788998877",
    "followup_date": "2030-01-15",
    "followup_time": "16:00",
}


class FakeModels:
    def __init__(self, extractions, fail_models=(), reply_text="Reply from Aarya."):
        self.extractions = list(extractions)
        self.fail_models = set(fail_models)
        self.reply_text = reply_text
        self.calls = []  # (model, kind, system_instruction, contents)

    def generate_content(self, model, contents, config):
        kind = "extract" if config.response_mime_type == "application/json" else "reply"
        self.calls.append((model, kind, config.system_instruction, contents))
        if model in self.fail_models:
            raise RuntimeError("503 UNAVAILABLE")
        if kind == "extract":
            item = self.extractions.pop(0) if self.extractions else {}
            if isinstance(item, Exception):
                raise item
            return SimpleNamespace(text=item if isinstance(item, str) else json.dumps(item))
        return SimpleNamespace(text=self.reply_text)

    def replies(self):
        return [c for c in self.calls if c[1] == "reply"]

    def extractions_made(self):
        return [c for c in self.calls if c[1] == "extract"]


def make_agent(extractions, **kwargs):
    models = FakeModels(
        extractions, kwargs.pop("fail_models", ()), kwargs.pop("reply_text", "Reply from Aarya.")
    )
    client = SimpleNamespace(models=models)
    return SalesAgent(client=client, model=kwargs.pop("model", "primary"), **kwargs), models


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module.time, "sleep", lambda s: None)
    monkeypatch.setenv("WAVENXD_DATA_DIR", str(tmp_path))
    for var in (
        "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SALES_TEAM_EMAIL",
        "GEMINI_FALLBACK_MODEL", "GEMINI_EXTRACT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------- state ----------
def test_state_merge_ignores_nulls_and_bad_values():
    s = ConversationState()
    s.merge({"application": "stent coating", "viscosity_cp": "10", "flow_ml_min": -3, "email": "null"})
    assert s.application == "stent coating" and s.viscosity_cp == 10.0
    assert s.flow_ml_min is None and s.email is None
    s.merge({"application": None, "viscosity_cp": None})
    assert s.application == "stent coating" and s.viscosity_cp == 10.0  # nulls never erase


def test_state_normalises_time_and_ignores_bad_dates():
    s = ConversationState()
    s.merge({"followup_date": "tomorrow", "followup_time": "4:00"})
    assert s.followup_date is None and s.followup_time == "04:00"


def test_state_missing_lists():
    s = ConversationState()
    s.merge({"client_does_not_know": ["droplet_size_um", "bogus"]})
    assert not any("droplet" in m for m in s.missing_discovery())
    assert s.missing_contact() == ["name", "company", "email", "phone", "follow-up call date and time"]


def test_numbers_are_ignored_when_the_message_has_no_digit():
    s = ConversationState()
    s.merge(STENT, STENT_MSG)
    s.merge({"viscosity_cp": 5000, "flow_ml_min": 500}, "yes")  # a model slip on a message like "yes"
    assert (s.viscosity_cp, s.flow_ml_min) == (10.0, 3.0)
    s.merge({"viscosity_cp": 5000}, "actually it is 5000 cP")
    assert s.viscosity_cp == 5000.0


def test_devanagari_digits_count_as_digits():
    s = ConversationState()
    s.merge({"application": "fuel cell coating", "viscosity_cp": 500}, "चिकटपणा ५०० cP आहे")
    assert s.viscosity_cp == 500.0


def test_new_application_clears_old_numbers_but_a_rephrasing_does_not():
    s = ConversationState()
    s.merge(STENT, STENT_MSG)
    s.merge({"application": "coating of stents"}, "same job")
    assert s.viscosity_cp == 10.0  # same keyword "stent": not a new job
    s.merge({"application": "fuel cell coating"}, "now a fuel cell job")
    assert s.viscosity_cp is None and s.flow_ml_min is None and s.droplet_size_um is None
    assert s.application == "fuel cell coating"


def test_numbers_in_the_same_message_as_a_new_application_are_kept():
    s = ConversationState()
    s.merge(STENT, STENT_MSG)
    s.merge({"application": "fuel cell coating", "viscosity_cp": 500}, "fuel cell, 500 cP")
    assert s.viscosity_cp == 500.0 and s.flow_ml_min is None


def test_parse_extraction_survives_garbage():
    assert parse_extraction("not json") == {}
    assert parse_extraction(None) == {}
    assert parse_extraction('```json\n{"application": "x"}\n```') == {"application": "x"}
    assert parse_extraction("[1, 2]") == {}


def test_extraction_prompt_focuses_on_the_latest_message_and_shows_known_facts():
    now = datetime(2026, 10, 20, 11, 0, tzinfo=IST)
    contents, config = extraction_request(
        [("user", "old"), ("model", "hello")], "yes", now, ["Application: stent coating"]
    )
    prompt = contents[0].parts[0].text
    assert "Client (LATEST): yes" in prompt
    assert "Application: stent coating" in prompt
    assert "Tuesday, 20 October 2026" in prompt
    assert config.response_mime_type == "application/json"


# ---------- recommender is run by code ----------
def test_recommender_runs_in_code_and_result_reaches_the_reply():
    agent, models = make_agent([STENT])
    assert agent.reply("s", STENT_MSG) == "Reply from Aarya."
    system = models.replies()[0][2]
    assert "NEW RECOMMENDER RESULT" in system
    assert "std_torch_180khz" in system


def test_no_recommendation_until_requirement_is_known():
    agent, models = make_agent([{}])
    agent.reply("s", "Hello")
    system = models.replies()[0][2]
    assert "has NOT run yet" in system
    assert "NEW RECOMMENDER RESULT" not in system


def test_recommender_not_rerun_when_nothing_changed_but_rerun_when_it_does():
    agent, models = make_agent(
        [STENT, {}, {"application": "fuel cell coating", "viscosity_cp": 500, "flow_ml_min": 2}]
    )
    agent.reply("s", STENT_MSG)
    agent.reply("s", "thanks")
    agent.reply("s", "now fuel cell coating, 500 cP, 2 ml/min")
    first, second, third = (c[2] for c in models.replies())
    assert "NEW RECOMMENDER RESULT" in first
    assert "NEW RECOMMENDER RESULT" not in second and "already given" in second
    assert "NEW RECOMMENDER RESULT" in third and "hv_conical" in third


def test_a_stale_number_slip_on_a_plain_yes_does_not_change_the_recommendation():
    slip = {"viscosity_cp": 5000, "flow_ml_min": 500}  # the model re-reports old numbers on "yes"
    agent, models = make_agent([STENT, slip])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes")
    second = models.replies()[1][2]
    assert "NEW RECOMMENDER RESULT" not in second and '"status": "no_match"' not in second
    assert "std_torch_180khz" in second


def test_switching_application_does_not_reuse_old_numbers():
    agent, models = make_agent([STENT, {"application": "fuel cell coating"}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "मला इंधन सेल कोटिंगसाठी नोझल हवा आहे")
    second = models.replies()[1][2]
    assert "has NOT run yet" in second
    assert "viscosity in cP" in second  # it will ask for the missing values again


def test_out_of_range_requirement_reaches_reply_as_no_match():
    agent, models = make_agent([{"application": "coating", "viscosity_cp": 5000, "flow_ml_min": 500}])
    agent.reply("s", "5000 cP at 500 ml/min")
    assert '"status": "no_match"' in models.replies()[0][2]


def test_quotation_is_not_re_offered_once_it_has_been_offered():
    agent, models = make_agent([STENT, {}], reply_text="Would you like a quotation?")
    agent.reply("s", STENT_MSG)
    agent.reply("s", "what is the delivery time?")
    assert "already offered a quotation" in models.replies()[1][2]
    assert "already offered a quotation" not in models.replies()[0][2]


# ---------- quotation is submitted by code ----------
def test_quotation_is_submitted_when_everything_is_known():
    agent, models = make_agent([STENT, CONTACT])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes please quote, details...")
    system = models.replies()[1][2]
    assert "QUOTATION RESULT (just completed" in system
    assert '"email_status": "saved"' in system
    rows = leads.list_leads()
    assert len(rows) == 1 and rows[0]["company"] == "Test Labs Pvt Ltd"
    assert rows[0]["recommended_product_id"] == "std_torch_180khz"
    assert "droplet size 10" in rows[0]["requirement_notes"]


def test_quotation_is_not_submitted_twice():
    agent, models = make_agent([STENT, CONTACT, {}])
    for text in (STENT_MSG, "quote please", "thanks"):
        agent.reply("s", text)
    assert len(leads.list_leads()) == 1
    assert "already submitted" in models.replies()[2][2]


def test_missing_contact_details_are_asked_for_and_nothing_is_claimed():
    agent, models = make_agent([STENT, {"wants_quotation": True, "name": "Test Client"}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes quote, I am Test Client")
    system = models.replies()[1][2]
    assert "Still needed: company, email, phone, follow-up call date and time" in system
    assert "Nothing has been emailed or booked yet" in system
    assert leads.list_leads() == []


def test_invalid_email_is_rejected_cleared_and_can_be_corrected():
    bad = {**CONTACT, "email": "not-an-email"}
    agent, models = make_agent([STENT, bad, {"email": "good.one@example.com"}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "details with a bad email")
    assert "QUOTATION ATTEMPT FAILED" in models.replies()[1][2]
    assert leads.list_leads() == []
    agent.reply("s", "my email is good.one@example.com")
    assert "QUOTATION RESULT (just completed" in models.replies()[2][2]
    assert len(leads.list_leads()) == 1


def test_quotation_without_a_recommendation_is_not_attempted():
    agent, models = make_agent([CONTACT])
    agent.reply("s", "send me a quotation")
    assert "no product has been recommended yet" in models.replies()[0][2]
    assert leads.list_leads() == []


def test_declined_quotation_is_not_pushed():
    agent, models = make_agent([STENT, {"wants_quotation": False}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "no thanks")
    assert "declined a quotation" in models.replies()[1][2]


# ---------- resilience ----------
def test_extraction_failure_still_produces_a_reply():
    agent, models = make_agent([RuntimeError("boom")] * 4)
    assert agent.reply("s", "hello") == "Reply from Aarya."


def test_garbage_extraction_still_produces_a_reply():
    agent, _ = make_agent(["this is not json"])
    assert agent.reply("s", "hello") == "Reply from Aarya."


def test_fallback_model_is_used_when_primary_is_overloaded():
    agent, models = make_agent([STENT], fail_models={"primary"}, fallback_model="backup")
    assert agent.reply("s", STENT_MSG) == "Reply from Aarya."
    assert {c[0] for c in models.calls} == {"primary", "backup"}
    assert models.replies()[-1][0] == "backup"


def test_a_separate_extraction_model_can_be_used():
    agent, models = make_agent([STENT], extract_model="strong")
    agent.reply("s", STENT_MSG)
    assert models.extractions_made()[0][0] == "strong"
    assert models.replies()[0][0] == "primary"


def test_total_failure_gives_a_friendly_message_and_keeps_history_clean():
    agent, _ = make_agent([], fail_models={"primary"})
    assert agent.reply("s", "hello") == FALLBACK_REPLY
    assert agent._sessions["s"].history == []


# ---------- sessions ----------
def test_history_is_sent_and_sessions_are_isolated():
    agent, models = make_agent([{}, {}, {}])
    agent.reply("a", "first")
    agent.reply("a", "second")
    agent.reply("b", "other client")
    texts_a = [p.parts[0].text for p in models.replies()[1][3]]
    assert texts_a == ["first", "Reply from Aarya.", "second"]
    assert [p.parts[0].text for p in models.replies()[2][3]] == ["other client"]


def test_reset_forgets_the_conversation():
    agent, models = make_agent([STENT, {}])
    agent.reply("a", STENT_MSG)
    agent.reset("a")
    agent.reply("a", "hello again")
    assert "has NOT run yet" in models.replies()[1][2]