"""Controller tests with a scripted fake Gemini: no network, no key.

The fake answers two kinds of calls: extraction (JSON mode) and reply (plain text).
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app import agent as agent_module
from app import leads
from app.agent import FALLBACK_REPLY, OVERCLAIM, SalesAgent
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
        text = self.reply_text.pop(0) if isinstance(self.reply_text, list) else self.reply_text
        return SimpleNamespace(text=text)

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


# ---------- honesty: nothing may be claimed before the controller has really submitted ----------
def test_next_step_asks_only_for_the_missing_phone_and_forbids_claims():
    partial = {"wants_quotation": True, "name": "Test User", "company": "Test company",
               "email": "test.user@example.com", "followup_date": "2030-01-15", "followup_time": "14:00"}
    agent, models = make_agent([STENT, partial])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "details, call day after tomorrow at 2 pm")
    system = models.replies()[1][2]
    assert "ask ONLY for: phone" in system
    assert "already suggested 2030-01-15 at 14:00" in system
    assert "Do NOT say that the call is scheduled" in system
    assert leads.list_leads() == []


def test_next_step_order_is_name_company_then_contact_then_time():
    agent, models = make_agent([STENT, {"wants_quotation": True}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes quote")
    assert "ask ONLY for: name and company" in models.replies()[1][2]


def test_false_booking_claim_is_rewritten_before_it_reaches_the_client():
    partial = {"wants_quotation": True, "name": "Test User", "company": "Test company",
               "email": "test.user@example.com", "followup_date": "2030-01-15", "followup_time": "14:00"}
    replies = ["Reply one.", "We have scheduled your follow-up call.", "Could you share your phone number?"]
    agent, models = make_agent([STENT, partial], reply_text=replies)
    agent.reply("s", STENT_MSG)
    final = agent.reply("s", "day after tomorrow at 2 pm")
    assert final == "Could you share your phone number?"
    assert "CORRECTION" in models.replies()[2][2]
    assert agent._sessions["s"].history[-1] == ("model", "Could you share your phone number?")


def test_persistent_false_claim_falls_back_to_the_safe_message():
    partial = {"wants_quotation": True, "name": "Test User", "company": "Test company",
               "email": "test.user@example.com", "followup_date": "2030-01-15", "followup_time": "14:00"}
    replies = ["Reply one.", "We have scheduled your call.", "Your call is booked and emailed. It has been sent."]
    agent, _ = make_agent([STENT, partial], reply_text=replies)
    agent.reply("s", STENT_MSG)
    assert agent.reply("s", "2 pm") == FALLBACK_REPLY


def test_a_true_confirmation_after_a_real_submission_is_not_touched():
    agent, models = make_agent(
        [STENT, CONTACT], reply_text=["Reply one.", "Your quotation has been successfully emailed."]
    )
    agent.reply("s", STENT_MSG)
    assert agent.reply("s", "details") == "Your quotation has been successfully emailed."
    assert len(models.replies()) == 2  # no rewrite


@pytest.mark.parametrize("text", [
    "We have scheduled your follow-up call for 22 October.",
    "Your quotation has been successfully emailed to you.",
    "I've booked the call.",
    "Our team has recorded these details.",
    "The call is confirmed.",
])
def test_overclaim_pattern_catches_completion_claims(text):
    assert OVERCLAIM.search(text)


@pytest.mark.parametrize("text", [
    "Could you share your phone number?",
    "We can schedule a call tomorrow at 11:00 AM or 4:00 PM IST.",
    "I will need your email address before the quotation can be sent.",
    "The WaveNxD team will confirm the price.",
])
def test_overclaim_pattern_leaves_normal_questions_alone(text):
    assert not OVERCLAIM.search(text)


def test_http_timeout_is_applied_to_the_gemini_client(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            captured["timeout"] = http_options.timeout

    monkeypatch.setattr(agent_module.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12")
    SalesAgent(api_key="x")
    assert captured["timeout"] == 12000


# ---------- a second client in the same chat must not cause duplicate or wrong emails ----------
def outbox_count(tmp_path):
    return len(list((tmp_path / "outbox").glob("*.eml")))


def test_a_second_contact_gets_exactly_one_request_after_all_details_are_given(tmp_path):
    agent, models = make_agent(
        [
            STENT,
            CONTACT,
            {"name": "User 3", "company": "Test company 3"},       # "do the same for user 3, details different"
            {"email": "user3@example.com"},
            {"phone": "7744558877"},
            {"followup_date": "2030-01-16", "followup_time": "19:00"},
        ]
    )
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes please quote, details")
    assert len(leads.list_leads()) == 1

    agent.reply("s", "do the same for user 3 from test company 3, his contact details are different")
    assert len(leads.list_leads()) == 1  # the old email, phone and time must NOT be reused
    assert "ask ONLY for: email and phone" in models.replies()[2][2]

    agent.reply("s", "user3@example.com")
    assert len(leads.list_leads()) == 1
    assert "ask ONLY for: phone" in models.replies()[3][2]

    agent.reply("s", "7744558877")
    assert len(leads.list_leads()) == 1
    assert "ask ONLY for: follow-up call date and time" in models.replies()[4][2]

    agent.reply("s", "tomorrow 7 pm")
    rows = leads.list_leads()
    assert len(rows) == 2 and outbox_count(tmp_path) == 2
    assert rows[1]["name"] == "User 3" and rows[1]["email"] == "user3@example.com"
    assert "QUOTATION RESULT (just completed" in models.replies()[5][2]


def test_new_contact_flag_clears_the_old_contact_details():
    agent, models = make_agent([STENT, CONTACT, {"new_contact": True}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes please quote, details")
    agent.reply("s", "do the same for another client")
    assert "ask ONLY for: name and company" in models.replies()[2][2]
    assert len(leads.list_leads()) == 1


def test_a_different_product_after_a_submission_does_not_send_an_unrequested_email():
    agent, models = make_agent(
        [STENT, CONTACT, {"application": "fuel cell coating", "viscosity_cp": 500, "flow_ml_min": 2}]
    )
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes please quote, details")
    agent.reply("s", "now fuel cell coating, 500 cP at 2 ml/min")
    assert len(leads.list_leads()) == 1
    third = models.replies()[2][2]
    assert "NEW RECOMMENDER RESULT" in third and "hv_conical" in third
    assert "QUOTATION RESULT (just completed" not in third


def test_changing_the_call_time_after_submission_is_not_claimed_as_done():
    replies = ["Reply one.", "Reply two.", "I have rescheduled your call to 5 pm.", "The WaveNxD team can confirm a change on the call."]
    agent, models = make_agent([STENT, CONTACT, {"followup_time": "17:00"}], reply_text=replies)
    agent.reply("s", STENT_MSG)
    agent.reply("s", "yes please quote, details")
    final = agent.reply("s", "make the call 5 pm instead")
    assert len(leads.list_leads()) == 1
    assert "You cannot change a submitted request yourself" in models.replies()[2][2]
    assert final == "The WaveNxD team can confirm a change on the call."  # the false claim was rewritten


def test_completed_requests_are_remembered_for_later_turns():
    agent, models = make_agent([STENT, CONTACT, {}])
    for text in (STENT_MSG, "yes please quote, details", "thanks"):
        agent.reply("s", text)
    third = models.replies()[2][2]
    assert "already submitted in this conversation" in third
    assert "Test Labs Pvt Ltd" in third
    assert "Never claim that any other request was submitted" in third


def test_correcting_a_name_before_any_submission_keeps_the_other_details():
    agent, _ = make_agent([STENT, {"wants_quotation": True, "name": "Test Usr", "email": "a@example.com"}, {"name": "Test User"}])
    agent.reply("s", STENT_MSG)
    agent.reply("s", "quote, I am Test Usr, a@example.com")
    agent.reply("s", "sorry, my name is Test User")
    state = agent._sessions["s"].state
    assert state.name == "Test User" and state.email == "a@example.com"


def test_discovery_hint_asks_for_viscosity_and_flow_once_the_application_is_known():
    agent, models = make_agent([{"application": "fuel cell coating"}])
    agent.reply("s", "मला इंधन सेल कोटिंगसाठी नोझल हवा आहे")
    system = models.replies()[0][2]
    assert "Do not recommend or name a specific product yet" in system
    assert "Ask first for the liquid viscosity and the required flow rate" in system


def test_recommender_runs_from_the_application_alone_if_the_client_knows_neither_number():
    agent, models = make_agent(
        [{"application": "stent coating", "client_does_not_know": ["viscosity_cp", "flow_ml_min"]}]
    )
    agent.reply("s", "stent coating, I do not know the viscosity or flow rate")
    assert "NEW RECOMMENDER RESULT" in models.replies()[0][2]


# ---------- state-level behaviour ----------
def test_state_identity_change_after_a_submission_starts_a_new_contact():
    s = ConversationState()
    s.merge({**CONTACT})
    s.submitted_history.append("done")
    s.quote_result, s.submitted_signature = {"status": "ok"}, (*s.contact_signature(), "std_torch_180khz")
    s.merge({"name": "Someone Else"})
    assert s.name == "Someone Else"
    assert s.email is None and s.phone is None and s.followup_date is None
    assert s.quote_result is None and s.wants_quotation is True


def test_new_contact_flag_is_ignored_when_nothing_was_submitted_yet():
    s = ConversationState()
    s.merge({"name": "Test User", "email": "a@example.com", "new_contact": True})
    assert s.email == "a@example.com"