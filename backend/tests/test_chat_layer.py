"""Tests for prompt, agent and API using fakes, so no Gemini key or network is needed."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import agent as agent_module
from app.agent import FALLBACK_REPLY, SalesAgent, recommend_nozzle
from app.main import app, get_agent
from app.prompts import build_system_prompt
from app.recommender import load_catalog


# ---------- prompt ----------
def test_prompt_contains_every_catalog_nozzle_and_the_rules():
    catalog = load_catalog()
    prompt = build_system_prompt(catalog)
    for nozzle in catalog["nozzles"]:
        assert nozzle["name"] in prompt
    for system in catalog["accessories_and_systems"]:
        assert system["name"] in prompt
    assert "Aarya" in prompt
    assert "Never invent" in prompt
    assert "NOT IN THE CATALOG" in prompt
    assert "{" not in prompt and "}" not in prompt  # no unrendered placeholders


def test_prompt_covers_language_rule_and_closing_steps():
    prompt = build_system_prompt(load_catalog())
    assert "Marathi" in prompt and "Hindi" in prompt and "Tamil" in prompt
    assert "follow-up call" in prompt
    assert "recommend_nozzle" in prompt


def test_prompt_declines_off_topic_and_forbids_nagging():
    prompt = build_system_prompt(load_catalog())
    assert "Do not tell jokes" in prompt
    assert "Never nag" in prompt
    assert "designed for" in prompt  # listed as a phrase to avoid


# ---------- tool wrapper ----------
def test_tool_wrapper_matches_recommender():
    result = recommend_nozzle("stent coating", viscosity_cp=10, flow_ml_min=3, droplet_size_um=10)
    assert result["status"] == "match"
    assert result["best"]["id"] == "std_torch_180khz"


def test_tool_wrapper_handles_unknowns_as_zero():
    assert recommend_nozzle()["status"] == "needs_more_info"


# ---------- agent ----------
class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send_message(self, message):
        self.sent.append(message)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


class FakeClient:
    def __init__(self, replies):
        self.created = 0
        self._replies = replies
        self.chats = SimpleNamespace(create=self._create)

    def _create(self, model, config):
        self.created += 1
        return FakeChat(self._replies)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(agent_module.time, "sleep", lambda s: None)


def test_agent_keeps_one_chat_per_session():
    client = FakeClient(["Hello!", "Sure.", "Another."])
    agent = SalesAgent(client=client)
    assert agent.reply("a", "hi") == "Hello!"
    assert agent.reply("a", "more") == "Sure."
    assert client.created == 1
    agent.reply("b", "hi")
    assert client.created == 2


def test_agent_retries_then_succeeds():
    client = FakeClient([RuntimeError("503"), "Recovered."])
    agent = SalesAgent(client=client)
    assert agent.reply("a", "hi") == "Recovered."


def test_agent_returns_friendly_fallback_after_repeated_failures():
    client = FakeClient([RuntimeError("w"), RuntimeError("x"), RuntimeError("y"), RuntimeError("z")])
    agent = SalesAgent(client=client)
    assert agent.reply("a", "hi") == FALLBACK_REPLY


def test_agent_reset_starts_a_fresh_chat():
    client = FakeClient(["one", "two"])
    agent = SalesAgent(client=client)
    agent.reply("a", "hi")
    agent.reset("a")
    agent.reply("a", "hi")
    assert client.created == 2


# ---------- HTTP API ----------
class FakeAgent:
    def reply(self, session_id, message):
        return f"echo:{message}"

    def reset(self, session_id):
        pass


@pytest.fixture()
def client():
    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_chat_returns_reply_and_session_id(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "echo:hello"
    assert body["session_id"]


def test_chat_keeps_the_given_session_id(client):
    r = client.post("/chat", json={"message": "hello", "session_id": "abc"})
    assert r.json()["session_id"] == "abc"


def test_chat_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "   "}).status_code == 400