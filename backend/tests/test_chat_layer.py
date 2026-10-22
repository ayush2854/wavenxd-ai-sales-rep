"""Prompt and HTTP API tests (fakes only, no Gemini key or network needed)."""
import pytest
from fastapi.testclient import TestClient

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
    assert "Live context" in prompt
    assert "RECOMMENDER RESULT" in prompt


def test_prompt_declines_off_topic_and_forbids_nagging():
    prompt = build_system_prompt(load_catalog())
    assert "Do not tell jokes" in prompt
    assert "Never nag" in prompt
    assert "designed for" in prompt  # listed as a phrase to avoid


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


# ---------- built UI is served by the backend ----------
def test_built_frontend_is_served_and_the_api_still_works(tmp_path):
    from fastapi import FastAPI

    from app.main import mount_frontend

    (tmp_path / "index.html").write_text("<html>hello ui</html>", encoding="utf-8")
    other = FastAPI()

    @other.get("/health")
    def health():
        return {"status": "ok"}

    assert mount_frontend(other, tmp_path) is True
    c = TestClient(other)
    assert "hello ui" in c.get("/").text
    assert c.get("/health").json() == {"status": "ok"}  # API routes registered before the mount still win


def test_frontend_is_not_mounted_when_it_has_not_been_built(tmp_path):
    from fastapi import FastAPI

    from app.main import mount_frontend

    assert mount_frontend(FastAPI(), tmp_path) is False