"""Gemini-powered sales agent.

One chat session per browser session. The model can call recommend_nozzle,
which runs the deterministic recommender, so recommendations always come
from catalog.json and not from the model's memory.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from .prompts import build_system_prompt
from .recommender import load_catalog, recommend

log = logging.getLogger("wavenxd.agent")

DEFAULT_MODEL = "gemini-3-flash-preview"  # override with GEMINI_MODEL in .env
MAX_SESSIONS = 200
FALLBACK_REPLY = (
    "I'm sorry, I ran into a technical problem just now. Could you please repeat that? "
    "If it keeps happening, the WaveNxD team will be happy to help you directly."
)


def recommend_nozzle(
    application: str = "",
    viscosity_cp: float = 0,
    flow_ml_min: float = 0,
    droplet_size_um: float = 0,
) -> dict[str, Any]:
    """Find the best WaveNxD ultrasonic nozzle or system for a client's requirements.

    Call this before recommending any product. Pass only what the client has told you;
    leave a value at 0 (or the application empty) if it is not known yet.

    Args:
        application: What the client wants to spray or coat, in English, for example
            "stent coating", "fuel cell coating", "industrial spray dryer",
            "PIPAC chemotherapy", "syringe coating", "transparent conductive coating".
        viscosity_cp: Liquid viscosity in centipoise (cP). 0 if unknown.
        flow_ml_min: Required flow rate in ml/min. 0 if unknown.
        droplet_size_um: Required droplet size in micrometres. 0 if unknown.

    Returns:
        A dictionary with status ("match", "no_match" or "needs_more_info"), the best
        nozzle with reasons and warnings, alternatives, and any systems to consider.
    """
    log.info(
        "Tool call recommend_nozzle(application=%r, viscosity_cp=%s, flow_ml_min=%s, droplet_size_um=%s)",
        application, viscosity_cp, flow_ml_min, droplet_size_um,
    )
    return recommend(
        application=application,
        viscosity_cp=viscosity_cp,
        flow_ml_min=flow_ml_min,
        droplet_size_um=droplet_size_um,
    )


class SalesAgent:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client: Any = None):
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.client = client or genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.system_prompt = build_system_prompt(load_catalog())
        self._sessions: dict[str, Any] = {}

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[recommend_nozzle],  # the SDK runs the function automatically
            temperature=0.4,
        )

    def _chat(self, session_id: str) -> Any:
        if session_id not in self._sessions:
            if len(self._sessions) >= MAX_SESSIONS:
                self._sessions.pop(next(iter(self._sessions)))  # drop the oldest
            self._sessions[session_id] = self.client.chats.create(model=self.model, config=self._config())
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reply(self, session_id: str, message: str) -> str:
        chat = self._chat(session_id)
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                response = chat.send_message(message)
                text = (response.text or "").strip()
                if text:
                    return text
                last_error = RuntimeError("empty response")
            except Exception as exc:  # network, quota, safety block, etc.
                last_error = exc
                log.warning("Gemini call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
        log.error("Giving up after retries: %s", last_error)
        return FALLBACK_REPLY