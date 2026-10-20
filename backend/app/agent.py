"""The sales agent: a small controller around Gemini.

Why not let the model call tools itself? In testing, the model often skipped tool calls and
answered from the prompt, which would let it claim a booking that never happened. So each turn is:

  1. EXTRACT  one low-temperature Gemini call turns the latest message into structured facts
  2. DECIDE   plain code updates the state, runs the recommender and submits the quotation
  3. REPLY    a second Gemini call writes the answer, given the verified results as "live context"

Recommendations and bookings therefore always come from code, and the model can only report them.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google import genai
from google.genai import types

from . import quotation
from .extractor import extraction_request, parse_extraction
from .prompts import build_system_prompt
from .recommender import load_catalog, recommend
from .state import ConversationState
from .timeutil import now_ist

log = logging.getLogger("wavenxd.agent")

DEFAULT_MODEL = "gemini-3-flash-preview"  # override with GEMINI_MODEL in .env
MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 40
PRIVATE_FIELDS = {"name", "company", "email", "phone"}
FALLBACK_REPLY = (
    "I'm sorry, I ran into a technical problem just now. Could you please repeat that? "
    "If it keeps happening, the WaveNxD team will be happy to help you directly."
)


@dataclass
class Session:
    history: list[tuple[str, str]] = field(default_factory=list)  # ("user" | "model", text)
    state: ConversationState = field(default_factory=ConversationState)


def _product_id(recommendation: Optional[dict[str, Any]]) -> Optional[str]:
    """The catalog id the quotation should be about."""
    if not recommendation or recommendation.get("status") != "match":
        return None
    if recommendation.get("lead") == "system" and recommendation.get("systems_to_consider"):
        return recommendation["systems_to_consider"][0]["id"]
    return recommendation["best"]["id"]


def _loggable(extracted: dict[str, Any]) -> dict[str, Any]:
    """Extraction result for the log, with personal details hidden."""
    return {
        k: ("<given>" if k in PRIVATE_FIELDS else v)
        for k, v in extracted.items()
        if v not in (None, "", [])
    }


class SalesAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
        fallback_model: Optional[str] = None,
        extract_model: Optional[str] = None,
    ):
        primary = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        backup = fallback_model if fallback_model is not None else os.getenv("GEMINI_FALLBACK_MODEL", "")
        extractor = extract_model or os.getenv("GEMINI_EXTRACT_MODEL", "") or primary
        self.models = [m for m in dict.fromkeys([primary, backup]) if m]
        self.extract_models = [m for m in dict.fromkeys([extractor, backup]) if m]
        self.client = client or genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._sessions: dict[str, Session] = {}

    # ---------- sessions ----------
    def _session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            if len(self._sessions) >= MAX_SESSIONS:
                self._sessions.pop(next(iter(self._sessions)))  # drop the oldest
            self._sessions[session_id] = Session()
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ---------- Gemini calls with retry and model fallback ----------
    def _generate(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        models: Optional[list[str]] = None,
    ) -> Optional[str]:
        models = models or self.models
        attempts = 4 if len(models) == 1 else 2
        for m_index, model in enumerate(models):
            for attempt in range(attempts):
                try:
                    response = self.client.models.generate_content(model=model, contents=contents, config=config)
                    text = (response.text or "").strip()
                    if text:
                        return text
                    log.warning("Empty response from %s", model)
                except Exception as exc:  # network, quota, overload, safety block, ...
                    log.warning("Gemini call failed (%s, attempt %d): %s", model, attempt + 1, exc)
                last_try = m_index == len(models) - 1 and attempt == attempts - 1
                if not last_try:
                    time.sleep(2 ** attempt)
        return None

    # ---------- the controller ----------
    def _decide(self, state: ConversationState, history: Optional[list[tuple[str, str]]] = None) -> list[str]:
        """Run what must run in code and return the live-context notes for the reply."""
        history = history or []
        notes: list[str] = []
        known = state.known_lines()
        if known:
            notes.append("Known about the client so far:\n" + "\n".join(f"- {line}" for line in known))

        # --- recommender ---
        if state.requirement_ready():
            key = state.requirement_key()
            if key != state.last_recommendation_key:
                result = recommend(
                    application=state.application,
                    viscosity_cp=state.viscosity_cp,
                    flow_ml_min=state.flow_ml_min,
                    droplet_size_um=state.droplet_size_um,
                )
                state.recommendation, state.last_recommendation_key = result, key
                best = (result.get("best") or {}).get("id")
                log.info("Controller: recommender ran -> status=%s lead=%s best=%s", result["status"], result.get("lead"), best)
                notes.append(
                    "NEW RECOMMENDER RESULT (authoritative; recommend only from this): "
                    + json.dumps(result, ensure_ascii=False)
                )
            elif state.recommendation:
                notes.append(
                    "The recommendation was already given earlier; stay consistent with it. "
                    "RECOMMENDER RESULT: " + json.dumps(state.recommendation, ensure_ascii=False)
                )
        else:
            state.recommendation = None
            state.last_recommendation_key = None
            notes.append(
                "The recommender has NOT run yet because the application and at least one of viscosity "
                "or flow rate are still missing. Do not recommend a specific product yet."
            )

        missing = state.missing_discovery()
        if missing and not state.wants_quotation:
            notes.append(
                "Still unknown (ask at most two, naturally; skip anything the client already said they "
                "do not know): " + "; ".join(missing)
            )

        # --- quotation ---
        if state.wants_quotation is False:
            notes.append("The client declined a quotation for now. Do not push it; keep helping.")
        elif state.wants_quotation:
            notes.extend(self._decide_quotation(state))
        elif state.recommendation and any("quotation" in text.lower() for role, text in history if role == "model"):
            notes.append(
                "You have already offered a quotation earlier in this conversation. Do not offer it again "
                "or ask for contact details unless the client asks for it; just answer their question."
            )
        return notes

    def _decide_quotation(self, state: ConversationState) -> list[str]:
        product_id = _product_id(state.recommendation)
        if not product_id:
            return [
                "The client wants a quotation but no product has been recommended yet. Finish the "
                "requirement discovery first (application plus viscosity or flow rate), then recommend."
            ]
        missing = state.missing_contact()
        if missing:
            return [
                "The client wants a quotation. Still needed: " + ", ".join(missing) + ". Ask for at most "
                "the next two items, in this order: name and company; email and phone; follow-up call date "
                "and time. If they have not picked a call time, offer two or three concrete future slots "
                "(for example tomorrow at 11:00 AM or 4:00 PM IST). Nothing has been emailed or booked yet."
            ]

        signature = (
            state.name, state.company, state.email, state.phone,
            state.followup_date, state.followup_time, product_id,
        )
        if signature == state.submitted_signature and state.quote_result:
            return [
                "The quotation request was already submitted and the call booked: "
                + json.dumps(state.quote_result, ensure_ascii=False)
                + " Do not submit it again."
            ]

        result = quotation.submit_quotation_request(
            name=state.name,
            company=state.company,
            email=state.email,
            phone=state.phone,
            application=state.application or "",
            recommended_product_id=product_id,
            followup_date=state.followup_date,
            followup_time=state.followup_time,
            requirement_notes=state.requirement_notes(),
        )
        log.info(
            "Controller: quotation submitted -> status=%s email_status=%s lead_id=%s",
            result["status"], result.get("email_status"), result.get("lead_id"),
        )
        if result["status"] == "ok":
            state.submitted_signature, state.quote_result = signature, result
            return ["QUOTATION RESULT (just completed, report it honestly): " + json.dumps(result, ensure_ascii=False)]

        errors = result["errors"]
        text = " ".join(errors).lower()
        if "email" in text:
            state.email = None
        if "phone" in text:
            state.phone = None
        if "follow-up" in text or "followup" in text:
            state.followup_date = state.followup_time = None
        if "name is missing" in text:
            state.name = None
        if "company is missing" in text:
            state.company = None
        return [
            "QUOTATION ATTEMPT FAILED (nothing was emailed or booked). Errors: " + "; ".join(errors)
            + ". Explain kindly and ask only for the corrected detail."
        ]

    # ---------- one conversation turn ----------
    def reply(self, session_id: str, message: str) -> str:
        session = self._session(session_id)
        now = now_ist()

        # 1. extract (a failure here only means "no new facts this turn")
        contents, config = extraction_request(session.history, message, now, session.state.known_lines())
        extracted = parse_extraction(self._generate(contents, config, self.extract_models))
        log.info("Extracted: %s", _loggable(extracted))
        session.state.merge(extracted, message)

        # 2. decide in code
        notes = self._decide(session.state, session.history)

        # 3. reply
        system = (
            build_system_prompt(load_catalog(), now)
            + "\n\n# Live context for this turn (written by the system)\n"
            + "\n\n".join(notes)
        )
        turns = (session.history + [("user", message)])[-MAX_HISTORY_MESSAGES:]
        contents = [types.Content(role=role, parts=[types.Part.from_text(text=text)]) for role, text in turns]
        reply = self._generate(contents, types.GenerateContentConfig(system_instruction=system, temperature=0.4))
        if reply is None:
            log.error("Giving up: no reply from any model")
            return FALLBACK_REPLY

        session.history += [("user", message), ("model", reply)]
        return reply