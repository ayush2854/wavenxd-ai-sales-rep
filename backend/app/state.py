"""What the assistant knows about the client so far (kept in code, not in the model's head)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .recommender import keyword_tokens

DISCOVERY_LABELS = {
    "industry": "the client's industry",
    "application": "the application (what they want to spray or coat)",
    "material": "the liquid or coating material",
    "viscosity_cp": "the liquid viscosity in cP",
    "flow_ml_min": "the required flow rate in ml/min",
    "droplet_size_um": "the required droplet size in micrometres",
}
UNKNOWN_KEYS = set(DISCOVERY_LABELS) - {"application"}

_EMPTY_WORDS = {"", "null", "none", "n/a", "na", "unknown"}


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in _EMPTY_WORDS else value


def _different_application(old: str, new: str) -> bool:
    """True when the client moved to a genuinely different application (no shared keywords)."""
    a, b = keyword_tokens(old), keyword_tokens(new)
    return bool(a) and bool(b) and not (a & b)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


@dataclass
class ConversationState:
    industry: Optional[str] = None
    application: Optional[str] = None
    material: Optional[str] = None
    viscosity_cp: Optional[float] = None
    flow_ml_min: Optional[float] = None
    droplet_size_um: Optional[float] = None

    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    followup_date: Optional[str] = None  # YYYY-MM-DD
    followup_time: Optional[str] = None  # HH:MM
    wants_quotation: Optional[bool] = None

    client_does_not_know: set[str] = field(default_factory=set)

    # controller bookkeeping
    recommendation: Optional[dict[str, Any]] = None
    last_recommendation_key: Optional[tuple] = None
    submitted_signature: Optional[tuple] = None
    quote_result: Optional[dict[str, Any]] = None
    submitted_history: list[str] = field(default_factory=list)  # one line per completed request

    # ---------- updating ----------
    def merge(self, extracted: dict[str, Any], message: Optional[str] = None) -> None:
        """Copy newly stated facts in. Nulls never erase what is already known.

        message is the client's latest message. Numbers are accepted only if it contains a digit,
        which blocks a model slip that re-reports an old number on a message like "yes".
        """
        self._maybe_start_new_contact(extracted)

        new_application = _text(extracted.get("application"))
        if new_application:
            if self.application and _different_application(self.application, new_application):
                # a different job: the earlier numbers described something else
                self.material = None
                self.viscosity_cp = self.flow_ml_min = self.droplet_size_um = None
                self.client_does_not_know = set()
            self.application = new_application

        for key in ("industry", "material", "name", "company", "email", "phone"):
            value = _text(extracted.get(key))
            if value:
                setattr(self, key, value)

        if message is None or any(ch.isdigit() for ch in message):
            for key in ("viscosity_cp", "flow_ml_min", "droplet_size_um"):
                value = _number(extracted.get(key))
                if value:
                    setattr(self, key, value)

        date = _text(extracted.get("followup_date"))
        if date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            self.followup_date = date
        time_ = _text(extracted.get("followup_time"))
        if time_ and re.fullmatch(r"\d{1,2}:\d{2}", time_):
            self.followup_time = time_.zfill(5)

        if isinstance(extracted.get("wants_quotation"), bool):
            self.wants_quotation = extracted["wants_quotation"]

        unknown = extracted.get("client_does_not_know")
        if isinstance(unknown, list):
            self.client_does_not_know |= {u for u in unknown if u in UNKNOWN_KEYS}

    def _maybe_start_new_contact(self, extracted: dict[str, Any]) -> None:
        """After a request was submitted, a different person or company means a NEW request.

        Without this, the previous person's email, phone and call time would be reused for the new
        person and several wrong quotation emails would go out.
        """
        if not self.submitted_history:
            return
        new_name, new_company = _text(extracted.get("name")), _text(extracted.get("company"))
        identity_changed = bool(new_name and self.name and new_name.lower() != self.name.lower()) or bool(
            new_company and self.company and new_company.lower() != self.company.lower()
        )
        if extracted.get("new_contact") is True or identity_changed:
            self.start_new_contact()

    def start_new_contact(self) -> None:
        self.name = self.company = self.email = self.phone = None
        self.followup_date = self.followup_time = None
        self.quote_result = None
        self.submitted_signature = None
        self.wants_quotation = True  # asking for another contact's quotation is asking for a quotation

    # ---------- questions the controller asks ----------
    def requirement_ready(self) -> bool:
        if not self.application:
            return False
        if self.viscosity_cp or self.flow_ml_min:
            return True
        # the client does not know either number: recommend from the application alone
        return {"viscosity_cp", "flow_ml_min"} <= self.client_does_not_know

    def requirement_key(self) -> tuple:
        return ((self.application or "").lower(), self.viscosity_cp, self.flow_ml_min, self.droplet_size_um)

    def missing_discovery(self) -> list[str]:
        missing = []
        for key, label in DISCOVERY_LABELS.items():
            if getattr(self, key) is None and key not in self.client_does_not_know:
                missing.append(label)
        return missing

    def missing_contact(self) -> list[str]:
        missing = []
        if not self.name:
            missing.append("name")
        if not self.company:
            missing.append("company")
        if not self.email:
            missing.append("email")
        if not self.phone:
            missing.append("phone")
        if not (self.followup_date and self.followup_time):
            missing.append("follow-up call date and time")
        return missing

    def contact_signature(self) -> tuple:
        return (self.name, self.company, self.email, self.phone, self.followup_date, self.followup_time)

    def contact_changed(self) -> bool:
        """True when the contact details or call time no longer match the submitted request."""
        return self.submitted_signature is not None and self.contact_signature() != self.submitted_signature[:6]

    def requirement_notes(self) -> str:
        parts = []
        if self.material:
            parts.append(f"material: {self.material}")
        if self.viscosity_cp:
            parts.append(f"viscosity {self.viscosity_cp:g} cP")
        if self.flow_ml_min:
            parts.append(f"flow {self.flow_ml_min:g} ml/min")
        if self.droplet_size_um:
            parts.append(f"droplet size {self.droplet_size_um:g} µm")
        return "; ".join(parts)

    def known_lines(self) -> list[str]:
        lines = []
        for key, title in (
            ("industry", "Industry"),
            ("application", "Application"),
            ("material", "Liquid or coating material"),
        ):
            if getattr(self, key):
                lines.append(f"{title}: {getattr(self, key)}")
        for key, title, unit in (
            ("viscosity_cp", "Viscosity", "cP"),
            ("flow_ml_min", "Flow rate", "ml/min"),
            ("droplet_size_um", "Droplet size", "µm"),
        ):
            if getattr(self, key):
                lines.append(f"{title}: {getattr(self, key):g} {unit}")
        for key, title in (("name", "Name"), ("company", "Company"), ("email", "Email"), ("phone", "Phone")):
            if getattr(self, key):
                lines.append(f"{title}: {getattr(self, key)}")
        if self.followup_date and self.followup_time:
            lines.append(f"Chosen follow-up call: {self.followup_date} {self.followup_time} IST")
        return lines