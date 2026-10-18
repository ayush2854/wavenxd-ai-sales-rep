"""Deterministic nozzle recommender.

Reads only backend/data/catalog.json. No LLM is involved here, so the same
requirements always give the same answer and every claim can be traced to a
catalog row. The chat agent calls recommend() as a tool and then explains the
result to the client in the client's language.

How it decides
--------------
1. Hard filters: a nozzle is dropped if the client's viscosity is above the
   nozzle's listed maximum, or the client's flow rate is above its feed rate.
2. Scoring of the survivors:
   - application match (strongest signal, 20 points per matching keyword)
   - droplet size closeness (up to 10 points)
   - flow rate fit, preferring a nozzle that is not hugely oversized (up to 5)
3. Honesty flags: warnings when the match is borderline or the application is
   not in the catalog, so the assistant can say the WaveNxD team will confirm.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

APPLICATION_POINTS = 20
DROPLET_POINTS = 10
FLOW_POINTS = 5

# Words that carry no information about the use case. Applied after trailing
# "s" removal, so "tubes" -> "tube" and "coatings" -> "coating".
STOPWORDS = {
    "spray", "coating", "coat", "system", "tube", "production", "high",
    "for", "and", "the", "use", "with", "want", "need", "our", "this", "that",
    "to", "in", "of", "an", "we", "is", "on", "my",
}

SYNONYMS = {
    "chemo": "chemotherapy",
    "drier": "dryer",
    "drying": "dryer",
}


@lru_cache(maxsize=1)
def load_catalog(path: str = str(CATALOG_PATH)) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: Optional[str]) -> set[str]:
    """Lower-case keywords of a text, with simple plural stripping."""
    if not text:
        return set()
    out: set[str] = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        word = SYNONYMS.get(word, word)
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if len(word) > 1 and word not in STOPWORDS:
            out.add(word)
    return out


def _positive(value: Optional[float]) -> Optional[float]:
    """Treat missing, zero or negative numbers as 'not provided'."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _best_application_match(client_tokens: set[str], phrases: list[str]) -> tuple[int, Optional[str]]:
    best_count, best_phrase = 0, None
    for phrase in phrases:
        count = len(client_tokens & _tokens(phrase))
        if count > best_count:
            best_count, best_phrase = count, phrase
    return best_count, best_phrase


def _fmt(value: float) -> str:
    return f"{value:g}"


def _score_nozzle(
    nozzle: dict[str, Any],
    client_tokens: set[str],
    viscosity_cp: Optional[float],
    flow_ml_min: Optional[float],
    droplet_size_um: Optional[float],
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    overlap, matched_phrase = _best_application_match(client_tokens, nozzle["applications"])
    if overlap:
        score += APPLICATION_POINTS * overlap
        reasons.append(f"The catalog lists it for {matched_phrase}.")

    if droplet_size_um is not None:
        listed = nozzle["droplet_size_um"]
        diff = abs(listed - droplet_size_um) / droplet_size_um
        score += DROPLET_POINTS * max(0.0, 1.0 - diff)
        reasons.append(
            f"Its droplet size is {_fmt(listed)} µm, against the {_fmt(droplet_size_um)} µm you need."
        )
        if diff > 0.25:
            warnings.append(
                f"Listed droplet size ({_fmt(listed)} µm) differs noticeably from the requested "
                f"{_fmt(droplet_size_um)} µm; the WaveNxD team should confirm."
            )

    if flow_ml_min is not None:
        capacity = nozzle["feed_rate_ml_min"]
        score += FLOW_POINTS * (flow_ml_min / capacity)
        prefix = "up to " if nozzle.get("feed_rate_is_max") else ""
        reasons.append(
            f"Its listed feed rate is {prefix}{_fmt(capacity)} ml/min, which covers your {_fmt(flow_ml_min)} ml/min."
        )

    if viscosity_cp is not None:
        limit = nozzle["viscosity_max_cp"]
        reasons.append(f"It handles liquids up to {_fmt(limit)} cP, and yours is {_fmt(viscosity_cp)} cP.")
        if viscosity_cp == limit:
            warnings.append(
                f"Your viscosity ({_fmt(viscosity_cp)} cP) is exactly at this nozzle's listed limit; "
                "the WaveNxD team should confirm."
            )

    return {
        "id": nozzle["id"],
        "name": nozzle["name"],
        "frequency_khz": nozzle["frequency_khz"],
        "droplet_size_um": nozzle["droplet_size_um"],
        "feed_rate_ml_min": nozzle["feed_rate_ml_min"],
        "viscosity_max_cp": nozzle["viscosity_max_cp"],
        "applications": nozzle["applications"],
        "application_match": matched_phrase,
        "score": round(score, 2),
        "reasons": reasons,
        "warnings": warnings,
    }


def _match_systems(catalog: dict[str, Any], client_tokens: set[str]) -> list[dict[str, Any]]:
    found = []
    for system in catalog.get("accessories_and_systems", []):
        phrases = system.get("applications")
        if not phrases:
            continue
        overlap, phrase = _best_application_match(client_tokens, phrases)
        if overlap:
            found.append(
                {
                    "id": system["id"],
                    "name": system["name"],
                    "details": system["details"],
                    "matched_application": phrase,
                    "_overlap": overlap,
                }
            )
    found.sort(key=lambda s: (-s["_overlap"], s["id"]))
    for s in found:
        del s["_overlap"]
    return found


def recommend(
    application: Optional[str] = None,
    viscosity_cp: Optional[float] = None,
    flow_ml_min: Optional[float] = None,
    droplet_size_um: Optional[float] = None,
    catalog: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a recommendation for the given client requirements.

    status is one of:
      "needs_more_info"  nothing usable was provided yet
      "match"            at least one catalog nozzle fits; see best / alternatives
      "no_match"         no single nozzle fits; the assistant must offer a team follow-up
    lead is "nozzle" or "system" and says what to present first when status is "match".
    """
    catalog = catalog or load_catalog()
    application = (application or "").strip() or None
    viscosity_cp = _positive(viscosity_cp)
    flow_ml_min = _positive(flow_ml_min)
    droplet_size_um = _positive(droplet_size_um)

    provided = {
        "application": application,
        "viscosity_cp": viscosity_cp,
        "flow_ml_min": flow_ml_min,
        "droplet_size_um": droplet_size_um,
    }
    missing = [name for name, value in provided.items() if value is None]
    if len(missing) == len(provided):
        return {
            "status": "needs_more_info",
            "missing_info": missing,
            "next_step": "Ask the client about their application, liquid viscosity, flow rate and droplet size.",
        }

    client_tokens = _tokens(application)
    systems = _match_systems(catalog, client_tokens)
    nozzles = catalog["nozzles"]

    candidates = []
    for nozzle in nozzles:
        if viscosity_cp is not None and viscosity_cp > nozzle["viscosity_max_cp"]:
            continue
        if flow_ml_min is not None and flow_ml_min > nozzle["feed_rate_ml_min"]:
            continue
        candidates.append(_score_nozzle(nozzle, client_tokens, viscosity_cp, flow_ml_min, droplet_size_um))

    if not candidates:
        reasons: list[str] = []
        max_visc = max(n["viscosity_max_cp"] for n in nozzles)
        max_flow = max(n["feed_rate_ml_min"] for n in nozzles)
        if viscosity_cp is not None and viscosity_cp > max_visc:
            reasons.append(f"The highest viscosity listed in the catalog is {_fmt(max_visc)} cP.")
        if flow_ml_min is not None and flow_ml_min > max_flow:
            reasons.append(f"The highest feed rate listed in the catalog is {_fmt(max_flow)} ml/min.")
        if not reasons and viscosity_cp is not None and flow_ml_min is not None:
            flow_of_visc_ok = max(n["feed_rate_ml_min"] for n in nozzles if viscosity_cp <= n["viscosity_max_cp"])
            visc_of_flow_ok = max(n["viscosity_max_cp"] for n in nozzles if flow_ml_min <= n["feed_rate_ml_min"])
            reasons.append(
                f"Nozzles that handle {_fmt(viscosity_cp)} cP are listed up to {_fmt(flow_of_visc_ok)} ml/min, "
                f"and nozzles that handle {_fmt(flow_ml_min)} ml/min are listed up to {_fmt(visc_of_flow_ok)} cP."
            )
        return {
            "status": "no_match",
            "reasons": reasons,
            "systems_to_consider": systems,
            "missing_info": missing,
            "next_step": (
                "Tell the client no catalog nozzle covers this combination, that the WaveNxD team "
                "will confirm what is possible, and offer a follow-up call."
            ),
        }

    candidates.sort(key=lambda c: (-c["score"], c["id"]))
    best, alternatives = candidates[0], candidates[1:3]

    warnings: list[str] = []
    application_matched = any(c["application_match"] for c in candidates)
    if application and not application_matched and not systems:
        warnings.append(
            f"No application in the catalog matches '{application}'. The ranking uses the technical "
            "specifications only; say the WaveNxD team will confirm suitability."
        )

    lead = "nozzle"
    if application and not best["application_match"] and systems:
        lead = "system"

    return {
        "status": "match",
        "lead": lead,
        "best": best,
        "alternatives": alternatives,
        "systems_to_consider": systems,
        "missing_info": missing,
        "warnings": warnings,
    }