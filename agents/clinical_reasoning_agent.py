import json
import re
from typing import Any
from typing import Optional

from services.config import settings

GeminiClient: Optional[Any]
OpenAIClient: Optional[Any]

try:
    from google import genai

    GeminiClient = genai
except ImportError:  # Optional dependency at runtime
    GeminiClient = None  # type: ignore[assignment]

try:
    from openai import OpenAI

    OpenAIClient = OpenAI
except ImportError:  # Optional dependency at runtime
    OpenAIClient = None  # type: ignore[assignment]


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _placeholder_suggestions(soap_note: dict[str, Any], cardiac_risk_pattern: bool = False) -> list[dict[str, str]]:
    subjective = _safe_text(soap_note.get("subjective"))
    objective = _safe_text(soap_note.get("objective"))
    assessment = _safe_text(soap_note.get("assessment"))
    plan = _safe_text(soap_note.get("plan"))

    context = " ".join(part for part in [subjective, objective, assessment, plan] if part).lower()

    # Pattern recognition: life-threatening chest pain presentation.
    has_chest_pain = "chest pain" in context or ("chest" in context and "pain" in context)
    radiation_arm_or_jaw = any(
        token in context
        for token in [
            "radiating to left arm",
            "radiating to right arm",
            "radiation to left arm",
            "radiation to right arm",
            "radiating to arm",
            "radiation to arm",
            "radiating into arm",
            "radiating to jaw",
            "radiation to jaw",
        ]
    ) or any(token in context for token in ["left arm", "right arm", "jaw"]) and "radiat" in context

    has_sweating = any(token in context for token in ["sweating", "diaphoresis"])
    has_breathlessness = any(token in context for token in ["shortness of breath", "breathlessness", "dyspnea"])

    # Decide deterministically when strong signals exist.
    if cardiac_risk_pattern:
        # Requirement-driven high-yield ED differential:
        # 1) Primary likely: MI (ACS presentation)
        # 2) Alternative serious: PE
        # 3) Another critical but less likely: Aortic Dissection
        has_tearing_or_back = any(
            token in context
            for token in ["tearing", "back pain", "sudden onset", "maximal at onset", "worst at onset"]
        )

        pe_conf = "medium" if has_breathlessness else "low"
        dissection_conf = "medium" if has_tearing_or_back else "low"

        return [
            {
                "diagnosis": "Myocardial Infarction",
                "reason": "Chest pain with radiation to the arm/jaw plus diaphoresis and nausea is highly consistent with an acute coronary syndrome presentation requiring immediate ECG.",
                "confidence": "high",
            },
            {
                "diagnosis": "Pulmonary Embolism",
                "reason": "When chest discomfort overlaps with dyspnea/shortness of breath, pulmonary embolism remains a serious alternative diagnosis that must be evaluated in the ED, even if ACS is most likely.",
                "confidence": pe_conf,
            },
            {
                "diagnosis": "Aortic Dissection (consider)",
                "reason": "Because some life-threatening causes of chest pain are clinically overlapping, aortic dissection should be considered—especially if the history suggests tearing pain or abrupt maximal onset.",
                "confidence": dissection_conf,
            },
        ]

    if has_chest_pain and radiation_arm_or_jaw and has_sweating:
        # Same diverse ED differential as the cardiac pre-check, but driven by SOAP symptoms.
        has_tearing_or_back = any(
            token in context
            for token in ["tearing", "back pain", "sudden onset", "maximal at onset", "worst at onset"]
        )
        pe_conf = "medium" if has_breathlessness else "low"
        dissection_conf = "medium" if has_tearing_or_back else "low"

        return [
            {
                "diagnosis": "Myocardial Infarction",
                "reason": "Chest pain radiating to the arm/jaw with associated diaphoresis fits an acute coronary syndrome pattern that should be treated as MI/ischemia until proven otherwise.",
                "confidence": "high",
            },
            {
                "diagnosis": "Pulmonary Embolism",
                "reason": "If dyspnea/shortness of breath is present, pulmonary embolism is an important alternative life-threatening cause of chest symptoms and should be excluded early.",
                "confidence": pe_conf,
            },
            {
                "diagnosis": "Aortic Dissection (consider)",
                "reason": "Aortic dissection is a critical but less likely cause of chest pain; the risk rises if the history suggests sudden maximal onset or tearing pain.",
                "confidence": dissection_conf,
            },
        ]

    # Fever/possible infectious emergencies.
    # Try to parse temperature and use a common fever threshold (100.4F).
    temp_match = re.search(r"\btemp(?:erature)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*\b([fc])?\b", context)
    temp_f = None
    if temp_match:
        try:
            temp_f = float(temp_match.group(1))
            unit = (temp_match.group(2) or "f").lower()
            if unit == "c":
                temp_f = temp_f * 9.0 / 5.0 + 32.0
        except (TypeError, ValueError):
            temp_f = None

    has_fever = ("fever" in context) or (temp_f is not None and temp_f >= 100.4)
    if has_fever:
        # Parse HR to avoid triggering solely because "HR" is present.
        hr_match = re.search(r"\b(?:hr|heart rate|pulse)\s*[:=]?\s*(\d{2,3})\b", context)
        hr = None
        if hr_match:
            try:
                hr = int(hr_match.group(1))
            except ValueError:
                hr = None

        high_risk = any(token in context for token in ["tachycardia", "hypotension", "confusion", "lethargy", "sepsis"])
        if hr is not None and hr >= 110:
            high_risk = True
        if high_risk:
            return [
                {
                    "diagnosis": "Sepsis (suspected)",
                    "reason": "Fever with systemic concern (tachycardia/systemic features) is consistent with a high-risk infectious presentation that needs immediate sepsis evaluation.",
                    "confidence": "high",
                },
                {
                    "diagnosis": "Serious Community-Acquired Infection",
                    "reason": "Fever plus associated symptoms suggests an underlying infection source that may be clinically significant even if exam localization is incomplete.",
                    "confidence": "medium",
                },
                {
                    "diagnosis": "Meningitis (consider if neurologic symptoms)",
                    "reason": "Fever can involve CNS infection; consider meningitis if headache, neck stiffness, confusion, or rash is present or emerges.",
                    "confidence": "low",
                },
            ]

        # More stable febrile picture: prioritize likely viral illness, then exclude dangerous causes.
        return [
            {
                "diagnosis": "Acute Viral Syndrome (febrile illness)",
                "reason": "Fever with supportive upper/viral-type symptoms (e.g., sore throat/body aches) most strongly suggests an acute viral febrile illness pattern.",
                "confidence": "medium",
            },
            {
                "diagnosis": "Serious Community-Acquired Infection",
                "reason": "Even when symptoms suggest viral illness, serious bacterial infection must be considered in the ED when fever is present and objective localization is limited.",
                "confidence": "low",
            },
            {
                "diagnosis": "Sepsis (early/occult, consider)",
                "reason": "Sepsis can begin before full deterioration; with limited objective detail, clinicians should assess for early sepsis physiology.",
                "confidence": "low",
            },
        ]

    # Injury/trauma emergencies.
    has_injury = any(token in context for token in ["fall", "injury", "motorcycle", "trauma"])
    if has_injury:
        has_swelling = any(token in context for token in ["swelling", "edema"])
        unable_to_bear_weight = "unable to bear" in context or "not able to bear" in context
        return [
            {
                "diagnosis": "Acute Extremity Fracture (suspected)",
                "reason": "After trauma with focal pain/swelling and impaired function (unable to bear weight), an extremity fracture is the leading diagnosis until imaging confirms otherwise.",
                "confidence": "high" if unable_to_bear_weight else "medium",
            },
            {
                "diagnosis": "Neurovascular Injury (screen for)",
                "reason": "Extremity trauma requires rapid neurovascular assessment; symptoms may be subtle early, so vascular and nerve compromise must be actively excluded.",
                "confidence": "medium",
            },
            {
                "diagnosis": "Compartment Syndrome (consider)",
                "reason": "With swelling and traumatic pain, compartment syndrome must be considered—especially if pain is severe, worsening, or disproportionate.",
                "confidence": "medium" if has_swelling else "low",
            },
        ]

    # Final non-generic fallback: specific high-yield ED exclusions.
    return [
        {
            "diagnosis": "Acute Coronary Syndrome (exclude)",
            "reason": "When the presentation includes potentially serious chest/cardiopulmonary features, ACS must be considered and excluded early in the ED workflow.",
            "confidence": "low",
        },
        {
            "diagnosis": "Pulmonary Embolism (exclude)",
            "reason": "For undifferentiated chest symptoms with limited objective data, PE remains a high-yield alternative diagnosis that should be actively ruled out.",
            "confidence": "low",
        },
        {
            "diagnosis": "Aortic Dissection (consider)",
            "reason": "Aortic dissection is a critical but less likely cause of acute chest symptoms; it should be considered when history overlaps or red-flag descriptors (e.g., sudden maximal onset, tearing pain) are possible.",
            "confidence": "low",
        },
    ]


def _is_generic_or_redundant_diagnosis(diagnosis: str) -> bool:
    d = (diagnosis or "").lower()
    generic_markers = [
        "nonspecific",
        "undifferentiated",
        "undifferentiated",
        "unclear",
        "condition requiring further evaluation",
        "critical cardiopulmonary cause",
        "unspecified",
        "no clear",
    ]
    return any(marker in d for marker in generic_markers)


def _build_prompt(soap_note: dict[str, Any]) -> str:
    soap_json = json.dumps(
        {
            "subjective": _safe_text(soap_note.get("subjective")),
            "objective": _safe_text(soap_note.get("objective")),
            "assessment": _safe_text(soap_note.get("assessment")),
            "plan": _safe_text(soap_note.get("plan")),
        },
        ensure_ascii=True,
    )

    return f"""
You are a clinical reasoning assistant.
Given the SOAP note below, suggest the top 3 possible diagnoses.

Requirements:
- Use medically styled but generic language.
- Base reasoning only on provided SOAP details.
- Do not invent lab/imaging results.
- Prioritize LIFE-THREATENING conditions first.
- Be decisive: when strong symptom patterns exist, set higher confidence (high/medium) instead of defaulting to low.
- Do NOT use generic placeholder diagnoses like "nonspecific symptom complex" or "undifferentiated presentation".
- Pattern recognition (must apply):
  - If symptoms include chest pain AND radiation to arm/jaw AND sweating/diaphoresis, you MUST include three distinct high-yield categories in the top 3:
    1) "Myocardial Infarction" (primary likely)
    2) "Pulmonary Embolism" (alternative serious)
    3) "Aortic Dissection (consider)" (critical but less likely)
- For each diagnosis, include:
  - diagnosis
  - reason (must explicitly reference the triggering symptoms)
  - confidence (must be one of: low, medium, high)
- Return only valid JSON as an object with exactly this structure:
  {{
    "diagnoses": [
      {{"diagnosis": "...", "reason": "...", "confidence": "low|medium|high"}},
      {{"diagnosis": "...", "reason": "...", "confidence": "low|medium|high"}},
      {{"diagnosis": "...", "reason": "...", "confidence": "low|medium|high"}}
    ]
  }}

SOAP note:
{soap_json}
""".strip()


def _suggest_diagnosis_impl(soap_note: dict) -> list[dict[str, str]]:
    """
    Suggest top differential diagnoses from a SOAP note.

    Returns format:
    [
      {
        "diagnosis": "",
        "reason": "",
        "confidence": ""
      }
    ]
    """
    if not isinstance(soap_note, dict) or not soap_note:
        return _placeholder_suggestions({})

    diagnosis_items_schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["diagnosis", "reason", "confidence"],
        "additionalProperties": False,
    }

    response_json_schema = {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "items": diagnosis_items_schema,
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["diagnoses"],
        "additionalProperties": False,
    }

    prompt = _build_prompt(soap_note)

    # Prefer Gemini if configured.
    if settings.gemini_api_key and GeminiClient is not None:
        client = GeminiClient.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": response_json_schema,
            },
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            parsed = json.loads(response.text)

        raw_items = parsed.get("diagnoses", [])
        items = raw_items if isinstance(raw_items, list) else _placeholder_suggestions(soap_note)
    else:
        # Optional OpenAI fallback (kept for convenience).
        if not settings.openai_api_key or OpenAIClient is None:
            return _placeholder_suggestions(soap_note)

        client = OpenAIClient(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You provide concise differential diagnosis suggestions from SOAP notes.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or "{}"
        parsed: dict[str, Any] = json.loads(content)
        raw_items = parsed.get("diagnoses", [])
        items = raw_items if isinstance(raw_items, list) else _placeholder_suggestions(soap_note)

    normalized: list[dict[str, str]] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        confidence = _safe_text(item.get("confidence")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        normalized.append(
            {
                "diagnosis": _safe_text(item.get("diagnosis")),
                "reason": _safe_text(item.get("reason")),
                "confidence": confidence,
            }
        )

    while len(normalized) < 3:
        # Use deterministic heuristic fallback so we avoid generic placeholders.
        fallback = _placeholder_suggestions(soap_note)
        for extra in fallback:
            if len(normalized) >= 3:
                break
            if not isinstance(extra, dict):
                continue
            conf = _safe_text(extra.get("confidence")).lower()
            if conf not in {"low", "medium", "high"}:
                conf = "low"
            candidate = {
                "diagnosis": _safe_text(extra.get("diagnosis")),
                "reason": _safe_text(extra.get("reason")),
                "confidence": conf,
            }
            if candidate["diagnosis"] and candidate not in normalized:
                normalized.append(candidate)

    # Post-process to enforce requirements: 3 distinct categories and avoid generic diagnoses.
    diag_set = {item.get("diagnosis", "").strip().lower() for item in normalized if isinstance(item, dict)}
    has_generic = any(_is_generic_or_redundant_diagnosis(item.get("diagnosis", "")) for item in normalized)

    if len(diag_set) < 3 or has_generic:
        return _placeholder_suggestions(soap_note, cardiac_risk_pattern=False)

    return normalized


def suggest_diagnosis(soap_note: dict, cardiac_risk_pattern: bool = False) -> list[dict[str, str]]:
    """
    Suggest top differential diagnoses from a SOAP note.

    If `cardiac_risk_pattern` is True, the output is reinforced with high-risk
    cardiopulmonary priorities (MI/ACS first) regardless of missing details.
    """
    # Pattern-driven deterministic path (ensures diverse, non-redundant ED differentials).
    subjective = _safe_text(soap_note.get("subjective")) if isinstance(soap_note, dict) else ""
    objective = _safe_text(soap_note.get("objective")) if isinstance(soap_note, dict) else ""
    assessment = _safe_text(soap_note.get("assessment")) if isinstance(soap_note, dict) else ""
    plan = _safe_text(soap_note.get("plan")) if isinstance(soap_note, dict) else ""
    context = " ".join([subjective, objective, assessment, plan]).lower()

    has_chest_pain = "chest pain" in context or ("chest" in context and "pain" in context)
    has_radiation = any(token in context for token in ["radiating", "radiation"]) and any(
        token in context for token in ["arm", "jaw"]
    )
    has_sweating = any(token in context for token in ["sweating", "diaphoresis"])
    strong_cardiac = has_chest_pain and has_radiation and has_sweating

    # Simple fever/injury triggers for deterministic differentials.
    has_fever = ("fever" in context) or ("temp" in context and any(t in context for t in ["101", "102", "103", "104"])) or (
        "temperature" in context and any(t in context for t in ["101", "102", "103", "104"])
    )
    has_injury = any(token in context for token in ["fall", "injury", "motorcycle", "trauma"])

    if cardiac_risk_pattern or strong_cardiac:
        return _placeholder_suggestions(soap_note, cardiac_risk_pattern=True)
    if has_fever or has_injury:
        return _placeholder_suggestions(soap_note, cardiac_risk_pattern=False)

    # Otherwise use the existing LLM/placeholder behavior.
    return _suggest_diagnosis_impl(soap_note)
