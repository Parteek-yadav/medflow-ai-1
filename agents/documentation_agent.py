import json
import re
from typing import Any, Optional

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


SOAP_OUTPUT_SHAPE = {
    "subjective": "",
    "objective": "",
    "assessment": "",
    "plan": "",
}

SOAP_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subjective": {"type": "string"},
        "objective": {"type": "string"},
        "assessment": {"type": "string"},
        "plan": {"type": "string"},
    },
    "required": ["subjective", "objective", "assessment", "plan"],
    "additionalProperties": False,
}


def _placeholder_soap(input_text: str) -> dict[str, str]:
    """
    Fallback SOAP output when an LLM provider isn't configured.

    Note: this intentionally does NOT copy raw conversation verbatim.
    """
    text = input_text or ""
    lowered = text.lower()

    # Basic extractions for concise, structured output.
    vitals_present = any(
        token in lowered
        for token in [
            "bp",
            "blood pressure",
            "hr",
            "heart rate",
            "pulse",
            "rr",
            "respiratory rate",
            "temp",
            "temperature",
            "spo2",
            "o2 sat",
        ]
    )
    labs_present = any(
        token in lowered
        for token in [
            "troponin",
            "wbc",
            "cbc",
            "hemoglobin",
            "creatinine",
            "crp",
            "d-dimer",
            "lactate",
            "bmp",
            "cmp",
            "glucose",
            "urinalysis",
            "culture",
        ]
    )

    bp = re.search(r"(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3}\/\d{2,3})", lowered)
    hr = re.search(r"(?:hr|heart rate|pulse)\s*[:=]?\s*(\d{2,3})", lowered)
    rr = re.search(r"(?:rr|respiratory rate)\s*[:=]?\s*(\d{1,3})", lowered)
    temp = re.search(r"(?:temp|temperature)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*([fc])?\b", lowered)
    spo2 = re.search(r"(?:spo2|o2 sat|o2 saturation)\s*[:=]?\s*(\d{2,3})\s*%?", lowered)

    vitals_parts: list[str] = []
    if bp:
        vitals_parts.append(f"BP {bp.group(1)}")
    if hr:
        vitals_parts.append(f"HR {hr.group(1)}")
    if rr:
        vitals_parts.append(f"RR {rr.group(1)}")
    if temp:
        unit = temp.group(2).upper() if temp.group(2) else ""
        vitals_parts.append(f"Temp {temp.group(1)}{unit}")
    if spo2:
        vitals_parts.append(f"SpO2 {spo2.group(1)}%")

    # Temperature parsing (used to decide whether “fever” is actually present).
    temp_val_f: float | None = None
    temp_match_f = re.search(
        r"\btemp(?:erature)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*([fc])?\b",
        lowered,
    )
    if temp_match_f:
        try:
            temp_val = float(temp_match_f.group(1))
            unit = (temp_match_f.group(2) or "f").lower()
            if unit == "c":
                temp_val_f = temp_val * 9.0 / 5.0 + 32.0
            else:
                temp_val_f = temp_val
        except (TypeError, ValueError):
            temp_val_f = None

    # Subjective symptoms (keyword extraction; no verbatim copying).
    def _first_match(patterns: list[str]) -> str | None:
        for p in patterns:
            m = re.search(p, lowered)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return None

    location = _first_match([r"\b(chest)\b", r"\b(left chest|right chest)\b", r"\b(leg|foot|ankle|knee|arm|hand)\b"])
    duration = _first_match([r"\bfor\s+(\d+\s*(?:minutes|hours|days|weeks))", r"\b(\d+\s*(?:minutes|hours|days|weeks))\b"])
    severity = _first_match([r"\b(severe|mild|moderate)\b"])
    radiation = _first_match(
        [
            r"\bradiat(?:e|ing)\w*\s*(?:to|into)?\s*(?:the\s*)?(left arm|right arm|left leg|right leg|arms|legs|jaw)"
        ]
    )

    def _is_negated(needle: str) -> bool:
        # Simple negation detection for common ED phrasing.
        # Examples: "Denies chest pain", "No chest pain", "denied fever".
        return (
            re.search(rf"\b(denies|denied|no)\s+{re.escape(needle)}\b", lowered) is not None
            or re.search(rf"\b{re.escape(needle)}\s+was\s+denied\b", lowered) is not None
        )

    symptoms: list[str] = []

    # Chest pain (respect negation).
    if "chest pain" in lowered and not _is_negated("chest pain"):
        symptoms.append("chest pain")

    # Fever: rely on explicit “fever” or temperature threshold.
    fever_explicit = "fever" in lowered and not _is_negated("fever")
    fever_by_temp = temp_val_f is not None and temp_val_f >= 100.4
    if fever_explicit or fever_by_temp:
        symptoms.append("fever")

    if "sore throat" in lowered and not _is_negated("sore throat"):
        symptoms.append("sore throat")
    if (
        ("shortness of breath" in lowered or "breathlessness" in lowered or "breathless" in lowered)
        and not (
            _is_negated("shortness of breath")
            or _is_negated("breathlessness")
            or _is_negated("breathless")
        )
    ):
        symptoms.append("breathlessness")

    if ("sweating" in lowered or "sweat" in lowered) and not _is_negated("sweating"):
        symptoms.append("diaphoresis")
    if "nausea" in lowered and not _is_negated("nausea"):
        symptoms.append("nausea")
    if ("body aches" in lowered or "body ache" in lowered or "aches" in lowered) and not _is_negated(
        "body aches"
    ):
        symptoms.append("body aches")
    if ("fall" in lowered or "injury" in lowered or "motorcycle fall" in lowered or "swelling" in lowered) and not _is_negated(
        "injury"
    ):
        symptoms.append("traumatic injury symptoms")

    associated: list[str] = []
    for token in ["diaphoresis", "sweating", "nausea", "shortness of breath", "breathlessness", "cough", "sore throat", "body aches", "swelling", "unable to bear weight"]:
        if token in lowered:
            associated.append(token)

    allergies = "Allergy status not provided."
    if "nkda" in lowered or "no known drug allergies" in lowered or "no known allergies" in lowered:
        allergies = "Allergies: NKDA."
    elif "allerg" in lowered:
        allergies = "Allergies reported (details not normalized)."

    medical_history = allergies

    subjective = "Symptoms: " + (", ".join(symptoms) if symptoms else "not clearly specified") + "."
    if location:
        subjective += f" Location: {location}."
    if duration:
        subjective += f" Duration: {duration}."
    if severity:
        subjective += f" Severity: {severity}."
    if radiation:
        subjective += f" Radiation: {radiation}."
    if associated:
        subjective += " Associated symptoms: " + ", ".join(associated) + "."
    subjective += " Medical history: " + medical_history

    objective_parts: list[str] = []
    if not vitals_parts:
        objective_parts.append("No vital signs recorded")
    else:
        objective_parts.append("Vitals: " + ", ".join(vitals_parts))

    if not labs_present:
        objective_parts.append("No lab data available")
    else:
        objective_parts.append("Lab data available (details not normalized).")

    objective = ". ".join(objective_parts) + "."

    # Assessment + Plan based on extracted (and negation-aware) symptoms.
    has_chest_pain = "chest pain" in symptoms
    has_fever_symptom = "fever" in symptoms
    has_injury_symptom = "traumatic injury symptoms" in symptoms

    if has_chest_pain:
        assessment = "Symptoms concerning for a potentially serious cardiopulmonary process; require prompt clinical evaluation."
        plan = "Obtain ECG and cardiac biomarkers (as appropriate), repeat/monitor vital signs, and consider imaging and additional labs per protocol; seek urgent evaluation."
    elif has_fever_symptom:
        assessment = "Fever pattern concerning for an acute infectious/inflammatory process; evaluate for source and severity."
        plan = "Check vitals, perform focused exam, consider CBC/CMP and relevant cultures or rapid testing as indicated, and monitor for red flags; seek urgent evaluation if unstable or high risk."
    elif has_injury_symptom:
        assessment = "Traumatic injury features concerning for underlying structural injury; assess for complications."
        plan = "Perform focused injury assessment (including neurovascular status), consider imaging such as X-ray/CT depending on findings, provide analgesia and immobilization as appropriate, and seek urgent evaluation if unable to bear weight or worsening."
    else:
        assessment = "Non-specific symptoms requiring clinical correlation and further evaluation."
        plan = "Reassess vitals, obtain targeted history/exam, and order basic labs/imaging as clinically indicated; seek urgent care for severe or worsening symptoms."

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan,
    }


def _build_prompt(input_text: str) -> str:
    return f"""
You are a medical documentation assistant.
Convert the provided clinical conversation into a concise SOAP note.

Critical requirements:
- Do NOT copy raw conversation verbatim into the SOAP output.
- Extract and summarize key clinical information only.
- If a field is not present, explicitly state what is missing using these phrases:
  - Objective missing vitals: "No vital signs recorded"
  - Objective missing labs: "No lab data available"

SOAP formatting requirements:
Subjective (summarize):
- Symptoms: location, duration, severity, radiation (if present)
- Associated symptoms (if present): e.g., sweating, nausea, breathlessness
- Medical history (if present): e.g., allergies, relevant PMH/meds

Objective:
- Summarize only objective information that appears in the conversation
- If vital signs are absent, include exactly: "No vital signs recorded"
- If lab data are absent, include exactly: "No lab data available"

Assessment:
- Provide a brief clinical interpretation in medically styled but generic terms.
  Example format: "Symptoms concerning for cardiac origin" (adapt to the case).

Plan:
- Suggest next steps and appropriate evaluations (generic and safe),
  e.g., "ECG", "relevant labs", "urgent evaluation if concerning features".

Return only valid JSON with exactly these keys:
  "subjective", "objective", "assessment", "plan"

Clinical conversation (for extraction only; do not quote):
\"\"\"{input_text}\"\"\"
""".strip()


def generate_soap_note(input_text: str) -> dict[str, str]:
    """
    Generate a structured SOAP note from clinical conversation text.

    Returns:
        {
          "subjective": "",
          "objective": "",
          "assessment": "",
          "plan": ""
        }
    """
    if not input_text or not input_text.strip():
        return SOAP_OUTPUT_SHAPE.copy()

    prompt = _build_prompt(input_text)

    # Prefer Gemini if configured.
    if settings.gemini_api_key and GeminiClient is not None:
        client = GeminiClient.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SOAP_JSON_SCHEMA,
            },
        )

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            parsed = json.loads(response.text)

        return {
            "subjective": str(parsed.get("subjective", "")),
            "objective": str(parsed.get("objective", "")),
            "assessment": str(parsed.get("assessment", "")),
            "plan": str(parsed.get("plan", "")),
        }

    # Optional: OpenAI fallback (kept for convenience).
    if not settings.openai_api_key or OpenAIClient is None:
        return _placeholder_soap(input_text)
    client = OpenAIClient(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You produce accurate SOAP notes from clinical dialogue.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content or "{}"
    parsed: dict[str, Any] = json.loads(content)

    return {
        "subjective": str(parsed.get("subjective", "")),
        "objective": str(parsed.get("objective", "")),
        "assessment": str(parsed.get("assessment", "")),
        "plan": str(parsed.get("plan", "")),
    }
