from typing import Any
import re


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def safety_check(
    soap_note: dict,
    diagnoses: list,
    cardiac_risk_pattern: bool = False,
) -> dict[str, Any]:
    """
    Run lightweight safety checks on SOAP note and diagnosis suggestions.

    Returns:
    {
      "risk_level": "low/medium/high",
      "issues": [],
      "warnings": []
    }
    """
    issues: list[str] = []
    warnings: list[str] = []

    if not isinstance(soap_note, dict):
        soap_note = {}
    if not isinstance(diagnoses, list):
        diagnoses = []

    subjective = _text(soap_note.get("subjective"))
    objective = _text(soap_note.get("objective"))
    assessment = _text(soap_note.get("assessment"))
    plan = _text(soap_note.get("plan"))

    combined = " ".join([subjective, objective, assessment, plan]).lower()

    # --- High-risk override (rule-based pattern detection) ---
    # Rule: chest pain + radiation (arm/jaw) + (sweating OR nausea) => high risk.
    has_chest_pain = "chest pain" in combined or ("chest" in combined and "pain" in combined)
    has_radiation = ("radiat" in combined) and any(token in combined for token in ["arm", "jaw"])
    has_sweating_or_nausea = any(token in combined for token in ["sweating", "sweat", "diaphoresis", "nausea"])

    critical_alert = False
    urgency_level = "low"  # low/medium/high/immediate
    recommended_action = ""
    if cardiac_risk_pattern:
        critical_alert = True
        urgency_level = "immediate"
        recommended_action = "Immediate ECG, cardiac monitoring, and emergency department admission required"
    elif has_chest_pain and has_radiation and has_sweating_or_nausea:
        critical_alert = True
        urgency_level = "immediate"
        recommended_action = "Immediate ECG, cardiac monitoring, and emergency department admission required"

    # 1) Missing important information
    if not _contains_any(combined, ["allergy", "allergies", "nka", "nkda"]):
        issues.append("Allergy status is missing or unclear.")
    if not _contains_any(
        combined,
        ["vital", "bp", "blood pressure", "heart rate", "pulse", "temperature", "spo2", "respiratory rate"],
    ):
        issues.append("Vital signs are missing or incomplete.")
    if not objective:
        warnings.append("Objective section is empty; clinical findings are limited.")

    # 2) Risky suggestions in plan or diagnoses
    risky_terms = [
        "high-dose",
        "double dose",
        "opioid",
        "benzodiazepine",
        "warfarin",
        "stop immediately",
        "discontinue all",
    ]
    if _contains_any(plan, risky_terms):
        issues.append("Plan contains potentially high-risk treatment language.")

    for item in diagnoses:
        if not isinstance(item, dict):
            continue
        diagnosis_text = _text(item.get("diagnosis"))
        reason_text = _text(item.get("reason"))
        confidence = _text(item.get("confidence")).lower()
        joined = f"{diagnosis_text} {reason_text}"

        if _contains_any(joined, ["stroke", "sepsis", "myocardial infarction", "pulmonary embolism"]) and confidence == "low":
            warnings.append(
                "A potentially high-acuity diagnosis has low confidence; urgent exclusion may still be necessary."
            )
        if confidence not in {"low", "medium", "high"}:
            warnings.append(f"Diagnosis confidence is not standardized for '{diagnosis_text}'.")

    # 3) Inconsistencies across SOAP sections
    if _contains_any(subjective, ["denies fever", "no fever"]) and _contains_any(
        combined, ["febrile", "fever present", "temperature 39", "temperature 38"]
    ):
        issues.append("Possible inconsistency: fever denied in history but documented elsewhere.")

    if not assessment and diagnoses:
        warnings.append("Diagnoses were suggested despite an empty assessment section.")

    # Risk scoring
    if critical_alert:
        risk_level = "high"
    elif len(issues) >= 3:
        risk_level = "high"
    elif len(issues) >= 1 or len(warnings) >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Set missing recommended_action/urgency based on final risk.
    if not recommended_action:
        if risk_level == "high":
            urgency_level = "immediate"
            recommended_action = "Immediate ECG and emergency evaluation required"
        elif risk_level == "medium":
            urgency_level = "high"
            recommended_action = "Prompt clinical evaluation recommended (consider ECG/labs as clinically indicated)"
        else:
            urgency_level = "low"
            recommended_action = "Routine clinical follow-up recommended (seek care if symptoms worsen)"

    # Ensure contract: if risk_level is high, treat it as a critical alert.
    if risk_level == "high":
        critical_alert = True
        urgency_level = "immediate"
        recommended_action = (
            "Immediate ECG, cardiac monitoring, and emergency department admission required"
        )

    return {
        "risk_level": risk_level,
        "critical_alert": critical_alert,
        "urgency_level": urgency_level,
        "issues": issues,
        "warnings": warnings,
        "recommended_action": recommended_action,
    }
