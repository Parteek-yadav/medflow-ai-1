import json
import os
import sys
import logging

# Ensure imports work when running as:
#   python utils/run_sample_cases.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.workflow_orchestrator import run_workflow

logging.basicConfig(level=logging.ERROR)


SAMPLE_CASES = [
    {
        "name": "Chest Pain Emergency (High Risk)",
        "text": (
            "Patient is a 58-year-old male with sudden central chest pain for 45 minutes, "
            "radiating to left arm and jaw, associated with sweating and nausea. "
            "No known drug allergies reported. Vitals: BP 158/96, HR 108, RR 22, Temp 98.6F, SpO2 95%."
        ),
    },
    {
        "name": "Chest Pain (Non-specific nausea omitted)",
        "text": (
            "Patient is a 58-year-old male with sudden central chest pain for 45 minutes, "
            "radiating to left arm, associated with mild shortness of breath. "
            "No known drug allergies reported. Vitals: BP 158/96, HR 108, RR 22, Temp 98.6F, SpO2 95%."
        ),
    },
    {
        "name": "Fever",
        "text": (
            "Patient is a 24-year-old female with fever, sore throat, and body aches for 2 days. "
            "Denies chest pain or breathing difficulty. Allergy history unclear. "
            "Vitals: BP 110/70, HR 102, RR 18, Temp 101.8F, SpO2 98%."
        ),
    },
    {
        "name": "Injury",
        "text": (
            "Patient is a 31-year-old male after a motorcycle fall with right leg pain and swelling. "
            "Unable to bear weight. No head injury symptoms reported. "
            "Allergies: NKDA. Vitals: BP 124/82, HR 96, RR 20, Temp 99.1F, SpO2 99%."
        ),
    },
]


def main() -> None:
    for index, case in enumerate(SAMPLE_CASES, start=1):
        print("=" * 80)
        print(f"Case {index}: {case['name']}")
        print("-" * 80)
        print("Input:")
        print(case["text"])
        print("-" * 80)

        result = run_workflow(case["text"])

        status = result.get("status")
        soap_note = result.get("soap_note") or {}
        diagnoses = result.get("diagnoses") or []
        safety = result.get("safety") or {}

        risk_level = str(safety.get("risk_level", "")).lower()
        is_high = risk_level == "high" or str(status).upper() == "EMERGENCY"

        if is_high:
            print("!!! HIGH RISK DETECTED !!!")
        else:
            print("No high-risk pattern detected.")

        print(f"Status: {status}")

        print("\nSOAP Note:")
        print(json.dumps(soap_note, indent=2))

        print("\nDiagnoses:")
        print(json.dumps(diagnoses, indent=2))

        print("\nSafety:")
        print(json.dumps(safety, indent=2))
        print()


if __name__ == "__main__":
    main()
