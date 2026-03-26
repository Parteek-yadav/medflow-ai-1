import json
import os
import sys

# Ensure imports work when running as:
#   python utils/run_final_state.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.workflow_orchestrator import workflow_graph


def main() -> None:
    input_text = (
        "Patient presents with chest pain and diaphoresis. "
        "Vitals reported: BP 150/95, HR 110, RR 20, Temp 98.6F, SpO2 95%."
    )

    initial_state = {
        "input_text": input_text,
        "audit_trail": [{"step": "input_received", "output": input_text}],
    }

    final_state = workflow_graph.invoke(initial_state)
    print(json.dumps(final_state, indent=2, default=str))


if __name__ == "__main__":
    main()

