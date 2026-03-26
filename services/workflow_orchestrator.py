import logging
from typing import Any
from typing import TypedDict

from agents.clinical_reasoning_agent import suggest_diagnosis
from agents.documentation_agent import generate_soap_note
from agents.safety_agent import safety_check
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    input_text: str
    cardiac_risk_pattern: bool
    soap_note: dict[str, str]
    diagnoses: list[dict[str, str]]
    safety: dict[str, Any]
    risk_level: str
    status: str
    message: str
    final_summary: str
    audit_trail: list[dict[str, str]]


def _build_final_summary(
    status: str,
    soap_note: dict[str, str],
    diagnoses: list[dict[str, str]],
    safety: dict[str, Any],
) -> str:
    top_dx = (diagnoses[0].get("diagnosis") if diagnoses else "") or ""
    risk_level = str(safety.get("risk_level") or "").lower()

    if status == "EMERGENCY" or risk_level == "high":
        dx_lower = top_dx.lower()
        if "myocardial infarction" in dx_lower or "acute coronary" in dx_lower or "acs" in dx_lower:
            return "High-risk chest pain presentation consistent with acute coronary syndrome. Immediate cardiac evaluation required."
        if "pulmonary embolism" in dx_lower:
            return "High-risk presentation concerning pulmonary embolism. Immediate emergency evaluation required."
        if "aortic dissection" in dx_lower:
            return "High-risk presentation concerning aortic dissection. Immediate imaging and emergency evaluation required."
        return "High-risk presentation identified based on symptom pattern. Immediate emergency evaluation required."

    recommended_action = (safety.get("recommended_action") or "").strip()
    if top_dx:
        return f"Presentation consistent with {top_dx}. {recommended_action}".strip()
    return recommended_action or "Clinical summary generated."


def _documentation_node(state: WorkflowState) -> WorkflowState:
    logger.info("Step 1/4: Generating SOAP note")
    soap_note = generate_soap_note(input_text=state.get("input_text", ""))
    audit = state.get("audit_trail", [])
    audit.append({"step": "documentation", "summary": "SOAP note generated"})
    return {"soap_note": soap_note, "audit_trail": audit}


def _diagnosis_node(state: WorkflowState) -> WorkflowState:
    logger.info("Step 2/4: Generating diagnosis suggestions")
    diagnoses = suggest_diagnosis(
        soap_note=state.get("soap_note", {}),
        cardiac_risk_pattern=bool(state.get("cardiac_risk_pattern", False)),
    )
    audit = state.get("audit_trail", [])
    audit.append({"step": "reasoning", "summary": "3 diagnoses identified"})
    return {"diagnoses": diagnoses, "audit_trail": audit}


def _safety_node(state: WorkflowState) -> WorkflowState:
    logger.info("Step 3/4: Running safety checks")
    safety = safety_check(
        soap_note=state.get("soap_note", {}),
        diagnoses=state.get("diagnoses", []),
        cardiac_risk_pattern=bool(state.get("cardiac_risk_pattern", False)),
    )
    risk_level = str(safety.get("risk_level", "low")).lower()
    audit = state.get("audit_trail", [])
    if risk_level == "high":
        audit.append({"step": "safety", "summary": "High risk detected"})
    else:
        audit.append({"step": "safety", "summary": "No high risk detected"})
    return {"safety": safety, "risk_level": risk_level, "audit_trail": audit}


def _decision_node(state: WorkflowState) -> WorkflowState:
    logger.info("Step 4/4: Deciding response by risk level")
    audit = state.get("audit_trail", [])
    if state.get("risk_level") == "high":
        logger.warning("High risk detected; returning alert response")
        audit.append({"step": "workflow", "summary": "Emergency status triggered"})
        safety = dict(state.get("safety", {}) or {})
        safety["alert_message"] = "High-risk clinical output detected. Manual clinician review is required."
        final_summary = _build_final_summary(
            status="EMERGENCY",
            soap_note=state.get("soap_note", {}),
            diagnoses=state.get("diagnoses", []),
            safety=safety,
        )
        return {
            "status": "EMERGENCY",
            "soap_note": state.get("soap_note", {}),
            "diagnoses": state.get("diagnoses", []),
            "safety": safety,
            "final_summary": final_summary,
            "audit_trail": audit,
        }

    logger.info("Workflow completed successfully")
    audit.append({"step": "workflow", "summary": "Emergency status not triggered"})
    final_summary = _build_final_summary(
        status="NORMAL",
        soap_note=state.get("soap_note", {}),
        diagnoses=state.get("diagnoses", []),
        safety=state.get("safety", {}),
    )
    return {
        "status": "NORMAL",
        "soap_note": state.get("soap_note", {}),
        "diagnoses": state.get("diagnoses", []),
        "safety": state.get("safety", {}),
        "final_summary": final_summary,
        "audit_trail": audit,
    }


def _build_workflow_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("documentation", _documentation_node)
    graph.add_node("diagnosis", _diagnosis_node)
    graph.add_node("safety", _safety_node)
    graph.add_node("decision", _decision_node)

    graph.set_entry_point("documentation")
    graph.add_edge("documentation", "diagnosis")
    graph.add_edge("diagnosis", "safety")
    graph.add_edge("safety", "decision")
    graph.add_edge("decision", END)
    return graph.compile()


workflow_graph = _build_workflow_graph()


def run_workflow(input_text: str) -> dict[str, Any]:
    """Core multi-agent workflow controller implemented with LangGraph."""
    logger.info("Workflow started")
    logger.info("Input received")

    lowered = (input_text or "").lower()
    chest_pain = "chest pain" in lowered
    arm_or_jaw = ("arm" in lowered) or ("jaw" in lowered)
    sweating_or_nausea = ("sweating" in lowered) or ("nausea" in lowered) or ("sweat" in lowered)
    cardiac_risk_pattern = chest_pain and arm_or_jaw and sweating_or_nausea

    initial_state: WorkflowState = {
        "input_text": input_text,
        "cardiac_risk_pattern": cardiac_risk_pattern,
        "audit_trail": [],
    }

    final_state = workflow_graph.invoke(initial_state)

    # Always return a consistent structure for downstream consumers.
    return {
        "status": final_state.get("status", "NORMAL"),
        "soap_note": final_state.get("soap_note", {}),
        "diagnoses": final_state.get("diagnoses", []),
        "safety": final_state.get("safety", {}),
        "audit_trail": final_state.get("audit_trail", []),
        "final_summary": final_state.get("final_summary", ""),
    }
