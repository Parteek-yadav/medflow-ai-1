from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.config import settings
from services.workflow_orchestrator import run_workflow

router = APIRouter()


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Clinical conversation text input")


class DiagnosisItem(BaseModel):
    diagnosis: str
    reason: str
    confidence: str


class SafetyResult(BaseModel):
    risk_level: str
    critical_alert: bool
    issues: list[str]
    warnings: list[str]
    recommended_action: str
    urgency_level: str
    alert_message: str | None = None


class AuditTrailStep(BaseModel):
    step: str
    summary: str


class ProcessResponse(BaseModel):
    status: Literal["EMERGENCY", "NORMAL"]
    soap_note: dict[str, str]
    diagnoses: list[DiagnosisItem]
    safety: SafetyResult | dict[str, Any]
    final_summary: str
    audit_trail: list[AuditTrailStep] = Field(default_factory=list)


@router.get("/health", tags=["test"])
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "service": "multi-agent-backend",
        "api_key_configured": bool(settings.openai_api_key),
    }


@router.post("/process", response_model=ProcessResponse, tags=["workflow"])
def process_clinical_text(payload: ProcessRequest) -> ProcessResponse:
    result = run_workflow(input_text=payload.text)
    return ProcessResponse(**result)
