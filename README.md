# Clinical Multi-Agent AI Backend

## Project Overview
This project is a modular Python backend for clinical text analysis using a multi-agent architecture.  
Given a clinical conversation, the system:
- generates a SOAP note,
- suggests likely diagnoses,
- runs safety checks,
- returns structured output with an audit trail.

The backend is built with FastAPI and includes a minimal Streamlit UI for quick manual testing.

## Problem Statement
Clinical documentation and initial reasoning are often time-consuming in emergency and high-throughput care settings. Teams need a fast way to convert unstructured conversations into consistent, structured clinical summaries while surfacing safety risks early.

This project addresses that need by automating first-pass documentation and differential support, while keeping a safety layer and auditability in the loop.

## Architecture (Multi-Agent System)
The system uses focused agents coordinated by a workflow orchestrator:

- **Documentation Agent** (`agents/documentation_agent.py`)
  - Converts clinical conversation text into SOAP format:
    - Subjective
    - Objective
    - Assessment
    - Plan
- **Clinical Reasoning Agent** (`agents/clinical_reasoning_agent.py`)
  - Suggests top 3 possible diagnoses
  - Adds reasoning and confidence (`low`, `medium`, `high`)
- **Safety Agent** (`agents/safety_agent.py`)
  - Checks for missing critical information (for example: allergies, vitals)
  - Flags risky suggestions
  - Detects inconsistencies
- **Workflow Orchestrator** (`services/workflow_orchestrator.py`)
  - Runs the full pipeline in sequence
  - Applies decision logic (high-risk alert vs full result)
  - Produces an audit trail of step outputs

### High-level flow
1. Input text received  
2. SOAP generated  
3. Diagnoses suggested  
4. Safety checks applied  
5. Result returned (`alert` or `ok`)

{additional way to run }
1. open two split terminal
2. in terminal 1 RUN - cd "C:\Users\parte\OneDrive\Desktop\ET hackathon"
py -m pip install -r requirements.txt

Then in again in terminal 1 RUN - py -m uvicorn main:app --reload

Then without closing terminal 1 , in terminal 2 RUN - py -m streamlit run streamlit_app.py

## Setup Instructions
### 1) Clone or open the project
```bash
cd "C:\Users\parte\OneDrive\Desktop\ET hackathon"
```

### 2) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure environment variables
Create a `.env` file in the project root (or copy from `.env.example`):

```env
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-flash-latest

# Optional fallback keys
OPENAI_API_KEY=your_openai_key_here_optional
ANTHROPIC_API_KEY=your_anthropic_key_here_optional
```

If `GEMINI_API_KEY` is not set, agents will use placeholder fallback behavior for local/dev usage.

## How to Run API
Start FastAPI with Uvicorn:

```bash
uvicorn main:app --reload
```

API base URL:
- `http://127.0.0.1:8000`

Useful endpoints:
- `GET /api/health`
- `POST /api/process`
- Swagger docs: `http://127.0.0.1:8000/docs`



## Optional: Run Streamlit UI
Start backend first, then run:

```bash
streamlit run streamlit_app.py
```
Final improvements and UI polishing

## Optional: Run Sample Emergency Cases
```bash
python utils/run_sample_cases.py
```

Includes:
- chest pain
- fever
- injury

## Impact Explanation
This system is designed to improve clinical workflow efficiency and consistency:

- **Documentation time reduction:** Converts unstructured conversation to SOAP in seconds.
- **Faster triage support:** Produces immediate differential suggestions with confidence labels.
- **Safety visibility:** Highlights missing data and potential risk patterns early.
- **Traceability:** Audit trail captures each workflow step for review and quality assurance.

In practical terms, it can reduce repetitive documentation burden, support faster initial decision-making, and standardize outputs across cases while preserving human clinical oversight.
