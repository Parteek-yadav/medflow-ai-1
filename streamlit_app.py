import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/api/process"

st.set_page_config(page_title="Clinical Multi-Agent System", layout="centered")
st.title("Clinical Multi-Agent Analysis")
st.caption("Enter a clinical case conversation and run analysis.")

input_text = st.text_area("Clinical Case Text", height=220, placeholder="Paste clinical conversation here...")

if st.button("Run Analysis", type="primary"):
    if not input_text.strip():
        st.warning("Please enter clinical case text before running analysis.")
    else:
        with st.spinner("Running workflow..."):
            try:
                response = requests.post(API_URL, json={"text": input_text}, timeout=120)
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not reach backend API: {exc}")
                st.stop()

        status = result.get("status", "NORMAL")
        if status == "EMERGENCY":
            st.error("🚨 HIGH RISK: Immediate clinical attention required")

        final_summary = result.get("final_summary", "")
        if final_summary:
            if status == "EMERGENCY":
                st.warning(final_summary)
            else:
                st.info(final_summary)

        st.subheader("SOAP Note")
        soap_note = result.get("soap_note") or {}
        st.markdown(f"**Subjective:** {soap_note.get('subjective', '')}")
        st.markdown(f"**Objective:** {soap_note.get('objective', '')}")
        st.markdown(f"**Assessment:** {soap_note.get('assessment', '')}")
        st.markdown(f"**Plan:** {soap_note.get('plan', '')}")

        st.subheader("Diagnoses")
        diagnoses = result.get("diagnoses") or []
        if diagnoses:
            for idx, item in enumerate(diagnoses, start=1):
                diagnosis = item.get("diagnosis", "")
                reason = item.get("reason", "")
                confidence = item.get("confidence", "")
                st.markdown(f"**{idx}. {diagnosis}**  \nReason: {reason}  \nConfidence: {confidence}")
        else:
            st.info("No diagnosis suggestions returned.")

        st.subheader("Safety")
        safety = result.get("safety") or {}
        risk_level = safety.get("risk_level", "unknown")
        if status == "EMERGENCY" or str(risk_level).lower() == "high":
            st.error(f"**Risk Level:** {risk_level}")
        else:
            st.markdown(f"**Risk Level:** {risk_level}")

        issues = safety.get("issues") or []
        warnings = safety.get("warnings") or []

        recommended_action = safety.get("recommended_action") or ""
        urgency_level = safety.get("urgency_level") or ""
        if recommended_action:
            if status == "EMERGENCY" or str(risk_level).lower() == "high":
                st.error(f"**Recommended Action:** {recommended_action}")
            else:
                st.markdown(f"**Recommended Action:** {recommended_action}")
        if urgency_level:
            st.markdown(f"**Urgency Level:** {urgency_level}")

        if issues:
            st.markdown("**Issues**")
            for issue in issues:
                st.write(f"- {issue}")

        if warnings:
            st.markdown("**Warnings**")
            for warning in warnings:
                st.write(f"- {warning}")

        if not issues and not warnings:
            st.success("No safety issues or warnings were flagged.")
