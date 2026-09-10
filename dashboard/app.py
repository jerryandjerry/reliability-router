"""Optional Streamlit dashboard for evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reliability Router", page_icon="🧭", layout="wide")
st.title("Reliability Router Evaluation")
st.caption("Inspect routing accuracy, escalation behavior, reason codes, and per-case failures.")

report_path = Path(st.sidebar.text_input("Report path", "artifacts/eval_report.json"))
if not report_path.exists():
    st.error(f"Report not found: {report_path}. Run `make eval` first.")
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))
predictions = pd.DataFrame(report["predictions"])
if not predictions.empty:
    predictions["reason_codes"] = predictions["reason_codes"].apply(lambda values: ", ".join(values))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Route accuracy", f"{report['route_accuracy']:.1%}")
col2.metric("DEEP precision", f"{report['deep_precision']:.1%}")
col3.metric("DEEP recall", f"{report['deep_recall']:.1%}")
col4.metric("P95 route latency", f"{report['p95_latency_ms']:.2f} ms")

st.subheader("Confusion matrix")
confusion = pd.DataFrame(
    [
        {
            "Actual": "DEEP",
            "Predicted DEEP": report["confusion_matrix"]["deep_as_deep"],
            "Predicted QUICK": report["confusion_matrix"]["deep_as_quick"],
        },
        {
            "Actual": "QUICK",
            "Predicted DEEP": report["confusion_matrix"]["quick_as_deep"],
            "Predicted QUICK": report["confusion_matrix"]["quick_as_quick"],
        },
    ]
).set_index("Actual")
st.dataframe(confusion, use_container_width=True)

st.subheader("Reason-code frequency")
reason_frame = pd.DataFrame(
    [{"reason_code": key, "count": value} for key, value in report["reason_code_counts"].items()]
).sort_values("count", ascending=False)
st.bar_chart(reason_frame.set_index("reason_code"))

st.subheader("Predictions")
failed_only = st.toggle("Show failures only", value=False)
visible = predictions[~predictions["correct"]] if failed_only else predictions
st.dataframe(visible, use_container_width=True, hide_index=True)

st.caption(
    "This dashboard visualizes the included curated smoke benchmark. It is not evidence of production-grade "
    "generalization; add domain-specific and adversarial datasets before deployment."
)
