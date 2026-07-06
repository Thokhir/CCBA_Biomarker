"""Plotly figures for live, per-patient content (probability gauge, SHAP
waterfall, risk-score placement) - distinct from the many pre-rendered
matplotlib PNGs from Modules 8/10/11/12, which are displayed as static
images via static_gallery.py rather than recreated interactively.
"""
import plotly.graph_objects as go

PRIMARY = "#0F6E68"
ALERT = "#8C1D18"
IMPUTED_COLOR = "#B8860B"


def probability_gauge(tumor_probability: float, predicted_label: int) -> go.Figure:
    label_text = "Tumor" if predicted_label == 1 else "Normal"
    color = ALERT if predicted_label == 1 else PRIMARY

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tumor_probability * 100,
        number={"suffix": "%"},
        title={"text": f"Tumor Probability - Predicted: {label_text}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#EAF2F1"},
                {"range": [50, 100], "color": "#FBEAEA"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def shap_waterfall(contributions, base_value: float, top_n: int = 15) -> go.Figure:
    top = contributions.head(top_n).iloc[::-1]
    colors = [IMPUTED_COLOR if imputed else (ALERT if v > 0 else PRIMARY)
              for v, imputed in zip(top["shap_value"], top["is_imputed"])]

    fig = go.Figure(go.Bar(
        x=top["shap_value"], y=top["gene_name"], orientation="h", marker_color=colors,
        text=[f"{v:+.3f}{' (imputed)' if imp else ''}" for v, imp in zip(top["shap_value"], top["is_imputed"])],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Top {len(top)} Gene Contributions (base value = {base_value:.3f})",
        xaxis_title="SHAP value (+ toward Tumor, - toward Normal)",
        height=max(400, 28 * len(top)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def risk_placement_histogram(cohort_risk_scores, patient_risk_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cohort_risk_scores, marker_color="#B7CFCC", name="TCGA cohort (n=34)"))
    fig.add_vline(x=patient_risk_score, line_color=ALERT, line_width=3,
                  annotation_text="This patient", annotation_position="top")
    fig.update_layout(
        title="Risk Score Placement vs. TCGA-CHOL Cohort",
        xaxis_title="Composite risk score", yaxis_title="Patients",
        height=350, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
