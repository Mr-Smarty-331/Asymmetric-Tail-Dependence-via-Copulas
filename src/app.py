"""Interactive Risk Methodologies Group (RMG) Autonomous Model Validation & Copula Analytics Dashboard."""

import io
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from fpdf import FPDF
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtester import BacktestResults, run_rolling_backtest
from src.data_loader import fetch_market_data
from src.risk_reporter import (
    MockLLMClient,
    ModelValidationReport,
    RealLLMClient,
    generate_validation_report,
)


def load_or_run_backtest(
    model_choice: str = "t",
    alpha: float = 0.99,
    window_size: int = 252,
) -> Tuple[BacktestResults, pd.DataFrame]:
    """Load cached backtest results or compute on demand."""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    returns_df = fetch_market_data(
        tickers=["SPY", "QQQ", "TLT"],
        cache_dir="tests/fixtures",
        use_cache=True,
    )

    if model_choice == "t":
        cache_file = cache_dir / "backtest_t_copula_2020_2022.json"
        copula_family = "t"
    else:
        cache_file = cache_dir / "backtest_gaussian_2020_2022.json"
        copula_family = "gaussian_var"

    if cache_file.exists() and alpha == 0.99 and window_size == 252:
        with open(cache_file, "r") as f:
            results = BacktestResults.from_dict(json.load(f))
    else:
        results = run_rolling_backtest(
            returns_df=returns_df,
            window_size=window_size,
            alpha=alpha,
            copula_family=copula_family,
            n_mc_samples=2000,
            cache_path=cache_file if (alpha == 0.99 and window_size == 252) else None,
            use_cache=True,
        )

    return results, returns_df


def build_risk_chart(results: BacktestResults) -> go.Figure:
    """Build interactive Plotly chart with Realized Losses, VaR, ES, and Breach Markers."""
    fig = go.Figure()

    # 1. Realized daily loss
    fig.add_trace(
        go.Scatter(
            x=results.dates,
            y=results.portfolio_losses * 100.0,
            mode="lines",
            name="Realized Daily Loss (%)",
            line=dict(color="#64748b", width=1.2),
            opacity=0.7,
        )
    )

    # 2. VaR Forecast Band
    fig.add_trace(
        go.Scatter(
            x=results.dates,
            y=results.var_forecasts * 100.0,
            mode="lines",
            name=f"{results.model_name} 99% VaR (%)",
            line=dict(color="#f59e0b", width=2.0),
        )
    )

    # 3. ES Forecast Band
    fig.add_trace(
        go.Scatter(
            x=results.dates,
            y=results.es_forecasts * 100.0,
            mode="lines",
            name=f"{results.model_name} 99% ES (%)",
            line=dict(color="#ef4444", width=1.5, dash="dash"),
        )
    )

    # 4. Exception Breach Markers
    breach_indices = np.where(results.exceptions > 0)[0]
    if len(breach_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=results.dates[breach_indices],
                y=results.portfolio_losses[breach_indices] * 100.0,
                mode="markers",
                name=f"VaR Breaches ({len(breach_indices)})",
                marker=dict(color="#dc2626", size=9, symbol="circle-open-dot", line=dict(width=2)),
            )
        )

    fig.update_layout(
        title=f"<b>{results.model_name} Backtest Performance (2020–2022)</b>",
        xaxis_title="Date",
        yaxis_title="Portfolio Loss / Risk (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def generate_pdf_report(report: ModelValidationReport) -> bytes:
    """Generate formal PDF document for Model Validation Report using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)

    # Header
    pdf.cell(0, 10, "RISK METHODOLOGIES GROUP (RMG) - MODEL VALIDATION REPORT", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, f"Model: {report.model_name} | Basel Zone: {report.traffic_light_status}", align="C")
    pdf.ln(10)

    # Status summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Sign-off Status: {report.sign_off_status}")
    pdf.ln(8)
    pdf.cell(0, 8, f"RNIV Capital Add-on: {report.rniv_capital_addon_bps:.1f} bps")
    pdf.ln(10)

    # Key Quantitative Metrics Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 7, "Metric", border=1)
    pdf.cell(60, 7, "Model Value", border=1)
    pdf.cell(60, 7, "Benchmark / Expected", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Observations", str(report.n_observations), "250+"),
        ("VaR Exceptions", str(report.n_exceptions), f"~{report.expected_exceptions:.1f} (1%)"),
        ("Exception Rate", f"{report.exception_rate_pct:.2f}%", "1.00%"),
        ("Kupiec POF p-value", f"{report.kupiec_pvalue:.4f}", "> 0.05"),
        ("Christoffersen p-value", f"{report.christoffersen_pvalue:.4f}", "> 0.05"),
        ("Traffic Light Zone", report.traffic_light_status, "Green"),
    ]
    for m, v, exp in metrics:
        pdf.cell(60, 6, m, border=1)
        pdf.cell(60, 6, v, border=1)
        pdf.cell(60, 6, exp, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Executive Summary")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, report.executive_summary)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Tail Risk & Dependence Assessment")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, report.tail_risk_assessment)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Regulatory & Capital Recommendation")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, report.regulatory_recommendation)

    return bytes(pdf.output())


def render_dashboard():
    """Main dashboard rendering interface."""
    st.set_page_config(
        page_title="RMG Copula Risk & Autonomous Validation",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS Styling
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; color: #f8fafc; }
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .badge-green { background-color: #065f46; color: #34d399; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .badge-amber { background-color: #78350f; color: #fbbf24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .badge-red { background-color: #7f1d1d; color: #f87171; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ Institutional Risk Methodologies Group (RMG)")
    st.subheader("Asymmetric Tail Dependence via Copulas vs. Gaussian Benchmark & Autonomous Validation")

    # Sidebar parameters
    st.sidebar.header("⚙️ Configuration")
    model_type = st.sidebar.selectbox(
        "Risk Methodology",
        options=["Student-t Copula (Asymmetric Tail Risk)", "Gaussian Variance-Covariance (Benchmark)"],
        index=0,
    )
    copula_code = "t" if "Student-t" in model_type else "gaussian"

    alpha = st.sidebar.slider("Confidence Level (VaR / ES)", min_value=0.90, max_value=0.999, value=0.99, step=0.01)
    window_size = st.sidebar.selectbox("Rolling Lookback Window (Days)", options=[252, 500], index=0)

    # Load results
    results, returns_df = load_or_run_backtest(
        model_choice=copula_code,
        alpha=alpha,
        window_size=window_size,
    )

    # Scorecard KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    zone_color = (
        "badge-green" if results.traffic_light == "Green" else ("badge-amber" if results.traffic_light == "Amber" else "badge-red")
    )
    c1.markdown(
        f"<div class='metric-card'><b>Basel Zone</b><br><span class='{zone_color}'>{results.traffic_light.upper()}</span></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='metric-card'><b>Exceptions</b><br><h3>{results.n_exceptions} / {results.n_obs} ({results.exception_rate*100:.2f}%)</h3></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='metric-card'><b>Kupiec p-value</b><br><h3>{results.kupiec_pvalue:.4f}</h3></div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='metric-card'><b>Christoffersen p-val</b><br><h3>{results.christoffersen_pvalue:.4f}</h3></div>",
        unsafe_allow_html=True,
    )
    rniv_bps = (
        0.0 if results.traffic_light == "Green" else (25.0 if results.traffic_light == "Amber" else 100.0)
    )
    if results.christoffersen_pvalue < 0.05 and results.n_exceptions > 1:
        rniv_bps += 20.0
    c5.markdown(
        f"<div class='metric-card'><b>RNIV Capital Add-on</b><br><h3>{rniv_bps:.1f} bps</h3></div>",
        unsafe_allow_html=True,
    )

    # Plotly interactive chart
    fig = build_risk_chart(results)
    st.plotly_chart(fig, use_container_width=True)

    # Autonomous Model Validation Section
    st.markdown("---")
    st.header("🤖 Autonomous Model Validation Reporting")

    col_btn, col_info = st.columns([1, 3])
    use_live_llm = st.sidebar.checkbox("Use Live OpenAI LLM (if API key set)", value=False)

    if "validation_report" not in st.session_state or st.sidebar.button("Re-run Model Audit"):
        client = RealLLMClient() if use_live_llm else MockLLMClient()
        st.session_state["validation_report"] = generate_validation_report(
            results=results,
            model_name=results.model_name,
            returns_df=returns_df,
            llm_client=client,
        )

    report: ModelValidationReport = st.session_state["validation_report"]

    st.markdown(
        f"**Governance Decision:** `{report.sign_off_status}` | **RNIV Recommendation:** `{report.rniv_capital_addon_bps:.1f} bps`"
    )

    t1, t2, t3 = st.tabs(["📋 Executive Summary", "📉 Tail Risk & Asymmetry Analysis", "📜 Regulatory Recommendations"])
    with t1:
        st.write(report.executive_summary)
    with t2:
        st.write(report.tail_risk_assessment)
    with t3:
        st.write(report.regulatory_recommendation)

    # Downloads
    d1, d2, d3 = st.columns(3)
    # 1. Markdown download
    md_content = f"""# MODEL VALIDATION REPORT: {report.model_name}
**Status:** {report.sign_off_status} | **Basel Zone:** {report.traffic_light_status} | **RNIV Add-on:** {report.rniv_capital_addon_bps:.1f} bps

## Executive Summary
{report.executive_summary}

## Tail Risk Assessment
{report.tail_risk_assessment}

## Regulatory Recommendation
{report.regulatory_recommendation}

## Quantitative Facts
- Observations: {report.n_observations}
- Exceptions: {report.n_exceptions} ({report.exception_rate_pct:.2f}%)
- Kupiec p-value: {report.kupiec_pvalue:.4f}
- Christoffersen p-value: {report.christoffersen_pvalue:.4f}
"""
    d1.download_button(
        "📥 Download Markdown Report",
        data=md_content,
        file_name=f"model_validation_{copula_code}.md",
        mime="text/markdown",
    )

    # 2. JSON Payload download
    d2.download_button(
        "📥 Download JSON Payload",
        data=json.dumps(report.quantitative_payload, indent=2),
        file_name=f"validation_payload_{copula_code}.json",
        mime="application/json",
    )

    # 3. PDF Report download
    pdf_bytes = generate_pdf_report(report)
    d3.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"model_validation_{copula_code}.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    render_dashboard()
