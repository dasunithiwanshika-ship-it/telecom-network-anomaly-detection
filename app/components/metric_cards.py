"""
Modular Metric Cards and Statistical Summaries for Telecom AI Platform.
"""

from typing import Optional
import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    value_color: str = "#14B8A6",
):
    """Render a single styled dark NOC-style metric card."""
    subtext_html = (
        f'<div class="telecom-metric-subtext">{subtext}</div>' if subtext else ""
    )
    html = f"""
    <div class="telecom-metric-card">
        <div class="telecom-metric-label">{label}</div>
        <div class="telecom-metric-value" style="color: {value_color};">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str):
    """Render the standard executive hero banner for any section."""
    html = f"""
    <div class="telecom-hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_academic_disclaimer():
    """Render the mandatory academic boundary disclaimer."""
    html = """
    <div class="telecom-alert-disclaimer">
        <strong>⚠️ Independent Academic Research Prototype:</strong><br>
        This platform is evaluated exclusively on the public <code>AliMaatouk/TelecomTS</code> benchmark dataset.
        The data is <strong>NOT</strong> SLTMobitel operational network telemetry, does not represent Sri Lankan
        telecommunications infrastructure, and does not predict physical field dispatches or hardware breakdowns.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_engineering_notice():
    """Render the engineering validation reminder for individual anomaly investigations."""
    html = """
    <div class="telecom-notice-box">
        <strong>ℹ️ Engineering Validation Notice:</strong><br>
        AI-generated anomaly scores represent statistical deviations in cell-level telemetry.
        They require engineering verification and do not independently prove physical hardware failures or dispatch requirements.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
