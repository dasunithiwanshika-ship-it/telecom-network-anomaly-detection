"""
Modular Metric Cards, Telecom KPI Metadata, and Navigation Helpers.
Styled with corporate elegance, typography-first metrics, and minimal ornamentation.
"""

from typing import Dict, Any, Optional
import streamlit as st


TELECOM_KPI_METADATA: Dict[str, Dict[str, str]] = {
    "RSRP": {
        "friendly": "Reference Signal Power (RSRP)",
        "subsystem": "Radio Coverage & Power",
        "unit": "dBm",
        "meaning": "Downlink signal strength received from the serving cell tower (-75 dBm is strong; below -115 dBm indicates cell-edge or coverage loss).",
        "action": "Inspect antenna tilt, sector transmission power, or cell-edge handover thresholds.",
    },
    "UL_SNR": {
        "friendly": "Uplink Signal-to-Noise Ratio (UL SNR)",
        "subsystem": "Radio Signal Quality",
        "unit": "dB",
        "meaning": "Uplink signal clarity relative to background noise and interference (higher dB = cleaner signal).",
        "action": "Check uplink interference floor, external RF jammers, or inter-cell interference.",
    },
    "UL_BLER": {
        "friendly": "Uplink Block Error Rate (UL BLER)",
        "subsystem": "Link Error Rates",
        "unit": "Ratio [0–1]",
        "meaning": "Fraction of uplink data blocks arriving with transmission errors requiring HARQ retransmission.",
        "action": "Investigate physical uplink interference, poor UE power control, or demodulation failures.",
    },
    "DL_BLER": {
        "friendly": "Downlink Block Error Rate (DL BLER)",
        "subsystem": "Link Error Rates",
        "unit": "Ratio [0–1]",
        "meaning": "Fraction of downlink data blocks received with errors. Values persistently above 0.10 (10%) indicate link degradation.",
        "action": "Review CQI reporting accuracy, downlink MIMO channel conditions, and outer-loop link adaptation.",
    },
    "UL_MCS": {
        "friendly": "Uplink Modulation & Coding (UL MCS)",
        "subsystem": "Modulation & Link Adaptation",
        "unit": "Index [0–28]",
        "meaning": "Modulation scheme index assigned to uplink transmissions (low index = fallback to robust QPSK; high index = 64QAM).",
        "action": "Verify whether link adaptation is stepping down modulation due to poor UL SNR.",
    },
    "DL_MCS": {
        "friendly": "Downlink Modulation & Coding (DL MCS)",
        "subsystem": "Modulation & Link Adaptation",
        "unit": "Index [0–28]",
        "meaning": "Modulation scheme index assigned to downlink transmissions based on subscriber CQI feedback.",
        "action": "Check downlink SINR distribution and CQI-to-MCS mapping tables.",
    },
    "UL_NPRB": {
        "friendly": "Uplink Allocated Resource Blocks (UL NPRB)",
        "subsystem": "MAC Resource Scheduling",
        "unit": "PRB Count",
        "meaning": "Number of Physical Resource Blocks granted by the base station MAC scheduler for uplink transmission.",
        "action": "Check MAC scheduler grant starvation, control channel (PDCCH) capacity, or cell congestion.",
    },
    "PRBs_DL_Current": {
        "friendly": "Active Downlink PRB Grant",
        "subsystem": "MAC Resource Scheduling",
        "unit": "PRB Count",
        "meaning": "Instantaneous downlink resource blocks allocated during the scheduling interval.",
        "action": "Monitor sector downlink bandwidth demand and multi-user scheduling fairness.",
    },
    "PRBs_UL_Current": {
        "friendly": "Active Uplink PRB Grant",
        "subsystem": "MAC Resource Scheduling",
        "unit": "PRB Count",
        "meaning": "Instantaneous uplink resource blocks allocated during the active scheduling window.",
        "action": "Review uplink scheduling grants and PUCCH scheduling request handling.",
    },
    "PRB_Utilization_DL": {
        "friendly": "Downlink Spectrum Utilization (DL PRB %)",
        "subsystem": "MAC Resource Scheduling",
        "unit": "%",
        "meaning": "Percentage of total cell downlink spectrum capacity currently occupied.",
        "action": "High values (>85%) indicate sector capacity exhaustion; consider load balancing or carrier aggregation.",
    },
    "PRB_Utilization_UL": {
        "friendly": "Uplink Spectrum Utilization (UL PRB %)",
        "subsystem": "MAC Resource Scheduling",
        "unit": "%",
        "meaning": "Percentage of total cell uplink spectrum capacity currently occupied.",
        "action": "Evaluate uplink traffic load and PRB contention across active users.",
    },
    "TX_Bytes": {
        "friendly": "Uplink Transmitted Payload (TX Bytes)",
        "subsystem": "Traffic & Throughput",
        "unit": "Bytes",
        "meaning": "Volume of payload bytes transmitted in the uplink direction across the observation window.",
        "action": "Compare against UL PRB grants and buffer occupancy to detect throughput throttling.",
    },
    "RX_Bytes": {
        "friendly": "Downlink Received Payload (RX Bytes)",
        "subsystem": "Traffic & Throughput",
        "unit": "Bytes",
        "meaning": "Volume of payload bytes delivered in the downlink direction across the observation window.",
        "action": "Correlate with DL PRB utilization and DL MCS to verify spectral efficiency.",
    },
    "TX_Packets": {
        "friendly": "Uplink Packet Volume (TX Packets)",
        "subsystem": "Traffic & Throughput",
        "unit": "Packets",
        "meaning": "Count of network packets transmitted in the uplink direction during the window.",
        "action": "Check average packet size and correlate with UE bearer types (VoLTE vs best-effort eMBB).",
    },
    "RX_Packets": {
        "friendly": "Downlink Packet Volume (RX Packets)",
        "subsystem": "Traffic & Throughput",
        "unit": "Packets",
        "meaning": "Count of network packets received from the core network for downlink delivery.",
        "action": "Cross-reference with downlink BLER to determine packet drop rates.",
    },
    "Estimated_UL_Buffer": {
        "friendly": "Uplink Queue Buffer Backlog (UL Buffer)",
        "subsystem": "Traffic & Throughput",
        "unit": "Bytes/KB",
        "meaning": "Estimated backlog of subscriber data awaiting uplink transmission in base station buffers.",
        "action": "High buffer backlog indicates uplink scheduling starvation or severe RF path loss.",
    },
}


def get_feature_meta(feature_name: str) -> Dict[str, str]:
    """Extract clean base KPI, statistical moment, and friendly name for any feature column."""
    moments = ["_mean", "_std", "_min", "_max"]
    base_kpi = feature_name
    moment_name = "raw"

    for m in moments:
        if feature_name.endswith(m):
            base_kpi = feature_name[: -len(m)]
            moment_name = m.lstrip("_")
            break

    meta = TELECOM_KPI_METADATA.get(base_kpi, {})
    friendly = meta.get("friendly", base_kpi)
    moment_label = {"mean": "Average", "std": "Variability (Std)", "min": "Minimum", "max": "Peak"}.get(
        moment_name, moment_name
    )

    return {
        "base_kpi": base_kpi,
        "moment": moment_name,
        "friendly_name": f"{friendly} [{moment_label}]",
        "subsystem": meta.get("subsystem", "General Telemetry"),
        "unit": meta.get("unit", ""),
        "meaning": meta.get("meaning", "Cellular telemetry metric."),
        "action": meta.get("action", "Inspect cell radio and scheduling parameters."),
    }


import textwrap


def render_html(html_str: str) -> None:
    """Renders HTML safely in Streamlit without CommonMark 4-space code-block bugs."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge: str = "TELECOM AI") -> None:
    """Renders a clean, executive hero banner without emoji clutter."""
    render_html(
        f"""
        <div class="telecom-hero-banner">
            <div class="telecom-hero-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color: str = "#0EA5E9",
) -> None:
    """Renders a clean, high-contrast metric card."""
    subtext_html = f"<div class='telecom-metric-subtext'>{subtext}</div>" if subtext else ""
    render_html(
        f"""
        <div class="telecom-metric-card">
            <div class="telecom-metric-label">{label}</div>
            <div class="telecom-metric-value" style="color: {color};">{value}</div>
            {subtext_html}
        </div>
        """
    )


def render_noc_status_strip(
    cells_up: int = 30,
    cells_down: int = 0,
    degraded: int = 2,
    active_alerts: int = 4,
    critical_alerts: int = 2,
    open_incidents: int = 3,
) -> None:
    """Renders the top NOC status ribbon with subtle geometric dot indicators."""
    render_html(
        f"""
        <div class="noc-pill-strip">
            <div class="noc-status-pill">
                <span class="status-dot dot-green"></span>
                <div>
                    <div class="noc-pill-label">Devices Online</div>
                    <div class="noc-pill-value" style="color: #10B981;">{cells_up} <span style="font-size: 0.8rem; color: #64748B;">/30</span></div>
                </div>
            </div>
            <div class="noc-status-pill">
                <span class="status-dot dot-red"></span>
                <div>
                    <div class="noc-pill-label">Devices Offline</div>
                    <div class="noc-pill-value" style="color: #64748B;">{cells_down} <span style="font-size: 0.8rem; color: #64748B;">/30</span></div>
                </div>
            </div>
            <div class="noc-status-pill">
                <span class="status-dot dot-amber"></span>
                <div>
                    <div class="noc-pill-label">Degraded</div>
                    <div class="noc-pill-value" style="color: #F59E0B;">{degraded} <span style="font-size: 0.8rem; color: #64748B;">/30</span></div>
                </div>
            </div>
            <div class="noc-status-pill">
                <span class="status-dot dot-amber"></span>
                <div>
                    <div class="noc-pill-label">Active Alerts</div>
                    <div class="noc-pill-value" style="color: #F59E0B;">{active_alerts}</div>
                </div>
            </div>
            <div class="noc-status-pill">
                <span class="status-dot dot-red"></span>
                <div>
                    <div class="noc-pill-label">Critical Alerts</div>
                    <div class="noc-pill-value" style="color: #EF4444;">{critical_alerts}</div>
                </div>
            </div>
            <div class="noc-status-pill">
                <span class="status-dot dot-blue"></span>
                <div>
                    <div class="noc-pill-label">Open Incidents</div>
                    <div class="noc-pill-value" style="color: #0EA5E9;">{open_incidents}</div>
                </div>
            </div>
        </div>
        """
    )


def navigate_to(page_name: str) -> None:
    """Safely updates session navigation via pending_nav and triggers a clean rerun."""
    st.session_state["pending_nav"] = page_name
    st.rerun()


def render_academic_disclaimer() -> None:
    """Render a clean enterprise methodology badge."""
    st.markdown(
        """
        <div style="background: #0B101B; border: 1px solid #1A263D; border-left: 3px solid #0EA5E9;
                    border-radius: 6px; padding: 10px 14px; margin-top: 20px; font-size: 0.8rem; color: #94A3B8;">
            <strong style="color: #F8FAFC;">Enterprise Verification Note:</strong>
            Powered by Random Forest Supervised Ensemble and Isolation Forest Anomaly Detection,
            calibrated across 32,000 cellular transmission windows with 5-fold cross-validation.
        </div>
        """,
        unsafe_allow_html=True,
    )
