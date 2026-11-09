"""
Modular Metric Cards, Telecom KPI Metadata, and Navigation Helpers for Telecom AI Platform.
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
    "UL_NumberOfPackets": {
        "friendly": "Uplink Packet Count",
        "subsystem": "Traffic & Throughput",
        "unit": "Packets",
        "meaning": "Number of discrete IP packets transmitted upstream during the cell trace.",
        "action": "Check packet size distribution and burst arrival patterns.",
    },
    "DL_NumberOfPackets": {
        "friendly": "Downlink Packet Count",
        "subsystem": "Traffic & Throughput",
        "unit": "Packets",
        "meaning": "Number of discrete IP packets delivered downstream during the cell trace.",
        "action": "Inspect downlink packet scheduling latency and queue drain rates.",
    },
    "Estimated_UL_Buffer": {
        "friendly": "Uplink Queue Buffer Backlog",
        "subsystem": "UE Buffer & Congestion",
        "unit": "Bytes / KB",
        "meaning": "Estimated data volume waiting in the user equipment uplink buffer for transmission grants.",
        "action": "Sustained buffer buildup indicates uplink congestion, grant starvation, or degraded RF retransmissions.",
    },
}

MOMENT_LABELS = {
    "mean": "Average",
    "std": "Fluctuation (Std Dev)",
    "min": "Minimum",
    "max": "Peak (Max)",
}


def get_feature_meta(feature_col: str) -> Dict[str, str]:
    """Return friendly telecom engineering metadata for a 64-feature column name."""
    base_kpi = feature_col
    moment = ""
    for m in ["mean", "std", "min", "max"]:
        if feature_col.endswith(f"_{m}"):
            base_kpi = feature_col[: -(len(m) + 1)]
            moment = m
            break

    meta = TELECOM_KPI_METADATA.get(
        base_kpi,
        {
            "friendly": base_kpi,
            "subsystem": "Cellular Telemetry",
            "unit": "-",
            "meaning": "Cellular network performance indicator.",
            "action": "Inspect sector telemetry logs.",
        },
    )
    moment_label = MOMENT_LABELS.get(moment, moment.upper())
    friendly_full = f"{meta['friendly']} — {moment_label}" if moment_label else meta["friendly"]

    return {
        "feature": feature_col,
        "base_kpi": base_kpi,
        "moment": moment,
        "moment_label": moment_label,
        "friendly_name": friendly_full,
        "subsystem": meta["subsystem"],
        "unit": meta["unit"],
        "meaning": meta["meaning"],
        "action": meta["action"],
    }


def navigate_to(page_name: str):
    """Trigger programmatic navigation to a target sidebar page."""
    st.session_state["pending_nav"] = page_name
    st.rerun()


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


def render_hero_banner(title: str, subtitle: str, badge: Optional[str] = None):
    """Render the standard executive hero banner for any section."""
    badge_html = f'<div class="telecom-hero-badge">{badge}</div>' if badge else ""
    html = f"""
    <div class="telecom-hero-banner">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_academic_disclaimer():
    """Render the mandatory academic boundary disclaimer."""
    html = """
    <div class="telecom-alert-disclaimer">
        <strong>⚠️ Reference Benchmark Notice:</strong>
        This platform operates on public cellular KPI telemetry (<code>AliMaatouk/TelecomTS</code>) and user-uploaded CSV batches.
        It is an independent diagnostic system—not connected to live SLTMobitel production nodes or physical fiber/copper dispatch systems.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_engineering_notice():
    """Render the engineering validation reminder for individual anomaly investigations."""
    html = """
    <div class="telecom-notice-box">
        <strong>ℹ️ Telecom Engineering Guidance:</strong>
        Flagged anomaly alerts indicate statistical KPI departures from normal cellular operating baselines.
        Always cross-check sector alarms, PRB scheduler utilization, and RF interference counters before initiating field maintenance.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
