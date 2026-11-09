"""
Page 02: NOC Network Overview & Operational Intelligence.
Refined enterprise interface featuring radial health gauges, status ribbon,
predictive risk forecasts, and an interactive Radio Access Network topology inspector.
"""

from typing import Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_noc_status_strip,
    render_html,
    navigate_to,
)
from ..components.charts import (
    create_radial_gauge,
    create_kpi_trend_splines,
    create_topology_network_graph,
    create_donut_class_distribution,
)
from src.data_loader import load_processed_features


NODE_TELEMETRY_DATA = {
    "cell-macro-02": {
        "title": "cell-macro-02 (eNodeB Sector 2)",
        "ip": "10.14.2.12",
        "role": "eNodeB Macro Sector",
        "switch": "Agg-Switch-North",
        "prb": "78%",
        "bler": "12.4% (Threshold: 8.0%)",
        "rsrp": "-104 dBm",
        "snr": "8.5 dB",
        "status": "DEGRADED",
        "status_class": "pill-high",
        "note": "High downlink block error rate detected. Review link adaptation MCS and MIMO channel fading.",
    },
    "cell-macro-04": {
        "title": "cell-macro-04 (eNodeB Sector 3)",
        "ip": "10.14.2.14",
        "role": "eNodeB Macro Sector",
        "switch": "Agg-Switch-South",
        "prb": "94%",
        "bler": "4.2%",
        "rsrp": "-96 dBm",
        "snr": "11.2 dB",
        "status": "CRITICAL",
        "status_class": "pill-critical",
        "note": "Uplink queue buffer backlog stall. PRB capacity near 100% saturation. Immediate load balancing required.",
    },
    "cell-macro-01": {
        "title": "cell-macro-01 (eNodeB Sector 1)",
        "ip": "10.14.2.11",
        "role": "eNodeB Macro Sector",
        "switch": "Agg-Switch-North",
        "prb": "42%",
        "bler": "1.1%",
        "rsrp": "-88 dBm",
        "snr": "18.4 dB",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "Operating nominally within baseline statistical boundaries.",
    },
    "cell-small-03": {
        "title": "cell-small-03 (gNodeB Hotspot)",
        "ip": "10.14.3.15",
        "role": "gNodeB Small Cell",
        "switch": "Agg-Switch-North",
        "prb": "35%",
        "bler": "0.9%",
        "rsrp": "-82 dBm",
        "snr": "21.0 dB",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "Operating nominally within baseline statistical boundaries.",
    },
    "cell-small-05": {
        "title": "cell-small-05 (gNodeB Micro Cell)",
        "ip": "10.14.3.17",
        "role": "gNodeB Small Cell",
        "switch": "Agg-Switch-South",
        "prb": "18%",
        "bler": "0.6%",
        "rsrp": "-79 dBm",
        "snr": "23.5 dB",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "Operating nominally within baseline statistical boundaries.",
    },
    "cell-macro-06": {
        "title": "cell-macro-06 (eNodeB Sector 4)",
        "ip": "10.14.2.16",
        "role": "eNodeB Macro Sector",
        "switch": "Agg-Switch-South",
        "prb": "45%",
        "bler": "1.4%",
        "rsrp": "-91 dBm",
        "snr": "16.8 dB",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "Operating nominally within baseline statistical boundaries.",
    },
    "agg-switch-01": {
        "title": "Agg-Switch-North (Backhaul Aggregation)",
        "ip": "10.14.1.2",
        "role": "Aggregation Switch",
        "switch": "Core UPF",
        "prb": "28% BW",
        "bler": "0.0%",
        "rsrp": "N/A",
        "snr": "N/A",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "10 Gbps optical fiber backhaul link nominal.",
    },
    "core-upf-01": {
        "title": "Core UPF (5G User Plane Function)",
        "ip": "10.14.0.1",
        "role": "5G Core Gateway",
        "switch": "Edge Fabric",
        "prb": "31% CPU",
        "bler": "0.0%",
        "rsrp": "N/A",
        "snr": "N/A",
        "status": "OPTIMAL",
        "status_class": "pill-normal",
        "note": "Core data plane latency 2.1ms with zero packet drops.",
    },
}


@st.cache_data
def _get_baseline_dataset():
    try:
        return load_processed_features()
    except Exception:
        return None, None


def render(pipeline: Any = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the executive NOC Overview screen."""
    now_str = datetime.now().strftime("%I:%M:%S %p")
    render_html(
        f"""
        <div class="noc-topbar">
            <div>
                <span style="font-size: 0.7rem; font-weight: 700; color: #38BDF8; letter-spacing: 1px; text-transform: uppercase;">
                    OPERATIONAL INTELLIGENCE ENGINE
                </span>
                <div style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em; margin-top: 2px;">
                    NOC Overview
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 4px; padding: 5px 12px; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #10B981; font-size: 0.7rem;">●</span>
                    <span style="font-size: 0.74rem; font-weight: 700; color: #10B981;">LIVE TELEMETRY {now_str}</span>
                </div>
            </div>
        </div>
        """
    )

    # 1. Top Status Ribbon
    render_noc_status_strip(
        cells_up=30,
        cells_down=0,
        degraded=2,
        active_alerts=4,
        critical_alerts=2,
        open_incidents=3,
    )

    # 2. Four Radial Gauges
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        fig_g1 = create_radial_gauge(100.0, "NETWORK HEALTH", color="#10B981")
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        fig_g2 = create_radial_gauge(66.0, "PRB SPECTRUM UTILIZATION", color="#F59E0B")
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        fig_g3 = create_radial_gauge(55.0, "BUFFER & QUEUE LOAD", color="#EF4444")
        st.plotly_chart(fig_g3, use_container_width=True)

    with g4:
        fig_g4 = create_radial_gauge(21.0, "ANOMALY RISK INDEX", color="#0EA5E9")
        st.plotly_chart(fig_g4, use_container_width=True)

    # 3. System Forecast Alert Banner
    col_fc_text, col_fc_btn = st.columns([3, 1])
    with col_fc_text:
        render_html(
            """
            <div class="noc-forecast-banner">
                <div>
                    <strong style="color: #F8FAFC; font-size: 0.92rem;">SYSTEM FORECAST &amp; PREDICTIVE RISK:</strong>
                    <div style="color: #CBD5E1; font-size: 0.84rem; margin-top: 2px;">
                        Infrastructure reliability is <span style="color: #EF4444; font-weight: 700;">21.0% AT RISK</span>.
                        4 alerts and 3 incidents are rising in sectors with elevated Downlink BLER.
                    </div>
                </div>
            </div>
            """
        )
    with col_fc_btn:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        if st.button("View AI Predictions", use_container_width=True, key="btn_fc_predictions"):
            navigate_to("05 — Alert Management")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Interactive RAN Topology & Node Inspector
    st.markdown("### Radio Access Network Topology & Node Diagnostics")
    st.caption("Inspect live cell towers, aggregation switches, and core gateway hierarchy:")

    col_topo, col_node = st.columns([1.3, 1])

    # Node selection control
    node_keys = list(NODE_TELEMETRY_DATA.keys())
    node_labels = [NODE_TELEMETRY_DATA[k]["title"] for k in node_keys]

    with col_node:
        with st.container(border=True):
            render_html(
                """
                <h3 style="margin-top: 0; margin-bottom: 8px; font-size: 1rem;">Node Telemetry Inspector</h3>
                """
            )

            chosen_node_idx = st.selectbox(
                "Select Network Node to Inspect:",
                range(len(node_keys)),
                format_func=lambda i: node_labels[i],
                index=0,  # Default to cell-macro-02 (degraded)
                key="topo_node_selector",
            )
            selected_key = node_keys[chosen_node_idx]
            node_info = NODE_TELEMETRY_DATA[selected_key]

            render_html(
                f"""
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.92rem; font-weight: 700; color: #F8FAFC;">{node_info['title']}</span>
                        <span class="{node_info['status_class']}">{node_info['status']}</span>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; color: #CBD5E1; margin-bottom: 12px;">
                        <tr style="border-bottom: 1px solid #1A263D;">
                            <td style="padding: 5px 0; color: #64748B;">Node Role / Type</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: #F8FAFC;">{node_info['role']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1A263D;">
                            <td style="padding: 5px 0; color: #64748B;">IP Address</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: #38BDF8;">{node_info['ip']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1A263D;">
                            <td style="padding: 5px 0; color: #64748B;">Uplink Gateway</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: #CBD5E1;">{node_info['switch']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1A263D;">
                            <td style="padding: 5px 0; color: #64748B;">PRB Spectrum Load</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: #F8FAFC;">{node_info['prb']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1A263D;">
                            <td style="padding: 5px 0; color: #64748B;">Downlink BLER</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: {'#EF4444' if 'DEGRADED' in node_info['status'] else '#10B981'};">{node_info['bler']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px 0; color: #64748B;">Radio RSRP / SNR</td>
                            <td style="padding: 5px 0; font-weight: 600; text-align: right; color: #F8FAFC;">{node_info['rsrp']} / {node_info['snr']}</td>
                        </tr>
                    </table>

                    <div style="background: #080C14; border: 1px solid #1A263D; border-radius: 4px; padding: 8px 10px; font-size: 0.76rem; color: #94A3B8; line-height: 1.4; margin-bottom: 12px;">
                        {node_info['note']}
                    </div>
                </div>
                """
            )

            if st.button(f"Triage Alarms for {selected_key} →", use_container_width=True, key="btn_triage_selected_node"):
                navigate_to("05 — Alert Management")

    with col_topo:
        topo_fig = create_topology_network_graph(selected_node=selected_key)
        st.plotly_chart(topo_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. 24-Hour Telemetry Dynamics & Baseline Distribution
    c_spline, c_donut = st.columns([1.5, 1])

    with c_spline:
        spline_fig = create_kpi_trend_splines()
        st.plotly_chart(spline_fig, use_container_width=True)

    with c_donut:
        st.markdown(
            """
            <div style="font-size: 0.86rem; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">
                Trained Baseline Distribution (N=32,000)
            </div>
            """,
            unsafe_allow_html=True,
        )
        donut_fig = create_donut_class_distribution(30765, 1235)
        st.plotly_chart(donut_fig, use_container_width=True)

    # Bottom Actions
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("Ingest Network Data (Stage 03) →", use_container_width=True, key="btn_ov_ingest"):
            navigate_to("03 — Data Ingestion")
    with c_btn2:
        if st.button("Configure AI Detection (Stage 04) →", use_container_width=True, key="btn_ov_detect"):
            navigate_to("04 — AI Detection")
