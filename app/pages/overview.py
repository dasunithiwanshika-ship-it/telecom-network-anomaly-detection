"""
Page 02: NOC Network Overview & Operational Intelligence.
Refined enterprise interface featuring radial health gauges, top status ribbon,
system forecast notices, infrastructure inventory, and interactive topology map.
"""

from typing import Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_noc_status_strip,
    navigate_to,
)
from ..components.charts import (
    create_radial_gauge,
    create_kpi_trend_splines,
    create_topology_network_graph,
    create_donut_class_distribution,
)
from src.data_loader import load_processed_features


@st.cache_data
def _get_baseline_dataset():
    try:
        return load_processed_features()
    except Exception:
        return None, None


def render(pipeline: Any = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the executive NOC Overview screen."""
    now_str = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(
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
        """,
        unsafe_allow_html=True,
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
        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )
    with col_fc_btn:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        if st.button("View AI Predictions", use_container_width=True, key="btn_fc_predictions"):
            navigate_to("05 — Alert Management")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Infrastructure Inventory & Topo-Map
    col_inv, col_topo = st.columns([1.1, 1])

    with col_inv:
        st.markdown(
            """
            <div class="telecom-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 0.95rem;">Infrastructure Inventory</h3>
                    <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">30 NODES REGISTERED</span>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; color: #CBD5E1;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1A263D; color: #64748B; text-align: left; font-size: 0.7rem; text-transform: uppercase;">
                            <th style="padding: 6px 4px;">Node Identity</th>
                            <th style="padding: 6px 4px;">Role</th>
                            <th style="padding: 6px 4px;">Load</th>
                            <th style="padding: 6px 4px;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #141E30;">
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">cell-macro-01</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">eNodeB Sector 1</td>
                            <td style="padding: 8px 4px; color: #10B981;">42% PRB</td>
                            <td style="padding: 8px 4px;"><span class="pill-normal">OPTIMAL</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #141E30;">
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">cell-macro-02</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">eNodeB Sector 2</td>
                            <td style="padding: 8px 4px; color: #F59E0B;">78% PRB</td>
                            <td style="padding: 8px 4px;"><span class="pill-high">HIGH BLER</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #141E30;">
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">cell-small-03</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">gNodeB Hotspot</td>
                            <td style="padding: 8px 4px; color: #10B981;">35% PRB</td>
                            <td style="padding: 8px 4px;"><span class="pill-normal">OPTIMAL</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #141E30;">
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">cell-macro-04</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">eNodeB Sector 3</td>
                            <td style="padding: 8px 4px; color: #EF4444;">94% PRB</td>
                            <td style="padding: 8px 4px;"><span class="pill-critical">BUFFER FULL</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #141E30;">
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">agg-switch-01</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">Backhaul Switch</td>
                            <td style="padding: 8px 4px; color: #10B981;">28% BW</td>
                            <td style="padding: 8px 4px;"><span class="pill-normal">OPTIMAL</span></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 4px; font-weight: 600; color: #F8FAFC;">core-upf-01</td>
                            <td style="padding: 8px 4px; color: #94A3B8;">5G User Plane</td>
                            <td style="padding: 8px 4px; color: #10B981;">31% CPU</td>
                            <td style="padding: 8px 4px;"><span class="pill-normal">OPTIMAL</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_topo:
        topo_fig = create_topology_network_graph()
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
        if st.button("Ingest Network Data (Stage 03)", use_container_width=True, key="btn_ov_ingest"):
            navigate_to("03 — Data Ingestion")
    with c_btn2:
        if st.button("Configure AI Detection (Stage 04)", use_container_width=True, key="btn_ov_detect"):
            navigate_to("04 — AI Detection")
