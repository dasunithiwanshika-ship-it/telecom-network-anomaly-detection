"""
Page 01: Royal Violet Cosmic Hero & Portal Access.
Features a pure-black starfield canvas, glowing violet planetary rim arching across the top,
bold centered display typography, and a rising dark-glass dashboard preview edged in violet light.
"""

from typing import Any, Optional
import streamlit as st
from ..components.metric_cards import navigate_to, render_html
from ..components.charts import create_royal_violet_area_chart
from src.feature_engineering import create_synthetic_demo_features


def render(pipeline: Optional[Any] = None) -> None:
    """Render the Royal Violet Telecom AI Hero and Portal Access screen."""
    # 1. Cosmic Hero Stage: Top Nav + Glowing Violet Planet Rim + Centered Headline
    render_html(
        """
        <div class="rv-hero-stage">
            <div class="rv-planet-rim">
                <div class="rv-planet-arc"></div>
            </div>

            <!-- Top Navigation Strip -->
            <div style="position: relative; z-index: 2; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 36px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #A020FF 0%, #7A1CF8 100%); width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 18px rgba(147, 18, 255, 0.7);">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                    </div>
                    <span style="font-weight: 700; font-size: 1rem; color: #F7F7F8; letter-spacing: -0.02em;">NEXVANTA</span>
                </div>

                <div style="display: flex; align-items: center; gap: 6px; background: rgba(14, 11, 22, 0.85); border: 1px solid rgba(147, 18, 255, 0.2); border-radius: 999px; padding: 4px 8px; font-size: 0.78rem; color: #8B8988;">
                    <span style="background: rgba(147, 18, 255, 0.22); color: #F7F7F8; padding: 4px 12px; border-radius: 999px; font-weight: 600;">Portal</span>
                    <span style="padding: 4px 10px;">Overview</span>
                    <span style="padding: 4px 10px;">Telemetry</span>
                    <span style="padding: 4px 10px;">AI Detection</span>
                    <span style="padding: 4px 10px;">Alerts</span>
                </div>

                <div style="font-size: 0.78rem; color: #D8B4FE; background: rgba(147, 18, 255, 0.14); border: 1px solid rgba(147, 18, 255, 0.35); border-radius: 999px; padding: 6px 14px; font-weight: 600;">
                    ● NOC Engine Online
                </div>
            </div>

            <!-- Centered Display Headline & Lede -->
            <div style="position: relative; z-index: 2; text-align: center; max-width: 820px; margin: 0 auto 24px auto;">
                <div style="display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #D8B4FE; background: rgba(147, 18, 255, 0.14); border: 1px solid rgba(147, 18, 255, 0.35); padding: 4px 12px; border-radius: 999px; margin-bottom: 14px;">
                    01 — TELECOM AI INTELLIGENCE PLATFORM
                </div>
                <h1 style="font-size: clamp(2.2rem, 4.4vw, 3.6rem); font-weight: 700; line-height: 1.06; margin: 0 0 16px 0; color: #F7F7F8; letter-spacing: -0.035em;">
                    <span class="violet-grad">Telecom AI</span> Monitor Smarter,<br>Faster, Safer.
                </h1>
                <p style="font-size: 0.98rem; color: #8B8988; line-height: 1.6; max-width: 640px; margin: 0 auto;">
                    The first fully integrated cellular anomaly intelligence experience. No code, no delays — NEXVANTA empowers telecom engineers to detect 4G/5G radio and scheduling faults directly from live KPI telemetry.
                </p>
            </div>
        </div>
        """
    )

    # 2. Centered Hero Pill CTA Buttons
    sp_l, c_cta1, c_cta2, sp_r = st.columns([1, 1.3, 1.3, 1])
    with c_cta1:
        if st.button("Get Started — Run AI Detection", use_container_width=True, key="btn_rv_get_started"):
            demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
            st.session_state["active_dataset_df"] = demo_df
            st.session_state["active_dataset_name"] = "Standard Field Batch (25 Windows)"
            navigate_to("04 — AI Detection")
    with c_cta2:
        if st.button("Explore NOC Overview", use_container_width=True, key="btn_rv_overview"):
            navigate_to("02 — Network Overview")

    # 3. Rising Dark-Glass Dashboard Preview Panel with Glowing Violet Top Edge
    render_html(
        """
        <div class="rv-dashboard-frame">
            <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 14px; border-bottom: 1px solid rgba(147, 18, 255, 0.16); margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 700; color: #F7F7F8;">Live NOC Preview</span>
                    <span class="pill-violet">3GPP TS 36.314</span>
                </div>
                <div style="display: flex; align-items: center; gap: 14px; font-size: 0.78rem; color: #8B8988;">
                    <span>Connected layers: <strong style="color: #F7F7F8;">Radio · MAC Scheduler</strong></span>
                    <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(147, 18, 255, 0.25); padding: 4px 10px; border-radius: 999px; color: #F7F7F8;">Alex T. · NOC-8491</span>
                </div>
            </div>
        </div>
        """
    )

    with st.container(border=True):
        dash_side, dash_main, dash_cal = st.columns([0.9, 2.0, 1.1])

        with dash_side:
            render_html(
                """
                <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                    <div>
                        <div style="font-size: 0.7rem; color: #8B8988; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 10px;">Workspace</div>
                        <div style="background: rgba(147, 18, 255, 0.2); border: 1px solid rgba(160, 32, 255, 0.45); color: #F7F7F8; padding: 7px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">Dashboard</div>
                        <div style="color: #8B8988; padding: 6px 12px; font-size: 0.8rem;">Cell Monitors</div>
                        <div style="color: #8B8988; padding: 6px 12px; font-size: 0.8rem;">Telemetry Batches</div>
                        <div style="color: #8B8988; padding: 6px 12px; font-size: 0.8rem;">Alert Rules</div>
                        <div style="color: #8B8988; padding: 6px 12px; font-size: 0.8rem;">Model Settings</div>
                    </div>
                    <div style="background: linear-gradient(135deg, rgba(147, 18, 255, 0.22) 0%, rgba(11, 9, 18, 0.9) 100%); border: 1px solid rgba(160, 32, 255, 0.35); border-radius: 10px; padding: 12px; margin-top: 18px;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #F7F7F8;">NEXVANTA Pro</div>
                        <div style="font-size: 0.72rem; color: #8B8988; margin: 4px 0 8px 0;">Sub-15ms RF anomaly inference active.</div>
                        <span class="pill-violet">Active Tier</span>
                    </div>
                </div>
                """
            )

        with dash_main:
            s_c1, s_c2 = st.columns(2)
            with s_c1:
                render_html(
                    """
                    <div style="background: rgba(18, 14, 28, 0.85); border: 1px solid rgba(147, 18, 255, 0.2); border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; color: #8B8988;">Telemetry windows evaluated</div>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin-top: 4px;">
                            <span style="font-size: 1.55rem; font-weight: 700; color: #F7F7F8;">32,000</span>
                            <span style="font-size: 0.74rem; font-weight: 600; color: #45B558; background: rgba(69, 181, 88, 0.14); padding: 2px 7px; border-radius: 999px;">+12.5%</span>
                        </div>
                    </div>
                    """
                )
            with s_c2:
                render_html(
                    """
                    <div style="background: rgba(18, 14, 28, 0.85); border: 1px solid rgba(147, 18, 255, 0.2); border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; color: #8B8988;">Active anomaly incidents</div>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin-top: 4px;">
                            <span style="font-size: 1.55rem; font-weight: 700; color: #F7F7F8;">4</span>
                            <span style="font-size: 0.74rem; font-weight: 600; color: #F03B2A; background: rgba(240, 59, 42, 0.14); padding: 2px 7px; border-radius: 999px;">-3.1%</span>
                        </div>
                    </div>
                    """
                )

            fig_area = create_royal_violet_area_chart()
            st.plotly_chart(fig_area, use_container_width=True)

        with dash_cal:
            render_html(
                """
                <div style="background: rgba(18, 14, 28, 0.85); border: 1px solid rgba(147, 18, 255, 0.2); border-radius: 10px; padding: 14px;">
                    <div style="font-size: 0.86rem; font-weight: 700; color: #F7F7F8;">Cluster Summary</div>
                    <div style="font-size: 0.72rem; color: #8B8988; margin-bottom: 12px;">Baseline Window · Monthly</div>

                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(147, 18, 255, 0.14); font-size: 0.8rem;">
                        <span style="color: #8B8988;">Total anomalies</span>
                        <span><strong style="color: #F7F7F8;">1,235</strong> <span style="color: #45B558; font-size: 0.72rem;">+8.4%</span></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(147, 18, 255, 0.14); font-size: 0.8rem;">
                        <span style="color: #8B8988;">Avg. execution</span>
                        <strong style="color: #F7F7F8;">14ms</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(147, 18, 255, 0.14); font-size: 0.8rem;">
                        <span style="color: #8B8988;">AI Precision</span>
                        <span><strong style="color: #F7F7F8;">99.2%</strong> <span style="color: #45B558; font-size: 0.72rem;">+2.1%</span></span>
                    </div>
                    <div style="padding-top: 10px; font-size: 0.74rem; color: #8B8988;">
                        Connected KPIs: <strong style="color: #D8B4FE;">RSRP · BLER · PRB</strong>
                    </div>
                </div>
                """
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Operator Profile Access & Stage Navigation
    c_auth, c_stages = st.columns([1, 2])
    with c_auth:
        with st.container(border=True):
            render_html(
                """
                <div style="margin-bottom: 10px;">
                    <div style="font-weight: 700; font-size: 1rem; color: #F7F7F8;">Operator Session</div>
                    <div style="font-size: 0.76rem; color: #8B8988;">Authorize role and enter workspace</div>
                </div>
                """
            )
            operator_role = st.selectbox(
                "Operator Profile:",
                [
                    "Alex Chen — Principal NOC Performance Engineer",
                    "Sarah Lin — Senior AI Network Specialist",
                    "Marcus Vance — Field Operations Lead",
                ],
                index=0,
                key="login_operator_role",
            )
            user_name = st.text_input("Operator ID:", value="noc-engineer-8491", key="login_user_id")
            if st.button("Authenticate & Enter Overview →", use_container_width=True, key="btn_login_enter"):
                st.session_state["logged_in_user"] = user_name
                navigate_to("02 — Network Overview")

    with c_stages:
        st.markdown("### End-to-End Operational Stages")
        st.caption("Select any stage to continue your telemetry investigation:")
        steps = [
            ("01", "Portal Access", "Operator auth & overview", "01 — Portal Access"),
            ("02", "Network Overview", "Gauges & topology map", "02 — Network Overview"),
            ("03", "Data Ingestion", "CSV upload & AI query", "03 — Data Ingestion"),
            ("04", "AI Detection", "Model & cutoff setup", "04 — AI Detection"),
            ("05", "Alert Management", "Incident triage console", "05 — Alert Management"),
            ("06", "Investigation", "Radar & report export", "06 — Investigation & Export"),
        ]
        cols = st.columns(3)
        for idx, (num, title, desc, target_nav) in enumerate(steps):
            with cols[idx % 3]:
                render_html(
                    f"""
                    <div style="background: rgba(11, 9, 18, 0.9); border: 1px solid rgba(147, 18, 255, 0.2); border-radius: 10px; padding: 10px 12px; margin-bottom: 6px;">
                        <div style="font-size: 0.66rem; font-weight: 700; color: #B76EFF;">STAGE {num}</div>
                        <div style="font-size: 0.84rem; font-weight: 700; color: #F7F7F8;">{title}</div>
                        <div style="font-size: 0.72rem; color: #8B8988;">{desc}</div>
                    </div>
                    """
                )
                if st.button(f"Open {num}", key=f"btn_step_nav_{num}", use_container_width=True):
                    navigate_to(target_nav)
