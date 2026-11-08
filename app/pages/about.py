"""
Page 9: About, Telecom Engineer's Guide & Academic Boundaries.
Provides a plain-English glossary for Telecom Network Engineers, system architecture,
technology specifications, and mandatory academic/industry boundary statements.
"""

import streamlit as st
from ..components.metric_cards import render_hero_banner, render_academic_disclaimer


def render():
    """Render the About & System Guide page."""
    render_hero_banner(
        title="About the Platform & Telecom Operator Guide",
        subtitle="Quick-reference glossary for network engineers, end-to-end diagnostic architecture, and operational boundary statements.",
        badge="OPERATOR GUIDE & ARCHITECTURE",
    )

    st.markdown("### 📖 1. Telecom Engineer's Quick Reference Glossary")
    st.caption("Plain-English definitions of key cellular KPIs and diagnostic terms used across the platform:")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>📡 Radio &amp; Link Quality Terms</h3>
                <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.65; padding-left: 18px;">
                    <li><strong>RSRP (Reference Signal Received Power):</strong> Measures downlink cell signal strength in dBm. Values near <code>-75 dBm</code> indicate strong coverage; below <code>-115 dBm</code> indicates weak cell-edge reception.</li>
                    <li><strong>UL SNR (Uplink Signal-to-Noise Ratio):</strong> Measures uplink signal clarity in dB relative to thermal noise and interference. Drops in UL SNR point to external RF interference or jamming.</li>
                    <li><strong>UL / DL BLER (Block Error Rate):</strong> Fraction of transport blocks arriving with CRC errors (0.0 to 1.0). Persistent BLER above <code>0.10 (10%)</code> signals link instability and HARQ retransmissions.</li>
                    <li><strong>UL / DL MCS (Modulation &amp; Coding Scheme):</strong> Link adaptation index (0–28). High MCS uses 64-QAM for high speed; low MCS falls back to QPSK when signal quality degrades.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>⚙️ Scheduling, Buffer &amp; AI Terms</h3>
                <ul style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.65; padding-left: 18px;">
                    <li><strong>PRB (Physical Resource Block):</strong> Fundamental unit of frequency-time spectrum allocated by the base station MAC scheduler (<code>UL_NPRB</code>, <code>PRB_Utilization_DL/UL</code>).</li>
                    <li><strong>Estimated UL Buffer:</strong> Volume of data waiting in the user device queue for uplink transmission grants. High buffer backlog combined with low PRB grants indicates scheduling starvation.</li>
                    <li><strong>Risk Score / Probability:</strong> Value from <code>0.00</code> to <code>1.00</code> indicating how closely a 12.8-second cell window matches known anomaly patterns.</li>
                    <li><strong>Deviation (σ / Sigma):</strong> Measures how far a KPI reading is from the normal network average in standard deviations. Readings beyond <code>±2.0σ</code> are outside normal limits.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏗️ 2. End-to-End Diagnostic Pipeline Architecture")
    st.markdown(
        """
        <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 10px; padding: 20px 24px; margin-bottom: 22px;">
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 800; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">1</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Telemetry Ingestion:</strong> Accepts 64 windowed KPI statistics (Average, Std Dev, Min, Max across 16 cellular KPIs) via CSV upload or built-in field samples.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 800; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">2</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Automated Validation &amp; Sanitization:</strong> Checks all 64 column headers against <code>feature_names.json</code> and safely cleans any missing or non-numeric entries.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 800; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">3</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Feature Normalization:</strong> Applies pre-fitted <code>StandardScaler</code> normalization for distance and linear models while preserving raw scales for tree ensembles.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #38BDF8; color: #0B1220; font-weight: 800; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">4</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Multi-Model AI Detection:</strong> Executes pre-trained Random Forest, Logistic Regression, Isolation Forest, or Local Outlier Factor artifacts.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #F59E0B; color: #0B1220; font-weight: 800; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;">5</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Baseline Comparison &amp; Root-Cause Triage:</strong> Compares each record against 30,765 healthy cell traces to isolate the exact subsystem and KPI driving the alert.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💻 3. System Specifications & Project Metadata")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>Core Software Stack</h3>
                <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Runtime Environment:</strong> Python 3.13 (64-bit)</li>
                    <li><strong>Machine Learning Engine:</strong> scikit-learn 1.9</li>
                    <li><strong>Data Processing:</strong> pandas 3.0, numpy 2.4</li>
                    <li><strong>Interactive Visuals:</strong> Plotly 7.1 &amp; Streamlit 1.63</li>
                    <li><strong>Automated QA Suite:</strong> pytest (22 verification tests)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>Research &amp; Benchmark Metadata</h3>
                <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Platform:</strong> Telecom AI Intelligence Platform v2.0</li>
                    <li><strong>Reference Dataset:</strong> Hugging Face <code>AliMaatouk/TelecomTS</code></li>
                    <li><strong>Benchmark Volume:</strong> 32,000 Cell Windows (25.6k Train / 6.4k Test)</li>
                    <li><strong>Monitored Telemetry:</strong> 16 Cellular KPIs &times; 4 Window Statistics</li>
                    <li><strong>Primary Detector Precision:</strong> 100.0% (at τ=0.50) / 99.17% (at τ=0.30)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🛡️ 4. Mandatory Academic & Industry Boundary Statement")
    st.markdown(
        """
        <div style="background-color: #111827; border: 2px solid #EF4444; border-radius: 8px; padding: 20px;">
            <div style="font-weight: 700; color: #EF4444; font-size: 1rem; margin-bottom: 8px;">OPERATIONAL &amp; ACADEMIC BOUNDARY NOTICE</div>
            <div style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.6;">
                <p>1. <strong>Independent Research Prototype:</strong> This application is an independent AI/ML telecom monitoring prototype built and evaluated on the open-access <em>AliMaatouk/TelecomTS</em> cellular KPI benchmark dataset.</p>
                <p>2. <strong>Zero Proprietary Data:</strong> This system does <strong>NOT</strong> contain or process proprietary telemetry from Sri Lanka Telecom (SLT), SLTMobitel, or any commercial Sri Lankan telecommunications network.</p>
                <p>3. <strong>Statistical Telemetry vs. Physical Faults:</strong> Flagged alerts identify statistical anomalies in radio and MAC-layer KPIs. They do <strong>NOT</strong> predict physical DP/FDP fiber/copper cable cuts or automatically dispatch field technicians.</p>
                <p>4. <strong>Offline / Batch Analysis Mode:</strong> When not connected to a live streaming interface, the platform operates in batch CSV analysis and historical benchmark mode.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_academic_disclaimer()
