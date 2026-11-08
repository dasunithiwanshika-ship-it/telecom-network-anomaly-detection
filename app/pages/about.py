"""
Page 9: About, Platform Architecture & Academic Boundaries
Presents platform architecture, system specifications, technology stack,
author credentials, and mandatory academic/industry boundary disclaimers.
"""

import streamlit as st
from app.components.metric_cards import render_hero_banner, render_academic_disclaimer

def render():
    render_hero_banner(
        title="About the Telecom AI Intelligence Platform",
        subtitle="System architecture, technological specifications, research credentials, and academic boundaries.",
        badge="PLATFORM ARCHITECTURE"
    )
    
    render_academic_disclaimer()
    
    st.markdown("### 1. Platform Architecture & Data Pipeline")
    
    st.markdown(
        """
        <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 20px; margin-bottom: 24px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 1.1rem; margin-bottom: 12px;">End-to-End Hybrid Machine Learning Pipeline</div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">1</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Telemetry Ingestion:</strong> Ingests 64 windowed statistical features (mean, std, min, max across 16 cellular KPIs) via batch CSV or sample generator.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">2</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Robust Preprocessing:</strong> Schema validation against <code>feature_names.json</code>, non-numeric coercion, and zero-imputation.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #14B8A6; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">3</span>
                    <div style="color: #14B8A6; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">3</div>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Feature Normalization:</strong> Zero-mean unit-variance transformation using fitted <code>StandardScaler</code> artifact for distance-based models.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #38BDF8; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">4</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Model Inference Engine:</strong> Multi-paradigm execution supporting Random Forest, Logistic Regression, Isolation Forest, and Local Outlier Factor.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #38BDF8; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">5</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>Dynamic Decision Thresholding:</strong> Configurable cutoffs (default &tau;=0.50 vs 5-fold CV-tuned &tau;=0.30) balancing false alarms against outage risk.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="background-color: #F59E0B; color: #0B1220; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">6</span>
                    <div style="color: #F8FAFC; font-size: 0.9rem;"><strong>XAI & Triage Analytics:</strong> Z-score statistical moment deviation against 30,765 baseline traces, attributing anomalous drivers per record.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 2. Technology Stack & Specifications")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #14B8A6; margin-bottom: 10px;">Core Engineering Libraries</div>
                <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Runtime Environment:</strong> Python 3.13.5 (64-bit)</li>
                    <li><strong>Machine Learning:</strong> scikit-learn 1.9.1</li>
                    <li><strong>Data Processing:</strong> pandas 3.0.5, numpy 2.4.2</li>
                    <li><strong>Model Serialization:</strong> joblib 1.5.3</li>
                    <li><strong>Visualization:</strong> plotly 7.1.0, matplotlib 3.10.0</li>
                    <li><strong>Web Architecture:</strong> Streamlit 1.63.0</li>
                    <li><strong>Testing Suite:</strong> pytest 9.1.1</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #38BDF8; margin-bottom: 10px;">Project Metadata</div>
                <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Application:</strong> Telecom AI Intelligence Platform</li>
                    <li><strong>System Version:</strong> 2.0 (Enterprise Hybrid Edition)</li>
                    <li><strong>Research Role:</strong> AI / Machine Learning Engineer</li>
                    <li><strong>Target Completion:</strong> 28 November 2026</li>
                    <li><strong>Active Branch:</strong> <code>feat/hybrid-telecom-ai-platform</code></li>
                    <li><strong>Dataset Source:</strong> Hugging Face (<code>AliMaatouk/TelecomTS</code>)</li>
                    <li><strong>Test Telemetry Volume:</strong> 6,400 evaluated cell traces</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 3. Industry Boundary & Academic Ethics Statement")
    
    st.markdown(
        """
        <div style="background-color: #111827; border: 2px solid #EF4444; border-radius: 8px; padding: 20px;">
            <div style="font-weight: 700; color: #EF4444; font-size: 1rem; margin-bottom: 8px;">MANDATORY ACADEMIC & OPERATIONAL BOUNDARY</div>
            <div style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.6;">
                <p>1. <strong>Independent Academic Prototype:</strong> This platform is an independent AI/ML research demonstration developed strictly for evaluation purposes. It is trained and benchmarked entirely on public, open-access cellular KPI time-series data (<em>AliMaatouk/TelecomTS</em>).</p>
                <p>2. <strong>Zero Proprietary Data:</strong> This platform does <strong>NOT</strong> ingest, utilize, or reference proprietary data from Sri Lanka Telecom (SLT), SLTMobitel, or any Sri Lankan telecommunications network.</p>
                <p>3. <strong>No Physical Breakdown Causality:</strong> The machine learning models identify mathematical and statistical degradation in cellular radio and transport KPIs. They do <strong>NOT</strong> predict physical distribution point (DP) or fiber distribution point (FDP) cable cuts, physical fiber breaks, or technician dispatch requirements.</p>
                <p>4. <strong>Offline/Historical Analysis Mode:</strong> The platform operates exclusively on historical and user-uploaded batch telemetry. There is no active integration with live production Network Operations Centers (NOC).</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
