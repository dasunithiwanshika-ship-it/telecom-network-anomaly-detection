"""
Page 8: Data Management & Feature Schema
Provides detailed documentation of the TelecomTS dataset, the full 64-feature dictionary,
telemetry validation rules, and template downloads for inference.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
from app.components.metric_cards import render_hero_banner, render_academic_disclaimer

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
SCHEMA_PATH = MODELS_DIR / "feature_names.json"

FEATURE_DESCRIPTIONS = {
    "RSRP": "Reference Signal Received Power (dBm): Measures cellular RF coverage and received signal power from the serving gNodeB/eNodeB.",
    "UL_SNR": "Uplink Signal-to-Noise Ratio (dB): Measures uplink RF link quality and channel clarity over thermal noise and interference.",
    "UL_BLER": "Uplink Block Error Rate (0.0–1.0): Proportion of transmitted uplink transport blocks received with unrecoverable CRC errors.",
    "DL_BLER": "Downlink Block Error Rate (0.0–1.0): Proportion of downlink transport blocks received in error, signaling downlink interference or fading.",
    "UL_MCS": "Uplink Modulation and Coding Scheme index (0–31): Adaptive modulation level reflecting link condition and spectral efficiency.",
    "DL_MCS": "Downlink Modulation and Coding Scheme index (0–31): Adaptive modulation level for downstream data stream.",
    "UL_NPRB": "Number of Physical Resource Blocks allocated in Uplink: Measures scheduled channel bandwidth and frequency slice allocation.",
    "PRBs_DL_Current": "Current Downlink PRB Count: Instantaneous physical resource blocks assigned for downlink user data.",
    "PRBs_UL_Current": "Current Uplink PRB Count: Instantaneous physical resource blocks assigned for uplink user data.",
    "PRB_Utilization_DL": "Downlink PRB Utilization Ratio (0.0–1.0): Fraction of total available cell frequency bandwidth utilized downstream.",
    "PRB_Utilization_UL": "Uplink PRB Utilization Ratio (0.0–1.0): Fraction of total available cell frequency bandwidth utilized upstream.",
    "TX_Bytes": "Transmitted Byte Count: Raw throughput volume pushed across the interface in the sampling interval.",
    "RX_Bytes": "Received Byte Count: Raw throughput volume pulled across the interface in the sampling interval.",
    "UL_NumberOfPackets": "Uplink Packet Count: Number of discrete data packets transmitted upstream.",
    "DL_NumberOfPackets": "Downlink Packet Count: Number of discrete data packets transmitted downstream.",
    "Estimated_UL_Buffer": "Estimated Uplink Buffer Occupancy: Backlogged user equipment queue size awaiting uplink scheduling grants."
}

KPI_DOMAINS = {
    "RSRP": "Radio Quality",
    "UL_SNR": "Signal Quality",
    "UL_BLER": "Error Rates",
    "DL_BLER": "Error Rates",
    "UL_MCS": "Modulation",
    "DL_MCS": "Modulation",
    "UL_NPRB": "Scheduling & Allocation",
    "PRBs_DL_Current": "Scheduling & Allocation",
    "PRBs_UL_Current": "Scheduling & Allocation",
    "PRB_Utilization_DL": "Scheduling & Allocation",
    "PRB_Utilization_UL": "Scheduling & Allocation",
    "TX_Bytes": "Traffic & Throughput",
    "RX_Bytes": "Traffic & Throughput",
    "UL_NumberOfPackets": "Traffic & Throughput",
    "DL_NumberOfPackets": "Traffic & Throughput",
    "Estimated_UL_Buffer": "Buffer Health"
}

@st.cache_data
def get_feature_dictionary():
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r") as f:
            features = json.load(f)
    else:
        features = []
        
    rows = []
    for f in features:
        # split feature into base KPI and moment
        base_kpi = None
        moment = None
        for m in ["mean", "std", "min", "max"]:
            if f.endswith(f"_{m}"):
                base_kpi = f[:-len(m)-1]
                moment = m.upper()
                break
        if not base_kpi:
            base_kpi = f
            moment = "N/A"
            
        domain = KPI_DOMAINS.get(base_kpi, "General Telemetry")
        desc = FEATURE_DESCRIPTIONS.get(base_kpi, "Cellular network KPI telemetry metric.")
        moment_desc = f"{moment} value aggregated over the 128-step time series window."
        
        rows.append({
            "Feature Name": f,
            "Domain Category": domain,
            "Base KPI": base_kpi,
            "Moment": moment,
            "Interpretation": f"{desc} ({moment_desc})"
        })
    return pd.DataFrame(rows)

def render():
    render_hero_banner(
        title="Data Management & Feature Schema",
        subtitle="TelecomTS telemetry specifications, 64-feature dictionary schema, and inference template generator.",
        badge="DATA ARCHITECTURE"
    )
    
    render_academic_disclaimer()
    
    st.markdown("### 1. TelecomTS Dataset Architecture")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Total Time Series</div>
                <div style="color: #14B8A6; font-size: 1.4rem; font-weight: 700; margin-top: 4px;">32,000</div>
                <div style="color: #64748B; font-size: 0.75rem;">Multivariate cell traces</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_d2:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Window Length</div>
                <div style="color: #38BDF8; font-size: 1.4rem; font-weight: 700; margin-top: 4px;">128 Steps</div>
                <div style="color: #64748B; font-size: 0.75rem;">Timesteps per trace</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_d3:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Base KPIs / Features</div>
                <div style="color: #F8FAFC; font-size: 1.4rem; font-weight: 700; margin-top: 4px;">16 / 64</div>
                <div style="color: #64748B; font-size: 0.75rem;">16 KPIs &times; 4 moments</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_d4:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Anomaly Prevalence</div>
                <div style="color: #F59E0B; font-size: 1.4rem; font-weight: 700; margin-top: 4px;">3.86%</div>
                <div style="color: #64748B; font-size: 0.75rem;">1,235 total anomaly traces</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 2. Feature Schema & Telemetry Dictionary")
    st.markdown(
        """
        Each cell trace is represented as a structured 64-dimensional feature vector consisting of four statistical moments 
        (<strong>mean</strong>, <strong>standard deviation</strong>, <strong>minimum</strong>, and <strong>maximum</strong>) 
        calculated over the 128 timesteps for all 16 cellular KPIs.
        """
    )
    
    df_dict = get_feature_dictionary()
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_query = st.text_input("Search Feature Name or Description", placeholder="e.g. BLER, SNR, Utilization, min...")
    with col_f2:
        category_filter = st.selectbox("Filter by Domain Category", ["All Categories"] + sorted(list(set(df_dict["Domain Category"]))))
        
    filtered_df = df_dict.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["Feature Name"].str.contains(search_query, case=False) |
            filtered_df["Interpretation"].str.contains(search_query, case=False)
        ]
    if category_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["Domain Category"] == category_filter]
        
    st.dataframe(filtered_df, use_container_width=True, height=360)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 3. Inference Template & Sample Telemetry Generator")
    st.markdown(
        """
        To run inference on your own batch data in the **Anomaly Detection Workspace**, 
        ensure the uploaded CSV contains all 64 feature columns listed above. 
        You can download an empty CSV template or generate sample synthetic telemetry below.
        """
    )
    
    col_tmp1, col_tmp2 = st.columns(2)
    with col_tmp1:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">CSV Schema Template</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;">
                    Empty CSV file containing exactly the 64 required column headers matching the trained model pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        empty_df = pd.DataFrame(columns=df_dict["Feature Name"].tolist())
        st.download_button(
            label="Download 64-Feature CSV Template",
            data=empty_df.to_csv(index=False),
            file_name="telecom_kpi_64feature_template.csv",
            mime="text/csv"
        )
        
    with col_tmp2:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 16px;">
                <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Sample Synthetic Telemetry</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;">
                    A generated dataset of 50 synthetic records (40 normal, 10 perturbed anomalies) ready for immediate testing.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        from src.data_loader import generate_synthetic_inference_batch
        sample_df = generate_synthetic_inference_batch(n_samples=50, anomaly_ratio=0.20, random_state=42)
        st.download_button(
            label="Download Sample Synthetic CSV (50 traces)",
            data=sample_df.to_csv(index=False),
            file_name="sample_telemetry_50records.csv",
            mime="text/csv"
        )
