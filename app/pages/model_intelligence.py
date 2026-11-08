"""
Page 6: Model Intelligence & Benchmarking
Presents comparative experimental evaluation across all 4 machine learning models,
confusion matrix analysis, ROC/PR comparisons, threshold sensitivity, and paradigm trade-offs.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.components.metric_cards import render_hero_banner, render_academic_disclaimer

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

@st.cache_data
def load_benchmark_data():
    comp_path = RESULTS_DIR / "unified_4model_comparison_table.csv"
    thresh_path = RESULTS_DIR / "threshold_sensitivity_analysis.csv"
    
    df_comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    df_thresh = pd.read_csv(thresh_path) if thresh_path.exists() else pd.DataFrame()
    
    return df_comp, df_thresh

def render():
    render_hero_banner(
        title="Model Intelligence & Benchmarking",
        subtitle="Empirical comparative evaluation of supervised classification vs unsupervised anomaly detection on 6,400 test traces.",
        badge="BENCHMARK LAB"
    )
    
    render_academic_disclaimer()
    
    df_comp, df_thresh = load_benchmark_data()
    
    if df_comp.empty:
        st.error("Model comparison table not found in `results/`. Please run Day 10 / Day 11 experiments.")
        return

    st.markdown("### 1. Unified 4-Model Experimental Benchmark Table")
    st.markdown(
        """
        All models were evaluated on the exact same held-out test split of **6,400 traces** (6,153 normal, 247 anomalous; anomaly prevalence = 3.86%). 
        No test telemetry was used during model fitting or threshold optimization.
        """
    )
    
    st.dataframe(df_comp, use_container_width=True, height=220)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual comparison chart
    st.markdown("### 2. Multi-Metric Performance Comparison")
    
    # Parse percentage strings to floats for plotting
    plot_df = df_comp.copy()
    for col in ["Test Precision", "Test Recall", "Test F1-Score", "ROC-AUC", "PR-AUC (Avg Precision)"]:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].astype(str).str.rstrip("%").astype(float)
            
    fig_comp = go.Figure()
    metrics = ["Test Precision", "Test Recall", "Test F1-Score", "ROC-AUC", "PR-AUC (Avg Precision)"]
    colors = ["#14B8A6", "#38BDF8", "#F59E0B", "#8B5CF6", "#EC4899"]
    
    for metric, color in zip(metrics, colors):
        fig_comp.add_trace(go.Bar(
            name=metric,
            x=plot_df["Model Name"],
            y=plot_df[metric],
            marker_color=color,
            text=plot_df[metric].apply(lambda v: f"{v:.1f}%"),
            textposition="auto"
        ))
        
    fig_comp.update_layout(
        barmode='group',
        paper_bgcolor="#111827",
        plot_bgcolor="#172235",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Score (%)", range=[0, 105], gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for Confusion Matrices, Threshold Optimization, and Paradigm Tradeoffs
    tab_cm, tab_thresh, tab_paradigm, tab_latency = st.tabs([
        "Confusion Matrix Analysis",
        "Threshold Sensitivity Trade-Offs",
        "Supervised vs Unsupervised Trade-Offs",
        "Computational Complexity & Latency"
    ])
    
    with tab_cm:
        st.markdown("##### 2x2 Confusion Matrix Breakdown on Held-Out Test Set (N=6,400)")
        cm_cols = st.columns(4)
        
        for idx, (_, row) in enumerate(df_comp.iterrows()):
            with cm_cols[idx]:
                tp = int(row["TP"])
                fp = int(row["FP"])
                fn = int(row["FN"])
                tn = int(row["TN"])
                model_name = row["Model Name"]
                
                st.markdown(
                    f"""
                    <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 14px; text-align: center;">
                        <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem; margin-bottom: 10px;">{model_name}</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; color: #F8FAFC;">
                            <tr style="border-bottom: 1px solid #334155;">
                                <td style="padding: 6px; background-color: rgba(20, 184, 166, 0.2);">TP: <strong style="color: #14B8A6;">{tp}</strong></td>
                                <td style="padding: 6px; background-color: rgba(239, 68, 68, 0.2);">FP: <strong style="color: #EF4444;">{fp}</strong></td>
                            </tr>
                            <tr>
                                <td style="padding: 6px; background-color: rgba(245, 158, 11, 0.2);">FN: <strong style="color: #F59E0B;">{fn}</strong></td>
                                <td style="padding: 6px; background-color: rgba(56, 189, 248, 0.15);">TN: <strong style="color: #38BDF8;">{tn}</strong></td>
                            </tr>
                        </table>
                        <div style="margin-top: 8px; font-size: 0.75rem; color: #94A3B8;">
                            Recall: <strong>{row['Test Recall']}</strong> | Precision: <strong>{row['Test Precision']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        st.markdown(
            """
            <div style="margin-top: 15px; color: #94A3B8; font-size: 0.85rem;">
                <strong>Interpretation</strong>: Random Forest achieved 0 False Positives on test telemetry at threshold 0.50, correctly isolating 230 out of 247 incidents.
                In contrast, unsupervised models (Isolation Forest and LOF) yielded ~174–193 False Positives due to normal multi-modal traffic distributions that deviate from the training centroid.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_thresh:
        st.markdown("##### Decision Threshold Optimization & Operating Points")
        st.markdown(
            """
            In an operational telecom NOC, the default classification threshold of $\\tau = 0.50$ is not necessarily optimal. 
            By adjusting the probability cutoff, network operations teams can tune the system to favor **higher recall** (zero missed outages at the cost of more false alerts) 
            or **higher precision** (zero false alarms at the cost of slight recall reduction).
            """
        )
        
        if not df_thresh.empty:
            fig_th = go.Figure()
            fig_th.add_trace(go.Scatter(
                x=df_thresh["Threshold"], y=df_thresh["RF_Recall"],
                mode='lines+markers', name='RF Recall', line=dict(color='#14B8A6', width=2.5)
            ))
            fig_th.add_trace(go.Scatter(
                x=df_thresh["Threshold"], y=df_thresh["RF_Precision"],
                mode='lines+markers', name='RF Precision', line=dict(color='#38BDF8', width=2.5)
            ))
            fig_th.add_trace(go.Scatter(
                x=df_thresh["Threshold"], y=df_thresh["RF_F1"],
                mode='lines+markers', name='RF F1-Score', line=dict(color='#F59E0B', width=2.5, dash='dash')
            ))
            
            # Highlight optimal threshold 0.30
            fig_th.add_vline(x=0.30, line_width=1.5, line_dash="dot", line_color="#EF4444", annotation_text="CV-Selected τ=0.30", annotation_position="top left")
            fig_th.add_vline(x=0.50, line_width=1.5, line_dash="dot", line_color="#94A3B8", annotation_text="Default τ=0.50", annotation_position="top right")
            
            fig_th.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#172235",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                height=380,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(title="Probability Decision Threshold (τ)", gridcolor="#1E293B"),
                yaxis=dict(title="Score", range=[0.65, 1.02], gridcolor="#1E293B"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_th, use_container_width=True)
            
            st.markdown(
                """
                - **Default Threshold ($\\tau = 0.50$)**: Precision: 100.00%, Recall: 93.12%, F1: 96.44% (17 missed anomalies, 0 false alarms).
                - **5-Fold CV Optimized ($\\tau = 0.30$)**: Precision: 99.17%, Recall: 97.17%, F1: 98.16% (Only 7 missed anomalies, 2 false alarms).
                - **Rigorous Methodological Note**: The threshold $\\tau=0.30$ was determined strictly via 5-Fold Cross-Validation on the training split ($X_{train}$), achieving a validation F1 of 97.08%, and subsequently evaluated on test telemetry without data leakage.
                """
            )

    with tab_paradigm:
        st.markdown("##### The Core Research Question: Supervised vs Unsupervised Trade-Offs")
        st.markdown(
            """
            Our research question specifically investigates:
            *How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data?*
            """
        )
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(
                """
                <div style="background-color: #172235; border-top: 3px solid #14B8A6; border-radius: 8px; padding: 16px;">
                    <h4 style="color: #14B8A6; margin-top: 0;">Supervised Classification (Random Forest & LR)</h4>
                    <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6; padding-left: 20px;">
                        <li><strong>Strengths</strong>: Extremely high precision (up to 100%) and recall (up to 97.2%). Distinguishes benign heavy-load spikes from genuine faults.</li>
                        <li><strong>Weaknesses</strong>: Requires costly, high-quality historical incident labels. Vulnerable to blind spots on entirely novel, zero-day failure modes.</li>
                        <li><strong>Operational Role</strong>: Primary automated triage engine for recurring, known incident archetypes (e.g., PRB exhaustion, coverage hole attenuation).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_p2:
            st.markdown(
                """
                <div style="background-color: #172235; border-top: 3px solid #38BDF8; border-radius: 8px; padding: 16px;">
                    <h4 style="color: #38BDF8; margin-top: 0;">Unsupervised Anomaly Detection (IF & LOF)</h4>
                    <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6; padding-left: 20px;">
                        <li><strong>Strengths</strong>: Requires zero ground-truth labels during training. Naturally flags statistical outliers and multivariate distribution shifts.</li>
                        <li><strong>Weaknesses</strong>: Moderate precision (~26–30%) on specific failure categories; elevated false positive alerts caused by rare normal traffic behaviors.</li>
                        <li><strong>Operational Role</strong>: Exploratory discovery buffer for zero-day network phenomena, misconfigurations, and novel anomalous patterns.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <strong style="color: #F8FAFC;">Hybrid Architecture Recommendation</strong>:
                A resilient enterprise network operations center should deploy a <strong>two-stage hybrid pipeline</strong>:
                Use Random Forest as the primary automated gate to rapidly resolve known high-confidence degradation patterns, 
                while running Isolation Forest concurrently in the background to capture novel distribution shifts and flag them for Tier-3 engineering review.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_latency:
        st.markdown("##### Computational Complexity & Production Serving Benchmarks")
        st.markdown(
            """
            Real-world telecom systems generate hundreds of thousands of cell-site KPI vectors per minute. 
            Production viability depends heavily on both training turnaround and real-time inference latency.
            """
        )
        
        col_lat1, col_lat2 = st.columns(2)
        with col_lat1:
            fig_fit = go.Figure(go.Bar(
                x=df_comp["Model Name"],
                y=df_comp["Training Fit Latency (s)"],
                marker_color="#38BDF8",
                text=df_comp["Training Fit Latency (s)"].apply(lambda v: f"{v:.2f}s"),
                textposition="auto"
            ))
            fig_fit.update_layout(
                title="Model Training Fit Time (N=25,600 traces)",
                paper_bgcolor="#111827",
                plot_bgcolor="#172235",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                height=300,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(title="Seconds (s)", gridcolor="#1E293B")
            )
            st.plotly_chart(fig_fit, use_container_width=True)
            
        with col_lat2:
            fig_inf = go.Figure(go.Bar(
                x=df_comp["Model Name"],
                y=df_comp["Inference Latency (ms/1k)"],
                marker_color="#14B8A6",
                text=df_comp["Inference Latency (ms/1k)"].apply(lambda v: f"{v:.1f}ms"),
                textposition="auto"
            ))
            fig_inf.update_layout(
                title="Batch Inference Latency (ms per 1,000 records)",
                paper_bgcolor="#111827",
                plot_bgcolor="#172235",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                height=300,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(title="Milliseconds (ms)", gridcolor="#1E293B", type="log")
            )
            st.plotly_chart(fig_inf, use_container_width=True)
            
        st.markdown(
            """
            > **Architectural Finding**: While Local Outlier Factor provides valuable density-based clustering, its inference latency (129.32 ms / 1k records) 
            is **~16x slower** than Random Forest (8.17 ms / 1k records) and **~760x slower** than Logistic Regression (0.17 ms / 1k records). 
            This is due to the $O(N_{train} \\times D)$ $k$-nearest-neighbor search required for each incoming test vector, making LOF impractical for high-throughput streaming telemetry without indexing approximations.
            """
        )
