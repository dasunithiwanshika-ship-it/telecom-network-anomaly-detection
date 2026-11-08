"""
Page 5: AI Insights & Explainability (Explainable AI & Feature Intelligence)
Presents Gini Impurity (MDI) vs Permutation Feature Importance, KPI category attribution,
and statistical moment distributions with strict academic rigor.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.components.metric_cards import render_hero_banner, render_academic_disclaimer

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

KPI_CATEGORIES = {
    "Signal Quality": ["UL_SNR"],
    "Radio Quality": ["RSRP"],
    "Error Rates": ["UL_BLER", "DL_BLER"],
    "Modulation": ["UL_MCS", "DL_MCS"],
    "Scheduling & Allocation": ["UL_NPRB", "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL"],
    "Traffic & Throughput": ["TX_Bytes", "RX_Bytes", "UL_NumberOfPackets", "DL_NumberOfPackets"],
    "Buffer Health": ["Estimated_UL_Buffer"]
}

def get_feature_category(feat_name: str) -> str:
    for cat, kpis in KPI_CATEGORIES.items():
        for k in kpis:
            if feat_name.startswith(k + "_") or feat_name == k:
                return cat
    return "Other"

def get_feature_moment(feat_name: str) -> str:
    for m in ["mean", "std", "min", "max"]:
        if feat_name.endswith(f"_{m}"):
            return m.upper()
    return "OTHER"

@st.cache_data
def load_feature_importances():
    gini_path = RESULTS_DIR / "rf_gini_feature_importance.csv"
    perm_path = RESULTS_DIR / "rf_permutation_importance.csv"
    
    df_gini = pd.read_csv(gini_path) if gini_path.exists() else pd.DataFrame(columns=["Feature", "MDI_Importance"])
    df_perm = pd.read_csv(perm_path) if perm_path.exists() else pd.DataFrame(columns=["Feature", "Permutation_Mean", "Permutation_Std"])
    
    return df_gini, df_perm

def render(show_header: bool = True):
    if show_header:
        render_hero_banner(
            title="AI Explainability & Feature Intelligence",
            subtitle="Global model interpretation, feature attribution, and permutation importance on held-out test telemetry.",
            badge="EXPLAINABLE AI (XAI)"
        )
        render_academic_disclaimer()
    
    df_gini, df_perm = load_feature_importances()
    
    if df_gini.empty or df_perm.empty:
        st.error("Feature importance result artifacts not found in `results/`. Please verify experimental outputs.")
        return

    # Add category and moment metadata
    df_gini["Category"] = df_gini["Feature"].apply(get_feature_category)
    df_gini["Moment"] = df_gini["Feature"].apply(get_feature_moment)
    df_perm["Category"] = df_perm["Feature"].apply(get_feature_category)
    df_perm["Moment"] = df_perm["Feature"].apply(get_feature_moment)
    
    st.markdown("### 1. Global Feature Attribution: MDI vs Permutation")
    st.markdown(
        """
        In complex ensemble architectures like Random Forest, interpreting feature influence requires comparing **Gini Mean Decrease in Impurity (MDI)** 
        against **Permutation Feature Importance** evaluated on strictly unseen test telemetry.
        """
    )
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Top Gini MDI Driver</div>
                <div style="color: #38BDF8; font-size: 1.3rem; font-weight: 700; margin-top: 4px;">RSRP_mean (11.26%)</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Sub-carrier signal power captures continuous tree split purity.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_kpi2:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Top Permutation Driver</div>
                <div style="color: #14B8A6; font-size: 1.3rem; font-weight: 700; margin-top: 4px;">UL_NPRB_min (+1.88%)</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Shuffling drops test accuracy by 1.88% (critical bottleneck).</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_kpi3:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Dominant KPI Domain</div>
                <div style="color: #F8FAFC; font-size: 1.3rem; font-weight: 700; margin-top: 4px;">Scheduling & Signal</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Physical layer SNR and PRB starvation dominate anomaly onset.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    top_n = st.slider("Select Top N Features to Display", min_value=5, max_value=30, value=15, step=5)
    
    tab_perm, tab_gini, tab_comp = st.tabs([
        "Permutation Importance (Held-Out Test Data)", 
        "Gini Impurity Importance (MDI)", 
        "Category & Moment Aggregation"
    ])
    
    with tab_perm:
        st.markdown(
            """
            **Permutation Importance** measures the drop in model accuracy when values of a single feature are randomly shuffled in the test set. 
            Because it is calculated on unseen test telemetry without model retraining, it provides an **unbiased** measurement of functional feature dependency.
            """
        )
        top_perm = df_perm.sort_values(by="Permutation_Mean", ascending=True).tail(top_n)
        
        fig_perm = go.Figure(go.Bar(
            x=top_perm["Permutation_Mean"],
            y=top_perm["Feature"],
            orientation='h',
            marker=dict(
                color='#14B8A6',
                line=dict(color='#38BDF8', width=1)
            ),
            error_x=dict(
                type='data',
                array=top_perm["Permutation_Std"],
                visible=True,
                color='#F8FAFC'
            ),
            hovertemplate="<b>%{y}</b><br>Accuracy Drop: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>"
        ))
        fig_perm.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#172235",
            font=dict(color="#F8FAFC", family="Inter, sans-serif"),
            height=max(400, top_n * 22),
            margin=dict(l=160, r=30, t=30, b=40),
            xaxis=dict(title="Mean Accuracy Degradation upon Permutation (5 Repeats)", gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_perm, use_container_width=True)
        
        st.markdown(
            """
            > **Methodological Insight**: Notice that **`UL_NPRB_min`** (minimum number of allocated physical resource blocks in the uplink) ranks highest in permutation importance. 
            When PRB allocations drop to near-zero, active transmissions experience immediate queue starvation and packet drops, representing a definitive operational anomaly.
            """
        )

    with tab_gini:
        st.markdown(
            """
            **Mean Decrease in Impurity (Gini MDI)** aggregates the total variance or Gini reduction brought by each feature across all 100 decision trees. 
            Note that MDI tends to exhibit a **cardinality bias** toward continuous, high-entropy features (such as `RSRP_mean` and `UL_SNR_mean`).
            """
        )
        top_gini = df_gini.sort_values(by="MDI_Importance", ascending=True).tail(top_n)
        
        fig_gini = go.Figure(go.Bar(
            x=top_gini["MDI_Importance"],
            y=top_gini["Feature"],
            orientation='h',
            marker=dict(
                color='#38BDF8',
                line=dict(color='#14B8A6', width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Gini Importance: %{x:.4f}<extra></extra>"
        ))
        fig_gini.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#172235",
            font=dict(color="#F8FAFC", family="Inter, sans-serif"),
            height=max(400, top_n * 22),
            margin=dict(l=160, r=30, t=30, b=40),
            xaxis=dict(title="Gini Impurity Reduction (MDI)", gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_gini, use_container_width=True)

    with tab_comp:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("##### Total Importance by KPI Category")
            cat_agg = df_gini.groupby("Category")["MDI_Importance"].sum().reset_index().sort_values(by="MDI_Importance", ascending=False)
            
            fig_cat = go.Figure(go.Pie(
                labels=cat_agg["Category"],
                values=cat_agg["MDI_Importance"],
                hole=0.45,
                marker=dict(colors=["#14B8A6", "#38BDF8", "#6366F1", "#EC4899", "#F59E0B", "#10B981", "#8B5CF6"]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Total MDI: %{value:.3f}<br>Share: %{percent}<extra></extra>"
            ))
            fig_cat.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#172235",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=340,
                showlegend=False
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with col_c2:
            st.markdown("##### Total Importance by Statistical Moment")
            mom_agg = df_gini.groupby("Moment")["MDI_Importance"].sum().reset_index().sort_values(by="MDI_Importance", ascending=False)
            
            fig_mom = go.Figure(go.Bar(
                x=mom_agg["Moment"],
                y=mom_agg["MDI_Importance"],
                marker=dict(color=["#38BDF8", "#14B8A6", "#F59E0B", "#6366F1"]),
                hovertemplate="<b>%{x}</b><br>MDI Share: %{y:.4f}<extra></extra>"
            ))
            fig_mom.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#172235",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=20, b=40),
                height=340,
                xaxis=dict(title="Window Moment (Aggregation)", gridcolor="#1E293B"),
                yaxis=dict(title="Aggregated MDI", gridcolor="#1E293B")
            )
            st.plotly_chart(fig_mom, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 2. Scientific Boundary: Correlation vs Physical Causation")
    st.markdown(
        """
        <div style="background-color: #172235; border-left: 4px solid #F59E0B; padding: 16px; border-radius: 4px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #F59E0B; margin-bottom: 6px;">Scientific Rigor Warning: Statistical Anomaly ≠ Physical Cause</div>
            <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Feature importance algorithms identify which numerical telemetry variables most strongly separate anomalous windows from baseline conditions.
                However, <strong>statistical correlation does not establish root-cause physical mechanics</strong>. For example, a sharp drop in <code>UL_NPRB_min</code> 
                along with an elevated <code>UL_BLER_mean</code> indicates radio link starvation and high transmission re-tries. It does not definitively confirm whether 
                the root origin is physical fiber degradation, optical attenuation, antenna misalignment, atmospheric ducting, or a core network packet drop.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 3. Feature Importance Data Table")
    merged_imp = pd.merge(df_gini, df_perm, on=["Feature", "Category", "Moment"])
    merged_imp = merged_imp.sort_values(by="Permutation_Mean", ascending=False)
    
    display_table = merged_imp[[
        "Feature", "Category", "Moment", "MDI_Importance", "Permutation_Mean", "Permutation_Std"
    ]].copy()
    display_table.columns = ["Feature Name", "Domain Category", "Moment", "Gini MDI", "Permutation Mean", "Permutation Std"]
    
    st.dataframe(
        display_table.style.format({
            "Gini MDI": "{:.4f}",
            "Permutation Mean": "{:.4f}",
            "Permutation Std": "{:.4f}"
        }),
        use_container_width=True,
        height=300
    )
