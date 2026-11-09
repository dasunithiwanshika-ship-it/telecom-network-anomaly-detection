"""
Interactive Plotly Charts for Telecom AI Intelligence Platform.
Styled uniformly for the dark NOC theme (#0B1220 / #172235).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_dark_chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply consistent enterprise dark styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif", size=11),
        margin=dict(t=35, b=35, l=40, r=25),
        height=height,
        legend=dict(
            bgcolor="rgba(23, 34, 53, 0.7)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#94A3B8"),
        ),
        xaxis=dict(
            gridcolor="#1E293B",
            zerolinecolor="#1E293B",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            zerolinecolor="#1E293B",
            tickfont=dict(color="#94A3B8"),
        ),
    )
    return fig


def create_donut_class_distribution(normal_count: int, anomaly_count: int) -> go.Figure:
    """Create a sleek donut chart showing normal vs anomaly distribution."""
    total = normal_count + anomaly_count
    normal_pct = (normal_count / total) * 100 if total > 0 else 0
    anomaly_pct = (anomaly_count / total) * 100 if total > 0 else 0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Normal Cell Traces", "Anomalous Traces"],
                values=[normal_count, anomaly_count],
                hole=0.62,
                marker=dict(
                    colors=["#10B981", "#EF4444"],
                    line=dict(color="#0B1220", width=2),
                ),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="#FFFFFF"),
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{anomaly_pct:.1f}%</b><br><span style='font-size:10px;color:#94A3B8'>Anomaly Rate</span>",
                x=0.5,
                y=0.5,
                font_size=16,
                font_color="#EF4444",
                showarrow=False,
            )
        ],
    )
    return apply_dark_chart_layout(fig, height=280)


def create_score_distribution_histogram(
    df: pd.DataFrame,
    score_col: str = "anomaly_score_or_prob",
    status_col: str = "anomaly_status",
    title: str = "Prediction Score Distribution",
    threshold: Optional[float] = None,
) -> go.Figure:
    """Create histogram of anomaly probabilities or scores."""
    fig = px.histogram(
        df,
        x=score_col,
        color=status_col,
        barmode="overlay",
        color_discrete_map={"Normal": "#10B981", "Anomaly": "#EF4444"},
        nbins=25,
        title=title,
    )
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Threshold (tau = {threshold:.2f})",
            annotation_position="top right",
            annotation_font_color="#F59E0B",
        )
    return apply_dark_chart_layout(fig, height=320)


def create_record_deviation_waterfall(dev_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of z-score deviations for an individual record."""
    df_sorted = dev_df.sort_values(by="Z-Score (Dev.)", ascending=True)
    colors = [
        "#EF4444" if z > 2.5 else ("#F59E0B" if z > 1.5 else ("#38BDF8" if z < -1.5 else "#10B981"))
        for z in df_sorted["Z-Score (Dev.)"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Z-Score (Dev.)"],
            y=df_sorted["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1E293B", width=1)),
            text=[f"{z:+.1f} σ" for z in df_sorted["Z-Score (Dev.)"]],
            textposition="auto",
        )
    )
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    fig.update_layout(
        title="Key Feature Deviations from Normal Population (Standard Deviations σ)",
        xaxis_title="Z-Score Deviation (Units of Normal σ)",
        yaxis_title=None,
    )
    return apply_dark_chart_layout(fig, height=max(260, len(dev_df) * 38))


def create_kpi_distribution_plot(
    data_df: pd.DataFrame, feature_col: str, class_col: str = "Class"
) -> go.Figure:
    """Create violin / boxplot comparing normal vs anomaly for a KPI."""
    fig = px.box(
        data_df,
        x=class_col,
        y=feature_col,
        color=class_col,
        color_discrete_map={"Normal": "#10B981", "Anomaly": "#EF4444", "Normal (0)": "#10B981", "Anomaly (1)": "#EF4444"},
        points="outliers",
        title=f"Statistical Distribution: {feature_col}",
    )
    return apply_dark_chart_layout(fig, height=320)


def create_model_comparison_bar(comp_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing precision, recall, and F1 across models."""
    df = comp_df.copy()
    for col in ["Test Precision", "Test Recall", "Test F1-Score"]:
        if col in df.columns:
            df[col + " (Float)"] = df[col].astype(str).str.rstrip("%").astype(float)

    chart_df = df.melt(
        id_vars=["Model Name"],
        value_vars=["Test Precision (Float)", "Test Recall (Float)", "Test F1-Score (Float)"],
        var_name="Metric",
        value_name="Score (%)",
    )
    chart_df["Metric"] = chart_df["Metric"].str.replace(" (Float)", "", regex=False)

    fig = px.bar(
        chart_df,
        x="Model Name",
        y="Score (%)",
        color="Metric",
        barmode="group",
        color_discrete_map={
            "Test Precision": "#38BDF8",
            "Test Recall": "#F59E0B",
            "Test F1-Score": "#14B8A6",
        },
        title="Comparative Model Performance on Held-Out Test Partition (N=6,400)",
    )
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return apply_dark_chart_layout(fig, height=340)
