"""
Interactive Enterprise Plotly Charts for Telecom AI Intelligence Platform.
Styled for a professional, high-contrast NOC dashboard.
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
        font=dict(color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11),
        margin=dict(t=35, b=35, l=45, r=25),
        height=height,
        legend=dict(
            bgcolor="rgba(14, 21, 36, 0.8)",
            bordercolor="#1A263D",
            borderwidth=1,
            font=dict(color="#94A3B8"),
        ),
        xaxis=dict(
            gridcolor="#162032",
            zerolinecolor="#162032",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="#162032",
            zerolinecolor="#162032",
            tickfont=dict(color="#94A3B8"),
        ),
    )
    return fig


def create_radial_gauge(
    value: float,
    title: str,
    color: str = "#0EA5E9",
    max_val: float = 100.0,
    suffix: str = "%",
) -> go.Figure:
    """Create a sleek circular radial gauge ring matching NOC monitors."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 24, "color": "#F8FAFC"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 12, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#162032", "visible": False},
                "bar": {"color": color, "thickness": 0.26},
                "bgcolor": "#101827",
                "borderwidth": 1,
                "bordercolor": "#1A263D",
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=185,
        margin=dict(t=25, b=10, l=20, r=20),
    )
    return fig


def create_donut_class_distribution(normal_count: int, anomaly_count: int) -> go.Figure:
    """Create a sleek donut chart showing normal vs anomaly distribution."""
    total = normal_count + anomaly_count
    anomaly_pct = (anomaly_count / total) * 100 if total > 0 else 0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Healthy Windows", "Anomalous Windows"],
                values=[normal_count, anomaly_count],
                hole=0.68,
                marker=dict(
                    colors=["#10B981", "#EF4444"],
                    line=dict(color="#080C14", width=2),
                ),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=11, color="#FFFFFF"),
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{anomaly_pct:.1f}%</b><br><span style='font-size:10px;color:#94A3B8'>Incident Rate</span>",
                x=0.5,
                y=0.5,
                font_size=14,
                font_color="#EF4444",
                showarrow=False,
            )
        ],
    )
    return apply_dark_chart_layout(fig, height=260)


def create_kpi_trend_splines() -> go.Figure:
    """Create dual spline curves for 24-hour KPI telemetry trends."""
    hours = [f"{h:02d}:00" for h in range(0, 24, 2)]
    throughput_mbps = [42.1, 38.5, 31.2, 28.4, 34.0, 56.8, 78.4, 92.1, 95.4, 88.0, 79.2, 54.6]
    dl_bler_pct = [0.8, 0.9, 1.1, 1.4, 1.2, 2.1, 3.8, 8.4, 12.2, 5.1, 2.4, 1.1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=throughput_mbps,
            mode="lines+markers",
            name="DL Throughput (Mbps)",
            line=dict(color="#0EA5E9", width=2.5, shape="spline"),
            marker=dict(size=5, color="#0EA5E9"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=dl_bler_pct,
            mode="lines+markers",
            name="Downlink BLER (%)",
            yaxis="y2",
            line=dict(color="#EF4444", width=2, shape="spline", dash="dot"),
            marker=dict(size=5, color="#EF4444"),
        )
    )
    fig.update_layout(
        title="24-Hour Telemetry Dynamics: Downlink Throughput vs. BLER Error Trend",
        yaxis=dict(title="Throughput (Mbps)", gridcolor="#162032"),
        yaxis2=dict(title="BLER (%)", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#EF4444")),
    )
    return apply_dark_chart_layout(fig, height=300)


def create_temporal_alert_spikes(df: pd.DataFrame) -> go.Figure:
    """Create timeline distribution spike chart for detected anomalies."""
    if "anomaly_score_or_prob" not in df.columns:
        return go.Figure()

    indices = list(range(1, len(df) + 1))
    scores = df["anomaly_score_or_prob"].values
    is_anom = (df["anomaly_prediction"] == 1).values if "anomaly_prediction" in df.columns else (scores > 0.3)

    colors = ["#EF4444" if a else "#10B981" for a in is_anom]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=indices,
            y=scores,
            marker_color=colors,
            name="Anomaly Score",
            hovertext=[f"Window {i}<br>Score: {s:.3f}<br>Status: {'ANOMALY' if a else 'NORMAL'}" for i, s, a in zip(indices, scores, is_anom)],
        )
    )
    fig.add_hline(y=0.30, line_dash="dash", line_color="#F59E0B", annotation_text="Threshold (0.30)", annotation_position="top right")
    fig.update_layout(
        title="Temporal Telemetry Spike Distribution (Anomaly Confidence by Window)",
        xaxis_title="Telemetry Observation Window Index",
        yaxis_title="Anomaly Confidence Score",
    )
    return apply_dark_chart_layout(fig, height=280)


def create_severity_breakdown_bars(df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart breaking down alert severity levels."""
    if "Severity Level" in df.columns:
        counts = df["Severity Level"].value_counts()
    else:
        counts = pd.Series({"Normal": len(df)})

    # Match both plain and emoji-prefixed labels
    color_map = {
        "Critical Alert": "#EF4444",
        "🔴 Critical Alert": "#EF4444",
        "High Risk": "#F59E0B",
        "🟠 High Risk": "#F59E0B",
        "Moderate Risk": "#EAB308",
        "🟡 Moderate Risk": "#EAB308",
        "Normal": "#10B981",
        "🟢 Normal": "#10B981",
    }
    bar_colors = [color_map.get(k, "#38BDF8") for k in counts.index]

    # Clean display labels for y-axis
    clean_labels = [k.replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").replace("🟢 ", "") for k in counts.index]

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=clean_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#1A263D", width=1)),
            text=counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Alert Severity Distribution",
        xaxis_title="Window Count",
        yaxis_title=None,
    )
    return apply_dark_chart_layout(fig, height=250)


def create_topology_network_graph() -> go.Figure:
    """Create an interactive network topology diagram showing Core, Switches, and Cell Sites."""
    nodes_x = [0.5, 0.25, 0.75, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
    nodes_y = [0.9, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    node_names = [
        "Core UPF (5GC)",
        "Agg-Switch-North",
        "Agg-Switch-South",
        "Cell-Macro-01",
        "Cell-Macro-02",
        "Cell-Small-03",
        "Cell-Macro-04",
        "Cell-Small-05",
        "Cell-Macro-06",
    ]
    node_status = [
        "Optimal (100%)",
        "Optimal (99%)",
        "Optimal (99%)",
        "Optimal",
        "Degraded (High BLER)",
        "Optimal",
        "Critical (Buffer Backlog)",
        "Optimal",
        "Optimal",
    ]
    node_colors = [
        "#0EA5E9",
        "#3B82F6",
        "#3B82F6",
        "#10B981",
        "#F59E0B",
        "#10B981",
        "#EF4444",
        "#10B981",
        "#10B981",
    ]

    edge_x = []
    edge_y = []
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8)]
    for u, v in edges:
        edge_x.extend([nodes_x[u], nodes_x[v], None])
        edge_y.extend([nodes_y[u], nodes_y[v], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#1A263D", width=1.5),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers+text",
            text=[f"<b>{name}</b>" for name in node_names],
            textposition="bottom center",
            textfont=dict(size=9.5, color="#94A3B8"),
            marker=dict(
                size=18,
                color=node_colors,
                line=dict(color="#0B101B", width=2),
            ),
            hovertext=[f"Node: {n}<br>Status: {s}" for n, s in zip(node_names, node_status)],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Infrastructure Topo-Map (Radio Access Network Hierarchy)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
    )
    return apply_dark_chart_layout(fig, height=300)


def create_kpi_radar_chart(
    cell_values: Dict[str, float],
    normal_bounds: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Create a 6-axis radar / spider chart comparing single cell against normal bounds."""
    categories = ["RSRP Coverage", "UL SNR Quality", "DL BLER Link", "UL BLER Link", "PRB Load", "Buffer Health"]
    baseline_vals = [85, 85, 95, 95, 80, 90]

    rsrp_score = min(100, max(10, int(cell_values.get("RSRP_mean", -95) + 140)))
    snr_score = min(100, max(10, int(cell_values.get("UL_SNR_mean", 12) * 5)))
    dl_bler_val = cell_values.get("DL_BLER_mean", 0.02)
    dl_bler_score = max(10, int(100 - dl_bler_val * 500))
    ul_bler_val = cell_values.get("UL_BLER_mean", 0.02)
    ul_bler_score = max(10, int(100 - ul_bler_val * 500))
    prb_score = min(100, max(15, int(100 - cell_values.get("PRB_Utilization_DL_mean", 40) * 0.8)))
    buf_score = max(15, int(100 - min(85, cell_values.get("Estimated_UL_Buffer_mean", 100) / 100)))

    current_cell_vals = [rsrp_score, snr_score, dl_bler_score, ul_bler_score, prb_score, buf_score]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=baseline_vals + [baseline_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Normal Baseline",
            line=dict(color="#10B981", width=1.5),
            fillcolor="rgba(16, 185, 129, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=current_cell_vals + [current_cell_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Suspect Cell Trace",
            line=dict(color="#EF4444", width=2),
            fillcolor="rgba(239, 68, 68, 0.20)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#64748B"), gridcolor="#162032"),
            angularaxis=dict(tickfont=dict(size=10, color="#CBD5E1"), gridcolor="#162032"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Multi-Axis KPI Health Profile (Higher Index = Healthier)",
    )
    return apply_dark_chart_layout(fig, height=330)


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
            marker=dict(color=colors, line=dict(color="#162032", width=1)),
            text=[f"{z:+.1f} σ" for z in df_sorted["Z-Score (Dev.)"]],
            textposition="auto",
        )
    )
    fig.add_vline(x=0, line_color="#64748B", line_width=1.2)
    fig.update_layout(
        title="Feature Deviations from Population Baseline (Units of σ)",
        xaxis_title="Z-Score Deviation (Normal σ)",
        yaxis_title=None,
    )
    return apply_dark_chart_layout(fig, height=max(250, len(dev_df) * 36))


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
        title=f"Distribution: {feature_col}",
    )
    return apply_dark_chart_layout(fig, height=310)


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
            "Test F1-Score": "#0EA5E9",
        },
        title="Comparative Model Performance on Held-Out Test Set (N=6,400)",
    )
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return apply_dark_chart_layout(fig, height=330)


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
            annotation_text=f"Cutoff (tau = {threshold:.2f})",
            annotation_position="top right",
            annotation_font_color="#F59E0B",
        )
    return apply_dark_chart_layout(fig, height=310)
