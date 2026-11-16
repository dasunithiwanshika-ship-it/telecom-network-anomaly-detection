"""
Interactive Enterprise Plotly Charts for Telecom AI Intelligence Platform.
Styled uniformly for the Royal Violet cosmic theme (#000000 / #9312FF / #A020FF).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_dark_chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply consistent Royal Violet cosmic styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F7F7F8", family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=11),
        margin=dict(t=35, b=35, l=45, r=25),
        height=height,
        legend=dict(
            bgcolor="rgba(11, 9, 18, 0.85)",
            bordercolor="rgba(147, 18, 255, 0.25)",
            borderwidth=1,
            font=dict(color="#8B8988"),
        ),
        xaxis=dict(
            gridcolor="rgba(147, 18, 255, 0.12)",
            zerolinecolor="rgba(147, 18, 255, 0.15)",
            tickfont=dict(color="#8B8988"),
        ),
        yaxis=dict(
            gridcolor="rgba(147, 18, 255, 0.12)",
            zerolinecolor="rgba(147, 18, 255, 0.15)",
            tickfont=dict(color="#8B8988"),
        ),
    )
    return fig


def create_radial_gauge(
    value: float,
    title: str,
    color: str = "#A020FF",
    max_val: float = 100.0,
    suffix: str = "%",
) -> go.Figure:
    """Create a sleek circular radial gauge ring matching Royal Violet NOC monitors."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 24, "color": "#F7F7F8"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 11.5, "color": "#8B8988"}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "rgba(147,18,255,0.2)", "visible": False},
                "bar": {"color": color, "thickness": 0.26},
                "bgcolor": "rgba(18, 14, 28, 0.9)",
                "borderwidth": 1,
                "bordercolor": "rgba(147, 18, 255, 0.24)",
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
                    colors=["#45B558", "#9312FF"],
                    line=dict(color="#000000", width=2),
                ),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=11, color="#F7F7F8"),
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{anomaly_pct:.1f}%</b><br><span style='font-size:10px;color:#8B8988'>Incident Rate</span>",
                x=0.5,
                y=0.5,
                font_size=14,
                font_color="#D8B4FE",
                showarrow=False,
            )
        ],
    )
    return apply_dark_chart_layout(fig, height=260)


def create_royal_violet_area_chart() -> go.Figure:
    """Create the signature Royal Violet area dynamics chart for the hero dashboard."""
    days = [f"Day {i:02d}" for i in range(1, 15)]
    telemetry_volume = [142, 168, 154, 192, 218, 205, 246, 232, 278, 294, 265, 312, 338, 364]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=telemetry_volume,
            mode="lines+markers",
            name="Evaluated Windows",
            fill="tozeroy",
            fillcolor="rgba(147, 18, 255, 0.22)",
            line=dict(color="#A020FF", width=2.8, shape="spline"),
            marker=dict(size=6, color="#E9D5FF", line=dict(color="#9312FF", width=1.5)),
        )
    )
    fig.update_layout(
        title="Daily Dynamics of Evaluated Cell Telemetry",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Windows (x100)"),
        showlegend=False,
    )
    return apply_dark_chart_layout(fig, height=230)


def create_kpi_trend_splines() -> go.Figure:
    """Create dual spline curves for 24-hour KPI telemetry trends in Royal Violet."""
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
            fill="tozeroy",
            fillcolor="rgba(147, 18, 255, 0.16)",
            line=dict(color="#A020FF", width=2.6, shape="spline"),
            marker=dict(size=5, color="#E9D5FF"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=dl_bler_pct,
            mode="lines+markers",
            name="Downlink BLER (%)",
            yaxis="y2",
            line=dict(color="#F03B2A", width=2, shape="spline", dash="dot"),
            marker=dict(size=5, color="#F03B2A"),
        )
    )
    fig.update_layout(
        title="24-Hour Telemetry Dynamics: Downlink Throughput vs. BLER Error Trend",
        yaxis=dict(title="Throughput (Mbps)", gridcolor="rgba(147, 18, 255, 0.12)"),
        yaxis2=dict(title="BLER (%)", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#F03B2A")),
    )
    return apply_dark_chart_layout(fig, height=300)


def create_temporal_alert_spikes(df: pd.DataFrame) -> go.Figure:
    """Create timeline distribution spike chart for detected anomalies."""
    if "anomaly_score_or_prob" not in df.columns:
        return go.Figure()

    indices = list(range(1, len(df) + 1))
    scores = df["anomaly_score_or_prob"].values
    is_anom = (df["anomaly_prediction"] == 1).values if "anomaly_prediction" in df.columns else (scores > 0.3)

    colors = ["#F03B2A" if a else "#9312FF" for a in is_anom]

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
    fig.add_hline(y=0.30, line_dash="dash", line_color="#D8B4FE", annotation_text="Threshold (0.30)", annotation_position="top right")
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

    color_map = {
        "Critical Alert": "#F03B2A",
        "🔴 Critical Alert": "#F03B2A",
        "High Risk": "#F59E0B",
        "🟠 High Risk": "#F59E0B",
        "Moderate Risk": "#A020FF",
        "🟡 Moderate Risk": "#A020FF",
        "Normal": "#45B558",
        "🟢 Normal": "#45B558",
    }
    bar_colors = [color_map.get(k, "#9312FF") for k in counts.index]
    clean_labels = [k.replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").replace("🟢 ", "") for k in counts.index]

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=clean_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(147, 18, 255, 0.25)", width=1)),
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


def create_topology_network_graph(selected_node: Optional[str] = None) -> go.Figure:
    """Create an interactive network topology diagram showing Core, Switches, and Cell Sites."""
    nodes_x = [0.5, 0.25, 0.75, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
    nodes_y = [0.88, 0.58, 0.58, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]
    node_names = [
        "Core UPF (5GC)",
        "Agg-Switch-North",
        "Agg-Switch-South",
        "cell-macro-01",
        "cell-macro-02",
        "cell-small-03",
        "cell-macro-04",
        "cell-small-05",
        "cell-macro-06",
    ]
    node_status = [
        "Optimal (100% Availability)",
        "Optimal (10 Gbps Uplink)",
        "Optimal (10 Gbps Uplink)",
        "Optimal (42% PRB Load)",
        "Degraded (High BLER 12.4%)",
        "Optimal (35% PRB Load)",
        "Critical (Buffer Full 94%)",
        "Optimal (18% PRB Load)",
        "Optimal (45% PRB Load)",
    ]
    node_colors = [
        "#A020FF",
        "#7A1CF8",
        "#7A1CF8",
        "#45B558",
        "#F59E0B",
        "#45B558",
        "#F03B2A",
        "#45B558",
        "#45B558",
    ]

    node_sizes = [28 if (selected_node and n.lower() in selected_node.lower()) else 20 for n in node_names]
    border_widths = [3 if (selected_node and n.lower() in selected_node.lower()) else 1.5 for n in node_names]
    border_colors = ["#FFFFFF" if (selected_node and n.lower() in selected_node.lower()) else "#000000" for n in node_names]

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
            line=dict(color="rgba(147, 18, 255, 0.35)", width=1.6),
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
            textfont=dict(size=9.5, color="#F7F7F8"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color=border_colors, width=border_widths),
            ),
            hovertext=[f"Node: {n}<br>Role: {s}" for n, s in zip(node_names, node_status)],
            hoverinfo="text",
        )
    )

    fig.add_annotation(x=0.02, y=0.88, text="CORE LAYER", showarrow=False, font=dict(size=9, color="#8B8988"), xanchor="left")
    fig.add_annotation(x=0.02, y=0.58, text="AGGREGATION", showarrow=False, font=dict(size=9, color="#8B8988"), xanchor="left")
    fig.add_annotation(x=0.02, y=0.22, text="RADIO ACCESS", showarrow=False, font=dict(size=9, color="#8B8988"), xanchor="left")

    fig.update_layout(
        title="Interactive Radio Access Network Topology",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.05, 1.0]),
        showlegend=False,
    )
    return apply_dark_chart_layout(fig, height=320)


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
            line=dict(color="#45B558", width=1.6),
            fillcolor="rgba(69, 181, 88, 0.14)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=current_cell_vals + [current_cell_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Suspect Cell Trace",
            line=dict(color="#A020FF", width=2.2),
            fillcolor="rgba(160, 32, 255, 0.24)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#8B8988"), gridcolor="rgba(147, 18, 255, 0.18)"),
            angularaxis=dict(tickfont=dict(size=10, color="#F7F7F8"), gridcolor="rgba(147, 18, 255, 0.18)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Multi-Axis KPI Health Profile (Higher Index = Healthier)",
    )
    return apply_dark_chart_layout(fig, height=330)


def create_record_deviation_waterfall(dev_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of z-score deviations for an individual record."""
    df_sorted = dev_df.sort_values(by="Z-Score (Dev.)", ascending=True)
    colors = [
        "#F03B2A" if z > 2.5 else ("#F59E0B" if z > 1.5 else ("#A020FF" if z < -1.5 else "#45B558"))
        for z in df_sorted["Z-Score (Dev.)"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Z-Score (Dev.)"],
            y=df_sorted["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(147, 18, 255, 0.25)", width=1)),
            text=[f"{z:+.1f} σ" for z in df_sorted["Z-Score (Dev.)"]],
            textposition="auto",
        )
    )
    fig.add_vline(x=0, line_color="#8B8988", line_width=1.2)
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
        color_discrete_map={"Normal": "#45B558", "Anomaly": "#F03B2A", "Normal (0)": "#45B558", "Anomaly (1)": "#F03B2A"},
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
            "Test Precision": "#D8B4FE",
            "Test Recall": "#7A1CF8",
            "Test F1-Score": "#A020FF",
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
        color_discrete_map={"Normal": "#45B558", "Anomaly": "#F03B2A"},
        nbins=25,
        title=title,
    )
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#D8B4FE",
            annotation_text=f"Cutoff (tau = {threshold:.2f})",
            annotation_position="top right",
            annotation_font_color="#D8B4FE",
        )
    return apply_dark_chart_layout(fig, height=310)
