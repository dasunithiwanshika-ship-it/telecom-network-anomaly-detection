"""
Executive Enterprise Theme for Telecom AI Intelligence Platform.
Designed for professional Network Operations Centers (NOC).
Emphasizes clean typography, refined contrast, subtle geometric indicators,
and an elegant, distraction-free aesthetic.
"""

import streamlit as st


def apply_custom_theme():
    """Injects refined, professional NOC CSS styles into the Streamlit application."""
    st.markdown(
        """
        <style>
        /* Hide default Streamlit multi-page sidebar navigation */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Root Application Canvas */
        .stApp {
            background-color: #080C14 !important;
            background-image: 
                radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.04) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.03) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.02) 0px, transparent 50%) !important;
            color: #F1F5F9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            letter-spacing: -0.01em;
        }

        /* Main Container Geometry */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1360px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0B101B !important;
            border-right: 1px solid #162032 !important;
            box-shadow: 2px 0 16px rgba(0, 0, 0, 0.35);
        }

        /* Refined Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #080C14;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }

        /* Top Executive Header Bar */
        .noc-topbar {
            background: #0E1524;
            border: 1px solid #1A263D;
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Hero Banner */
        .telecom-hero-banner {
            background: linear-gradient(135deg, #0E1626 0%, #121D32 60%, #0F2236 100%);
            border: 1px solid #1C2D47;
            border-left: 4px solid #0EA5E9;
            border-radius: 8px;
            padding: 22px 26px;
            margin-bottom: 22px;
            position: relative;
        }
        .telecom-hero-badge {
            display: inline-block;
            background: rgba(14, 165, 233, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(14, 165, 233, 0.3);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 1px;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .telecom-hero-banner h1 {
            color: #F8FAFC;
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .telecom-hero-banner p {
            color: #94A3B8;
            font-size: 0.92rem;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.5;
        }

        /* NOC Status Pill Ribbon */
        .noc-pill-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }
        .noc-status-pill {
            background: #0E1524;
            border: 1px solid #1A263D;
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
            min-width: 140px;
            transition: border-color 0.15s ease;
        }
        .noc-status-pill:hover {
            border-color: #273857;
        }
        .noc-pill-label {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: #64748B;
            text-transform: uppercase;
        }
        .noc-pill-value {
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.1;
            font-feature-settings: "tnum";
        }

        /* Clean Architectural Cards */
        .telecom-card {
            background: #0E1524;
            border: 1px solid #1A263D;
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 18px;
        }
        .telecom-card h3 {
            color: #F8FAFC;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 10px;
            letter-spacing: -0.01em;
        }

        /* Metric Cards */
        .telecom-metric-card {
            background: #0E1524;
            border: 1px solid #1A263D;
            border-radius: 6px;
            padding: 16px 18px;
            margin-bottom: 14px;
        }
        .telecom-metric-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: #64748B;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .telecom-metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0EA5E9;
            line-height: 1.15;
            font-feature-settings: "tnum";
        }
        .telecom-metric-subtext {
            font-size: 0.76rem;
            color: #64748B;
            margin-top: 6px;
        }

        /* Status Dot Indicators */
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background-color: #10B981; }
        .dot-red { background-color: #EF4444; }
        .dot-amber { background-color: #F59E0B; }
        .dot-blue { background-color: #0EA5E9; }

        /* System Forecast Alert Banner */
        .noc-forecast-banner {
            background: rgba(239, 68, 68, 0.06);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-left: 3px solid #EF4444;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Refined Executive Buttons */
        div.stButton > button {
            background: #0284C7 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            border: 1px solid #0369A1 !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            letter-spacing: 0.2px !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button:hover {
            background: #0369A1 !important;
            border-color: #0284C7 !important;
            color: #FFFFFF !important;
        }

        /* Secondary Download Buttons */
        div.stDownloadButton > button {
            background: #0E1524 !important;
            color: #38BDF8 !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            border: 1px solid #1E293B !important;
            border-radius: 6px !important;
            padding: 8px 14px !important;
            transition: all 0.15s ease !important;
        }
        div.stDownloadButton > button:hover {
            background: #162238 !important;
            border-color: #38BDF8 !important;
            color: #FFFFFF !important;
        }

        /* Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1A263D !important;
            border-radius: 6px !important;
            background-color: #080C14 !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #64748B !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 14px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38BDF8 !important;
            border-bottom-color: #38BDF8 !important;
        }

        /* Form Inputs */
        div[data-baseweb="select"] > div {
            background-color: #0B101B !important;
            border-color: #1A263D !important;
            color: #F8FAFC !important;
            border-radius: 6px !important;
        }
        input[data-testid="stTextInput"] {
            background-color: #0B101B !important;
            border: 1px solid #1A263D !important;
            color: #F8FAFC !important;
            border-radius: 6px !important;
        }

        /* Badge Pills for Tables & Statuses */
        .pill-critical {
            background: rgba(239, 68, 68, 0.12);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-high {
            background: rgba(245, 158, 11, 0.12);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-normal {
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_styles = apply_custom_theme
