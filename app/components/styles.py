"""
Royal Violet Enterprise Theme for Telecom AI Intelligence Platform.
Inspired by pure-black cosmic aesthetics (#000000), glowing planetary violet rims (#9312FF),
violet-edged dark glass dashboard panels, and crisp Inter / Space Grotesk typography.
"""

import textwrap
import streamlit as st


def apply_custom_theme():
    """Injects the Royal Violet dark glass & cosmic CSS theme into Streamlit."""
    css_html = textwrap.dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

        /* Hide default Streamlit multi-page sidebar navigation */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Pure Black Cosmic Starfield Canvas */
        .stApp {
            background-color: #000000 !important;
            background-image:
                radial-gradient(1.5px 1.5px at 12% 18%, rgba(255, 255, 255, 0.55) 100%, transparent 0),
                radial-gradient(1px 1px at 28% 8%, rgba(255, 255, 255, 0.45) 100%, transparent 0),
                radial-gradient(1.5px 1.5px at 44% 24%, rgba(216, 180, 254, 0.5) 100%, transparent 0),
                radial-gradient(1px 1px at 63% 11%, rgba(255, 255, 255, 0.5) 100%, transparent 0),
                radial-gradient(1.5px 1.5px at 78% 22%, rgba(255, 255, 255, 0.45) 100%, transparent 0),
                radial-gradient(1px 1px at 91% 14%, rgba(216, 180, 254, 0.55) 100%, transparent 0),
                radial-gradient(1px 1px at 19% 41%, rgba(255, 255, 255, 0.3) 100%, transparent 0),
                radial-gradient(1.5px 1.5px at 84% 38%, rgba(255, 255, 255, 0.35) 100%, transparent 0),
                radial-gradient(ellipse 85% 42% at 50% -12%, rgba(147, 18, 255, 0.28) 0%, rgba(122, 28, 248, 0.08) 48%, transparent 78%) !important;
            color: #F7F7F8;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            letter-spacing: -0.01em;
        }

        h1, h2, h3, .display-font {
            font-family: 'Space Grotesk', 'Inter', -apple-system, sans-serif !important;
            letter-spacing: -0.025em;
            color: #F7F7F8;
        }

        /* Main Container Geometry */
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 3rem !important;
            max-width: 1340px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #050409 !important;
            border-right: 1px solid rgba(147, 18, 255, 0.18) !important;
            box-shadow: 4px 0 28px rgba(0, 0, 0, 0.85);
        }

        /* Subtle Scrollbars */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #000000; }
        ::-webkit-scrollbar-thumb { background: rgba(147, 18, 255, 0.35); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(160, 32, 255, 0.6); }

        /* Entrance Motion */
        @keyframes rvFadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }

        /* Violet Gradient Text */
        .violet-grad {
            background: linear-gradient(135deg, #E9D5FF 0%, #B76EFF 45%, #9312FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Top Executive Header Bar with Violet Edge Glow */
        .noc-topbar {
            background: rgba(10, 8, 16, 0.88);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(147, 18, 255, 0.24);
            border-top: 1px solid rgba(180, 100, 255, 0.55);
            border-radius: 14px;
            padding: 15px 22px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 -10px 30px -15px rgba(147, 18, 255, 0.35), 0 12px 32px rgba(0, 0, 0, 0.65);
            animation: rvFadeUp 0.45s ease-out both;
        }

        /* Hero Banner */
        .telecom-hero-banner {
            background: linear-gradient(180deg, rgba(16, 11, 26, 0.92) 0%, rgba(9, 7, 14, 0.95) 100%);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(147, 18, 255, 0.25);
            border-top: 1px solid rgba(186, 114, 255, 0.65);
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 22px;
            position: relative;
            box-shadow: 0 -12px 38px -18px rgba(147, 18, 255, 0.45), 0 14px 36px rgba(0, 0, 0, 0.7);
            animation: rvFadeUp 0.45s ease-out both;
        }
        .telecom-hero-badge {
            display: inline-block;
            background: rgba(147, 18, 255, 0.14);
            color: #D8B4FE;
            border: 1px solid rgba(147, 18, 255, 0.38);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 1px;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .telecom-hero-banner h1 {
            color: #F7F7F8;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.03em;
        }
        .telecom-hero-banner p {
            color: #8B8988;
            font-size: 0.94rem;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.55;
        }

        /* Cosmic Planet Rim Hero (Stage 01) */
        .rv-hero-stage {
            position: relative;
            background: #000000;
            border: 1px solid rgba(147, 18, 255, 0.22);
            border-radius: 16px;
            padding: 28px 32px 0 32px;
            margin-bottom: 24px;
            overflow: hidden;
            box-shadow: 0 24px 64px rgba(0, 0, 0, 0.9);
            animation: rvFadeUp 0.55s ease-out both;
        }
        .rv-planet-rim {
            position: relative;
            width: 100%;
            height: 110px;
            margin: -28px 0 -24px 0;
            overflow: hidden;
            pointer-events: none;
        }
        .rv-planet-arc {
            position: absolute;
            left: 50%;
            top: 18px;
            transform: translateX(-50%);
            width: 140%;
            height: 680px;
            border-radius: 50%;
            background: radial-gradient(circle at 50% 0%, rgba(160, 32, 255, 0.22) 0%, rgba(122, 28, 248, 0.05) 14%, #000000 32%);
            border-top: 2.5px solid #D8B4FE;
            box-shadow:
                0 -4px 24px rgba(160, 32, 255, 0.85),
                0 -16px 65px rgba(147, 18, 255, 0.65),
                inset 0 6px 32px rgba(160, 32, 255, 0.45);
        }

        /* Rising Violet-Edged Dashboard Frame */
        .rv-dashboard-frame {
            background: linear-gradient(180deg, rgba(13, 10, 22, 0.94) 0%, rgba(6, 5, 11, 0.98) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(147, 18, 255, 0.28);
            border-top: 1.5px solid #B76EFF;
            border-radius: 14px 14px 0 0;
            padding: 20px 24px;
            margin-top: 28px;
            box-shadow:
                0 -14px 44px -12px rgba(147, 18, 255, 0.5),
                inset 0 1px 0 rgba(233, 213, 255, 0.35);
        }

        /* NOC Status Pill Ribbon */
        .noc-pill-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }
        .noc-status-pill {
            background: rgba(11, 9, 18, 0.9);
            border: 1px solid rgba(147, 18, 255, 0.2);
            border-top: 1px solid rgba(160, 32, 255, 0.38);
            border-radius: 12px;
            padding: 11px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
            min-width: 145px;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .noc-status-pill:hover {
            border-color: rgba(160, 32, 255, 0.55);
            transform: translateY(-1px);
        }
        .noc-pill-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.6px;
            color: #8B8988;
            text-transform: uppercase;
        }
        .noc-pill-value {
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.15;
            font-feature-settings: "tnum";
        }

        /* Dark Glass Cards & Streamlit Bordered Containers */
        .telecom-card,
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background: rgba(11, 9, 18, 0.9) !important;
            backdrop-filter: blur(16px);
            border: 1px solid rgba(147, 18, 255, 0.22) !important;
            border-top: 1px solid rgba(180, 100, 255, 0.45) !important;
            border-radius: 14px !important;
            padding: 20px 22px !important;
            margin-bottom: 16px;
            box-shadow: 0 -8px 28px -16px rgba(147, 18, 255, 0.3), 0 10px 30px rgba(0, 0, 0, 0.55);
        }
        .telecom-card h3 {
            color: #F7F7F8;
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 10px;
            letter-spacing: -0.015em;
        }

        /* Metric Cards */
        .telecom-metric-card {
            background: rgba(11, 9, 18, 0.92);
            border: 1px solid rgba(147, 18, 255, 0.2);
            border-top: 1px solid rgba(160, 32, 255, 0.42);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        }
        .telecom-metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #8B8988;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .telecom-metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #F7F7F8;
            line-height: 1.15;
            font-feature-settings: "tnum";
        }
        .telecom-metric-subtext {
            font-size: 0.76rem;
            color: #8B8988;
            margin-top: 6px;
        }

        /* Status Dot Indicators */
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background-color: #45B558; box-shadow: 0 0 8px rgba(69, 181, 88, 0.65); }
        .dot-red { background-color: #F03B2A; box-shadow: 0 0 8px rgba(240, 59, 42, 0.65); }
        .dot-amber { background-color: #F59E0B; box-shadow: 0 0 8px rgba(245, 158, 11, 0.65); }
        .dot-blue { background-color: #A020FF; box-shadow: 0 0 8px rgba(160, 32, 255, 0.75); }

        /* System Forecast Alert Banner */
        .noc-forecast-banner {
            background: linear-gradient(90deg, rgba(240, 59, 42, 0.1) 0%, rgba(147, 18, 255, 0.08) 100%);
            border: 1px solid rgba(240, 59, 42, 0.3);
            border-left: 3px solid #F03B2A;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Royal Violet Pill Action Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #A020FF 0%, #7A1CF8 100%) !important;
            color: #F7F7F8 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            border: 1px solid rgba(216, 180, 254, 0.35) !important;
            border-radius: 999px !important;
            padding: 8px 18px !important;
            letter-spacing: 0.1px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 18px rgba(147, 18, 255, 0.35) !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #B145FF 0%, #8B34FF 100%) !important;
            border-color: rgba(233, 213, 255, 0.65) !important;
            box-shadow: 0 0 26px rgba(160, 32, 255, 0.65) !important;
            transform: translateY(-1px) !important;
            color: #FFFFFF !important;
        }

        /* Secondary Dark Glass Download Buttons */
        div.stDownloadButton > button {
            background: rgba(18, 14, 28, 0.92) !important;
            color: #E9D5FF !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            border: 1px solid rgba(147, 18, 255, 0.32) !important;
            border-radius: 999px !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
        }
        div.stDownloadButton > button:hover {
            background: rgba(147, 18, 255, 0.2) !important;
            border-color: #A020FF !important;
            box-shadow: 0 0 18px rgba(147, 18, 255, 0.4) !important;
            color: #FFFFFF !important;
        }

        /* Radio Pill Navigation */
        div[data-testid="stRadio"] > div { gap: 6px; }
        div[data-testid="stRadio"] label {
            background: rgba(14, 11, 22, 0.85);
            border: 1px solid rgba(147, 18, 255, 0.16);
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 0.86rem;
            transition: all 0.18s ease;
        }
        div[data-testid="stRadio"] label:hover {
            border-color: rgba(160, 32, 255, 0.55);
            background: rgba(147, 18, 255, 0.12);
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(147, 18, 255, 0.24) !important;
            border-radius: 12px !important;
            background-color: #07050C !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #8B8988 !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 14px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #D8B4FE !important;
            border-bottom-color: #9312FF !important;
        }

        /* Form Inputs & Selects */
        div[data-baseweb="select"] > div {
            background-color: #09070F !important;
            border-color: rgba(147, 18, 255, 0.25) !important;
            color: #F7F7F8 !important;
            border-radius: 10px !important;
        }
        input[data-testid="stTextInput"] {
            background-color: #09070F !important;
            border: 1px solid rgba(147, 18, 255, 0.25) !important;
            color: #F7F7F8 !important;
            border-radius: 10px !important;
        }

        /* Delta & Status Pills */
        .pill-critical {
            background: rgba(240, 59, 42, 0.14);
            color: #F03B2A;
            border: 1px solid rgba(240, 59, 42, 0.35);
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-high {
            background: rgba(245, 158, 11, 0.14);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.35);
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-normal {
            background: rgba(69, 181, 88, 0.14);
            color: #45B558;
            border: 1px solid rgba(69, 181, 88, 0.35);
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-violet {
            background: rgba(147, 18, 255, 0.16);
            color: #D8B4FE;
            border: 1px solid rgba(147, 18, 255, 0.38);
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
        }
        </style>
        """
    ).strip()
    css_html = "\n".join(line.strip() for line in css_html.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(css_html)
    else:
        st.markdown(css_html, unsafe_allow_html=True)


apply_custom_styles = apply_custom_theme
