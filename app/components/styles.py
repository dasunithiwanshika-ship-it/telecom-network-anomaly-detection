"""
Clean, Smooth Dark Theme for Telecom AI Platform.
Focused on simplicity, readability, and generous spacing.
"""

import streamlit as st


def apply_custom_theme():
    """Injects clean, minimal dark CSS for an intuitive user experience."""
    st.markdown(
        """
        <style>
        /* Hide default Streamlit multipage sidebar navigation */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Clean App Background & Typography */
        .stApp {
            background-color: #0B1220;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Reduce top whitespace for a smoother layout */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1E293B;
        }

        /* Clean Page Title Banner */
        .telecom-hero-banner {
            background: #172235;
            border: 1px solid #1E293B;
            border-left: 4px solid #14B8A6;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .telecom-hero-badge {
            display: inline-block;
            color: #14B8A6;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .telecom-hero-banner h1 {
            color: #F8FAFC;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0;
        }
        .telecom-hero-banner p {
            color: #94A3B8;
            font-size: 0.92rem;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.5;
        }

        /* Simple, Spacious Metric Cards */
        .telecom-metric-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 12px;
        }
        .telecom-metric-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .telecom-metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #14B8A6;
            line-height: 1.2;
        }
        .telecom-metric-subtext {
            font-size: 0.78rem;
            color: #64748B;
            margin-top: 6px;
        }

        /* Clean Content Cards */
        .telecom-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 16px;
        }
        .telecom-card h3 {
            color: #F8FAFC;
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 8px;
        }

        /* Stepper Card */
        .step-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .step-badge {
            background-color: rgba(20, 184, 166, 0.15);
            color: #14B8A6;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            margin-right: 8px;
        }

        /* Subtle Info & Disclaimer Boxes */
        .telecom-alert-disclaimer, .telecom-notice-box {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-left: 3px solid #38BDF8;
            border-radius: 8px;
            padding: 12px 16px;
            color: #94A3B8;
            font-size: 0.84rem;
            margin: 16px 0;
            line-height: 1.5;
        }

        /* Primary Action Buttons */
        div.stButton > button:first-child {
            background-color: #14B8A6;
            color: #0B1220;
            font-weight: 700;
            font-size: 0.92rem;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.15s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #2DD4BF;
            color: #0B1220;
        }

        /* Download Buttons */
        div.stDownloadButton > button:first-child {
            background-color: #172235;
            color: #38BDF8;
            font-weight: 600;
            font-size: 0.88rem;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px 16px;
        }
        div.stDownloadButton > button:first-child:hover {
            border-color: #38BDF8;
            color: #F8FAFC;
        }

        /* Clean Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1E293B;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_styles = apply_custom_theme
