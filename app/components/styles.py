"""
Custom Enterprise Dark Styling for Telecom AI Intelligence Platform.
Palette:
  Main Background: #0B1220
  Sidebar:         #111827
  Card Background: #172235
  Primary Accent:  #14B8A6 (Teal)
  Secondary Accent:#38BDF8 (Cyan/Sky)
  Text Primary:    #F8FAFC
  Text Secondary:  #94A3B8
  Normal:          #10B981
  Warning:         #F59E0B
  Critical Anomaly:#EF4444
"""

import streamlit as st


def apply_custom_theme():
    """Injects custom enterprise CSS for a dark NOC-style telecom interface."""
    st.markdown(
        """
        <style>
        /* Hide default Streamlit multipage sidebar navigation */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* App Background */
        .stApp {
            background-color: #0B1220;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }

        /* Main Header Banner */
        .telecom-hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #172235 60%, #0F2338 100%);
            border: 1px solid #1E293B;
            border-left: 5px solid #14B8A6;
            border-radius: 12px;
            padding: 22px 28px;
            margin-bottom: 22px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        }
        .telecom-hero-badge {
            display: inline-block;
            background-color: rgba(20, 184, 166, 0.15);
            color: #14B8A6;
            border: 1px solid rgba(20, 184, 166, 0.35);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .telecom-hero-banner h1 {
            color: #F8FAFC;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.3px;
        }
        .telecom-hero-banner p {
            color: #94A3B8;
            font-size: 0.95rem;
            margin-top: 6px;
            margin-bottom: 0;
            line-height: 1.5;
        }

        /* Metric Cards */
        .telecom-metric-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .telecom-metric-card:hover {
            border-color: #14B8A6;
            transform: translateY(-2px);
        }
        .telecom-metric-label {
            font-size: 0.76rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 4px;
        }
        .telecom-metric-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #14B8A6;
            line-height: 1.2;
        }
        .telecom-metric-subtext {
            font-size: 0.76rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Content Container Cards */
        .telecom-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
        }
        .telecom-card h3 {
            color: #F8FAFC;
            font-size: 1.12rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 10px;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 8px;
        }

        /* Stepper Card for Guided Workflows */
        .step-card {
            background-color: #172235;
            border: 1px solid #1E293B;
            border-top: 3px solid #14B8A6;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .step-badge {
            background-color: #14B8A6;
            color: #0B1220;
            font-weight: 800;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 8px;
            text-transform: uppercase;
        }

        /* Status & Severity Badges */
        .badge-normal {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .badge-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .badge-anomaly {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .badge-model {
            background-color: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.78rem;
        }

        /* Disclaimers & Alerts */
        .telecom-alert-disclaimer {
            background-color: rgba(245, 158, 11, 0.08);
            border: 1px solid #F59E0B;
            border-left: 4px solid #F59E0B;
            border-radius: 8px;
            padding: 12px 16px;
            color: #FDE68A;
            font-size: 0.86rem;
            margin: 12px 0 18px 0;
            line-height: 1.45;
        }

        .telecom-notice-box {
            background-color: rgba(56, 189, 248, 0.08);
            border: 1px solid #0284C7;
            border-left: 4px solid #38BDF8;
            border-radius: 8px;
            padding: 12px 16px;
            color: #BAE6FD;
            font-size: 0.86rem;
            margin: 12px 0 18px 0;
            line-height: 1.45;
        }

        /* Primary Buttons Styling */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
            color: #0B1220;
            font-weight: 700;
            font-size: 0.92rem;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            box-shadow: 0 3px 12px rgba(20, 184, 166, 0.25);
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 100%);
            color: #04101A;
            box-shadow: 0 4px 16px rgba(45, 212, 191, 0.4);
            transform: translateY(-1px);
        }

        /* Download Buttons Styling */
        div.stDownloadButton > button:first-child {
            background-color: #1E293B;
            color: #38BDF8;
            font-weight: 600;
            font-size: 0.88rem;
            border: 1px solid #38BDF8;
            border-radius: 8px;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }
        div.stDownloadButton > button:first-child:hover {
            background-color: rgba(56, 189, 248, 0.15);
            color: #F8FAFC;
            border-color: #14B8A6;
        }

        /* Table & Dataframe Dark Customization */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1E293B;
            border-radius: 8px;
            background-color: #172235;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Alias for backward compatibility across modules
apply_custom_styles = apply_custom_theme
