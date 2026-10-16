"""
Anomalix+ Enterprise Hospital CMMS  --  Demo / Proof-of-Concept (v4: clean roles & branding)
===================================================================================
PROTOTYPE ONLY. All asset data, sensor-style readings, risk scores, priorities and
work orders are SYNTHETIC and held in memory (Streamlit session state). Nothing is
connected to real equipment, an EHR/HIS, or a database. The risk engine is a
transparent, rules-based simulation -- NOT a trained or validated ML model.
The FHIR output is illustrative JSON that has NOT been validated for conformance.

Run:  streamlit run app.py
"""
import base64
import hmac
import json
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants and configuration
# ----------------------------------------------------------------------------
DEMO_CREDENTIALS = {
    "Executive Management": {"user": "admin_exec", "pass": "exec2026"},
    "E-Health Management": {"user": "ehealth_mgr", "pass": "ehealth2026"},
    "Data Quality & Governance": {"user": "data_gov", "pass": "datagov2026"},
    "Technical Support & Applications": {"user": "tech_support", "pass": "support2026"},
    "IT Infrastructure & Biomed": {"user": "tech_biomed", "pass": "biomed2026"},
}

ROLE_FOCUS = {
    "Executive Management": "Strategic overview: which assets and departments need decisions first.",
    "E-Health Management": "Digital-service continuity: IT assets, exposed departments and interoperability readiness.",
    "Data Quality & Governance": "Asset-data completeness, validation results and traceability of every score.",
    "Technical Support & Applications": "Proactive alert queue, maintenance schedule and work-order status.",
    "IT Infrastructure & Biomed": "Operating conditions: temperature vs. band, error events, runtime and due dates.",
}

# Brand palette
BRAND = {"navy": "#0F172A", "blue": "#1E3A8A", "cyan": "#38BDF8", "amber": "#F59E0B", "light": "#E2E8F0"}

# Logo concept "Pulse Deviation"
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#0F172A"/>'
    '<polyline points="6,38 19,38 25,38 32,14 40,50 46,38 58,38" fill="none" stroke="#38BDF8" '
    'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="32" cy="14" r="4.5" fill="#F59E0B"/></svg>'
)

FACTORS = {
    "health": "Current health score",
    "thermal": "Temperature vs. asset-specific band",
    "errors": "Recent error events (30 days)",
    "runtime": "Runtime vs. rated service life",
    "maintenance": "Maintenance due date",
    "stress": "Simulated operating stress",
}

DEFAULT_WEIGHTS = {
    "Medical Equipment": {"health": 0.25, "thermal": 0.10, "errors": 0.20,
                          "runtime": 0.10, "maintenance": 0.20, "stress": 0.15},
    "IT Infrastructure": {"health": 0.20, "thermal": 0.20, "errors": 0.20,
                          "runtime": 0.10, "maintenance": 0.10, "stress": 0.20},
}
DEFAULT_LOW_MAX = 30
DEFAULT_HIGH_MIN = 55
MIN_COVERAGE = 0.50
ERROR_CAP = {"Medical Equipment": 10, "IT Infrastructure": 20}
CRITICALITY_MULTIPLIER = {"Critical": 1.25, "High": 1.10, "Medium": 1.00, "Low": 0.90}
MAINT_WARNING_DAYS = 14
WO_STATUSES = ["Open", "Assigned", "In Progress", "Awaiting Parts", "Completed"]
WO_PRIORITIES = ["Low", "Medium", "High", "Urgent"]
WO_TEAMS = ["Biomedical Engineering (demo)", "IT Infrastructure Team (demo)",
            "Vendor Service (demo)"]

TEXT_COLS = ["asset_id", "asset_name", "category", "location", "department",
             "criticality", "status", "vendor"]
NUM_COLS = ["health_score", "temperature_c", "temp_band_min", "temp_band_max", "temp_margin",
            "error_events_30d", "runtime_hours", "rated_hours", "operating_stress"]
CLIP_RANGES = {"health_score": (0, 100), "operating_stress": (0, 100),
               "error_events_30d": (0, 10_000), "runtime_hours": (0, 10_000_000)}
PLAUSIBLE_TEMP = (-80.0, 200.0)

LOCAL_CODE_SYSTEM = "urn:anomalix:demo:codes"
LOCAL_ID_SYSTEM = "urn:anomalix:demo:asset-id"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def to_float(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def fmt(value, digits=1, default="n/a", signed=False):
    v = to_float(value)
    if v is None:
        return default
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(value, default=None):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def csv_safe(df):
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v and v[0] in "=+-@\t\r" else v)
    return out


def logo_img(size=56):
    b64 = base64.b64encode(LOGO_SVG.encode("utf-8")).decode("ascii")
    return (f'<img alt="Anomalix+ logo" width="{size}" height="{size}" '
            f'src="data:image/svg+xml;base64,{b64}"/>')


def inject_css():
    st.markdown(
        f"""
        <style>
        div[data-testid="stMetric"] {{
            background: #F8FAFC; border: 1px solid {BRAND['light']};
            border-left: 4px solid {BRAND['cyan']};
            border-radius: 10px; padding: 12px 16px;
        }}
        .anx-header {{
            background: linear-gradient(135deg, {BRAND['navy']} 0%, {BRAND['blue']} 100%);
            padding: 18px 24px; border-radius: 10px; color: white; margin-bottom: 14px;
        }}
        .anx-brand {{ display: flex; align-items: center; gap: 16px; }}
        .anx-header h1 {{ margin: 0; font-size: 28px; color: white; }}
        .anx-header p  {{ margin: 4px 0 0 0; color: {BRAND['cyan']}; }}
        .anx-ar {{ direction: rtl; text-align: right; }}
        .anx-login-logo {{ text-align: center; margin-bottom: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sample data and validation
# ----------------------------------------------------------------------------
def build_sample_assets():
    today =
