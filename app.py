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
# Demo-only credentials (prototype authentication, NOT production security).
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

# Brand palette (shared by CSS, logo and header)
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
CRITICALITY_MULTIP
