"""
Anomalix+ Enterprise Hospital CMMS  --  Demo / Proof-of-Concept (v3: role workspaces + branding)
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
# The role only selects which "Role Workspace" lens opens first; it is NOT access control.
DEMO_CREDENTIALS = {
    "Executive Management": {"user": "admin_exec", "pass": "exec2026"},
    "E-Health Management": {"user": "ehealth_mgr", "pass": "ehealth2026"},
    "Data Quality & Governance": {"user": "data_gov", "pass": "datagov2026"},
    "Technical Support & Applications": {"user": "tech_support", "pass": "support2026"},
    "IT Infrastructure & Biomed": {"user": "tech_biomed", "pass": "biomed2026"},
    "Quality & Patient Safety": {"user": "quality_lead", "pass": "quality2026"},
}

ROLE_FOCUS = {
    "Executive Management": "Strategic overview: which assets and departments need decisions first.",
    "E-Health Management": "Digital-service continuity: IT assets, exposed departments and interoperability readiness.",
    "Data Quality & Governance": "Asset-data completeness, validation results and traceability of every score.",
    "Technical Support & Applications": "Proactive alert queue, maintenance schedule and work-order status.",
    "IT Infrastructure & Biomed": "Operating conditions: temperature vs. band, error events, runtime and due dates.",
    "Quality & Patient Safety": "Medical equipment needing review by biomedical engineering and clinical staff.",
}

# Brand palette (shared by CSS, logo and header)
BRAND = {"navy": "#0F172A", "blue": "#1E3A8A", "cyan": "#38BDF8", "amber": "#F59E0B", "light": "#E2E8F0"}

# Logo concept 1 "Pulse Deviation": a steady pulse line with one amber point that departs from the pattern.
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

# Default weights per asset category (re-normalised to sum to 1 at use).
DEFAULT_WEIGHTS = {
    "Medical Equipment": {"health": 0.25, "thermal": 0.10, "errors": 0.20,
                          "runtime": 0.10, "maintenance": 0.20, "stress": 0.15},
    "IT Infrastructure": {"health": 0.20, "thermal": 0.20, "errors": 0.20,
                          "runtime": 0.10, "maintenance": 0.10, "stress": 0.20},
}
DEFAULT_LOW_MAX = 30     # score <  30  -> Low
DEFAULT_HIGH_MIN = 55    # score >= 55  -> High; otherwise Moderate
MIN_COVERAGE = 0.50      # if valid inputs cover < 50% of total weight, no score is produced
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
PLAUSIBLE_TEMP = (-80.0, 200.0)   # readings outside this are treated as invalid

LOCAL_CODE_SYSTEM = "urn:anomalix:demo:codes"
LOCAL_ID_SYSTEM = "urn:anomalix:demo:asset-id"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def to_float(value, default=None):
    """Convert to a finite float, or return default (None/NaN/inf/strings/NA)."""
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
    """Safe number formatting (never raises on None/NaN/strings)."""
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
    """Neutralise spreadsheet-formula injection in text cells before CSV export."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v and v[0] in "=+-@\t\r" else v)
    return out


def logo_img(size=56):
    """Inline logo as a base64 data-URI <img> (no external files or services)."""
    b64 = base64.b64encode(LOGO_SVG.encode("utf-8")).decode("ascii")
    return (f'<img alt="Anomalix+ logo (concept)" width="{size}" height="{size}" '
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
# Sample data (synthetic, deterministic) and validation
# ----------------------------------------------------------------------------
def build_sample_assets():
    """Original 5 sample assets plus synthetic signals. Fully deterministic."""
    today = date.today()
    data = {
        "asset_id": ["AST-ICU-01", "AST-LAB-04", "AST-SRV-12", "AST-RAD-02", "AST-NET-05"],
        "asset_name": ["Ventilator Alpha X", "Centrifuge Pro 5000", "Core Database Server",
                       "MRI Scanner Unit", "Core Switch Cisco 9K"],
        "category": ["Medical Equipment", "Medical Equipment", "IT Infrastructure",
                     "Medical Equipment", "IT Infrastructure"],
        "location": ["Main Hospital Building", "Main Hospital Building", "Data Center Floor 2",
                     "Radiology Dept", "Server Room A"],
        "department": ["Intensive Care (ICU)", "Pathology Lab", "IT Operations",
                       "Radiology", "Network Operations"],
        "criticality": ["High", "Medium", "Critical", "High", "Critical"],
        "status": ["Operational", "Degraded", "Operational", "Maintenance", "Operational"],
        "health_score": [92.5, 74.0, 98.1, 61.2, 95.0],
        "vendor": ["MedTech Global", "LabCorp Systems", "Dell Enterprise",
                   "Siemens Health", "Cisco Systems"],
        # --- synthetic signals (illustrative only) ---
        "temperature_c": [38.0, 41.0, 25.0, 24.0, 33.0],
        "temp_band_min": [25.0, 20.0, 18.0, 18.0, 20.0],   # assumed normal band, per asset
        "temp_band_max": [45.0, 40.0, 27.0, 22.0, 40.0],
        "temp_margin": [15.0, 15.0, 10.0, 6.0, 15.0],      # deviation (deg C) that maxes the factor
        "error_events_30d": [1, 4, 0, 9, 3],
        "runtime_hours": [18500, 14200, 31000, 52000, 27000],
        "rated_hours": [40000, 20000, 50000, 80000, 60000],
        "operating_stress": [35, 62, 40, 71, 30],
        "maintenance_offset_days": [45, -12, 25, 8, 90],
    }
    df = pd.DataFrame(data)
    df["next_maintenance_due"] = [today + timedelta(days=int(d)) for d in df["maintenance_offset_days"]]
    return df.drop(columns=["maintenance_offset_days"])


def validate_assets(df):
    """
    Return (clean_df, warnings). Adds missing columns, trims text, drops rows with no/duplicate
    asset_id, converts non-finite numbers to missing, clips ranged values, and treats
    non-positive rated_hours / temp_margin and implausible temperatures as missing.
    """
    out = df.copy()
    warnings = []
    for col in TEXT_COLS + NUM_COLS + ["next_maintenance_due"]:
        if col not in out.columns:
            out[col] = np.nan
            warnings.append(f"Column '{col}' is missing; treated as missing values.")

    for col in TEXT_COLS:
        out[col] = out[col].map(lambda v: clean_text(v, ""))
    blank_ids = int((out["asset_id"] == "").sum())
    if blank_ids:
        warnings.append(f"{blank_ids} row(s) without asset_id were dropped.")
        out = out[out["asset_id"] != ""]
    dups = int(out["asset_id"].duplicated().sum())
    if dups:
        warnings.append(f"{dups} duplicate asset_id row(s) were dropped (first kept).")
        out = out[~out["asset_id"].duplicated()]
    for col in TEXT_COLS[1:]:
        out[col] = out[col].replace("", "Unknown")

    for col in NUM_COLS:
        coerced = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        bad = int(coerced.isna().sum())
        if col in CLIP_RANGES:
            lo, hi = CLIP_RANGES[col]
            bad += int(((coerced < lo) | (coerced > hi)).sum())
            coerced = coerced.clip(lo, hi)
        elif col in ("rated_hours", "temp_margin"):
            nonpos = int((coerced <= 0).sum())
            bad += nonpos
            coerced = coerced.where(~(coerced <= 0))
        elif col in ("temperature_c", "temp_band_min", "temp_band_max"):
            outside = int(((coerced < PLAUSIBLE_TEMP[0]) | (coerced > PLAUSIBLE_TEMP[1])).sum())
            bad += outside
            coerced = coerced.where(~((coerced < PLAUSIBLE_TEMP[0]) | (coerced > PLAUSIBLE_TEMP[1])))
        out[col] = coerced
        if bad:
            warnings.append(f"{bad} missing/invalid/out-of-range value(s) in '{col}' "
                            "were clipped or treated as missing.")

    dates = pd.to_datetime(out["next_maintenance_due"], errors="coerce")
    if int(dates.isna().sum()):
        warnings.append(f"{int(dates.isna().sum())} missing/invalid maintenance due date(s).")
    out["next_maintenance_due"] = dates.dt.date.where(dates.notna(), None)
    return out.reset_index(drop=True), warnings


# ----------------------------------------------------------------------------
# Risk engine (rules-based, explainable, deterministic)
# ----------------------------------------------------------------------------
def get_config():
    """Read current weights/thresholds from session state (defaults if unset/invalid)."""
    weights = {}
    for cat, defaults in DEFAULT_WEIGHTS.items():
        raw = {}
        for f in FACTORS:
            v = to_float(st.session_state.get(f"w_{cat}_{f}"), defaults[f])
            raw[f] = max(0.0, v)
        total = sum(raw.values())
        weights[cat] = {f: (v / total if total > 0 else defaults[f]) for f, v in raw.items()}
    low = to_float(st.session_state.get("cfg_low_max"), DEFAULT_LOW_MAX)
    high = to_float(st.session_state.get("cfg_high_min"), DEFAULT_HIGH_MIN)
    if high <= low:
        high = low + 1
    return {"weights": weights, "low_max": low, "high_min": high}


def days_until(due):
    """Days from today to the due date (negative = overdue); None if unusable."""
    try:
        if due is None or pd.isna(due):
            return None
        return (pd.Timestamp(due).date() - date.today()).days
    except (TypeError, ValueError, OverflowError):
        return None


def signals_from_row(row):
    """Extract raw scoring inputs from an asset row (dict-like)."""
    return {
        "health_score": row.get("health_score"),
        "temperature_c": row.get("temperature_c"),
        "temp_band_min": row.get("temp_band_min"),
        "temp_band_max": row.get("temp_band_max"),
        "temp_margin": row.get("temp_margin"),
        "error_events_30d": row.get("error_events_30d"),
        "runtime_hours": row.get("runtime_hours"),
        "rated_hours": row.get("rated_hours"),
        "operating_stress": row.get("operating_stress"),
        "days_to_maintenance": days_until(row.get("next_maintenance_due")),
    }


def sanitize_signals(raw):
    """Idempotent cleaning: every value is a finite float in a sane range, or None."""
    s = {k: to_float(raw.get(k)) for k in (
        "health_score", "temperature_c", "temp_band_min", "temp_band_max", "temp_margin",
        "error_events_30d", "runtime_hours", "rated_hours", "operating_stress",
        "days_to_maintenance")}
    for k in ("health_score", "operating_stress"):
        if s[k] is not None:
            s[k] = clamp(s[k], 0.0, 100.0)
    for k in ("error_events_30d", "runtime_hours"):
        if s[k] is not None and s[k] < 0:
            s[k] = None
    for k in ("rated_hours", "temp_margin"):
        if s[k] is not None and s[k] <= 0:
            s[k] = None
    if s["temperature_c"] is not None and not (PLAUSIBLE_TEMP[0] <= s["temperature_c"] <= PLAUSIBLE_TEMP[1]):
        s["temperature_c"] = None
    lo, hi = s["temp_band_min"], s["temp_band_max"]
    if lo is None or hi is None or lo > hi:     # inverted or incomplete band -> unusable
        s["temp_band_min"] = s["temp_band_max"] = None
    if s["days_to_maintenance"] is not None:
        s["days_to_maintenance"] = int(np.floor(s["days_to_maintenance"]))
    return s


def condition_label(health):
    """CURRENT condition (distinct from future risk)."""
    h = to_float(health)
    if h is None:
        return "Unknown"
    if h >= 85:
        return "Good"
    if h >= 70:
        return "Fair"
    return "Poor"


def normalise_criticality(value):
    text = clean_text(value, "")
    name = text.title() if text else ""
    return name if name in CRITICALITY_MULTIPLIER else None


def compute_risk(raw_signals, category, criticality, cfg):
    """
    Illustrative failure-risk score (0-100) = weighted average of factor sub-scores (0-1) x 100.
    Missing/invalid inputs are excluded and weights re-normalised; if valid inputs cover less
    than MIN_COVERAGE of the total weight, no score is produced ("Insufficient data").
    Criticality NEVER changes the risk score; it only scales the separate maintenance priority.
    """
    s = sanitize_signals(raw_signals)
    notes = []
    weights = cfg["weights"].get(category)
    if weights is None:
        weights = cfg["weights"]["Medical Equipment"]
        notes.append(f"Category '{category}' not recognised; medical-equipment weights used.")

    sub = {}
    h = s["health_score"]
    sub["health"] = (clip01((100 - h) / 50.0), f"{h:.1f}") if h is not None else None

    t, lo, hi, mg = s["temperature_c"], s["temp_band_min"], s["temp_band_max"], s["temp_margin"]
    if None not in (t, lo, hi, mg):
        deviation = max(t - hi, lo - t, 0.0)
        sub["thermal"] = (clip01(deviation / mg), f"{t:.1f} C (band {lo:.0f}-{hi:.0f})")
    else:
        sub["thermal"] = None

    e = s["error_events_30d"]
    cap = ERROR_CAP.get(category, 10)
    sub["errors"] = (clip01(e / cap), f"{e:.0f}") if e is not None else None

    r, rated = s["runtime_hours"], s["rated_hours"]
    sub["runtime"] = (clip01(r / rated), f"{r:,.0f} / {rated:,.0f} h") if (r is not None and rated) else None

    d = s["days_to_maintenance"]
    if d is None:
        sub["maintenance"] = None
    else:
        if d < 0:
            m = 0.6 + 0.4 * clip01(-d / 60.0)           # overdue: 0.6 -> 1.0 over 60 days
        elif d <= MAINT_WARNING_DAYS:
            m = 0.4 * (1 - d / MAINT_WARNING_DAYS)      # due soon: 0.4 -> 0 over 14 days
        else:
            m = 0.0
        sub["maintenance"] = (clip01(m), f"{d} days to due" if d >= 0 else f"{-d} days overdue")

    st_ = s["operating_stress"]
    sub["stress"] = (clip01(st_ / 100.0), f"{st_:.0f}/100") if st_ is not None else None

    available = {f: v for f, v in sub.items() if v is not None}
    missing = [FACTORS[f] for f, v in sub.items() if v is None]
    total_all = sum(weights.values())
    total_w = sum(weights[f] for f in available)
    coverage = (total_w / total_all) if total_all > 0 else 0.0

    rows, score = [], float("nan")
    if available and total_w > 0 and coverage >= MIN_COVERAGE:
        acc = 0.0
        for f, (subscore, raw) in available.items():
            eff_w = weights[f] / total_w
            pts = subscore * eff_w * 100
            acc += pts
            rows.append({"factor": FACTORS[f], "input": raw, "sub_score": round(subscore, 2),
                         "weight": round(eff_w, 3), "points": round(pts, 1)})
        score = round(float(clamp(acc, 0.0, 100.0)), 1)   # rounded BEFORE classification

    if np.isnan(score):
        level = "Insufficient data"
    elif score < cfg["low_max"]:
        level = "Low"
    elif score < cfg["high_min"]:
        level = "Moderate"
    else:
        level = "High"

    crit = normalise_criticality(criticality)
    if crit is None:
        notes.append(f"Criticality '{criticality}' not recognised; factor 1.00 used.")
    factor = CRITICALITY_MULTIPLIER.get(crit, 1.0)
    priority = round(float(min(100.0, score * factor)), 1) if not np.isnan(score) else float("nan")

    return {
        "score": score, "level": level, "priority": priority, "criticality_factor": factor,
        "contributions": pd.DataFrame(rows, columns=["factor", "input", "sub_score", "weight", "points"]),
        "missing": missing, "coverage": coverage, "notes": notes,
        "condition": condition_label(s["health_score"]),
    }


def explain(result):
    """Plain-language explanation derived from the actual contributions."""
    if result["level"] == "Insufficient data":
        return (f"Too few valid inputs (data coverage {result['coverage']:.0%}, minimum "
                f"{MIN_COVERAGE:.0%}); no score was produced.")
    c = result["contributions"]
    top = c.sort_values("points", ascending=False).head(3)
    top = top[top["points"] >= 1.0]
    if top.empty:
        text = "No factor contributes materially; all simulated signals are in a healthy range."
    else:
        text = "Largest contributors: " + "; ".join(
            f"{r.factor} ({r.points:.0f} pts)" for r in top.itertuples()) + "."
    if result["missing"]:
        text += (f" Missing inputs excluded (data coverage {result['coverage']:.0%}): "
                 + ", ".join(result["missing"]) + ".")
    for n in result["notes"]:
        text += " " + n
    return text


def recommend(result, category, criticality):
    """Suggested next step for demonstration. Never a clinical decision."""
    level = result["level"]
    if level == "Insufficient data":
        return "Collect the missing inputs before relying on this assessment."
    base = {
        "Low": "Continue routine preventive maintenance and monitoring.",
        "Moderate": "Schedule an inspection in the next planned maintenance window and review the top contributing signals.",
        "High": "Prioritise an inspection and open a work order for review.",
    }[level]
    crit = normalise_criticality(criticality)
    if category == "Medical Equipment" and level != "Low" and crit in ("Critical", "High"):
        base += " Review by qualified biomedical engineering and clinical personnel is required before any operational decision."
    elif category == "IT Infrastructure" and level == "High" and crit == "Critical":
        base += " Confirm failover/redundancy status with IT operations."
    return base


def suggest_wo_priority(priority_score):
    p = to_float(priority_score)
    if p is None:
        return "Medium"
    return "Urgent" if p >= 70 else "High" if p >= 55 else "Medium" if p >= 30 else "Low"


def score_dataframe(df, cfg):
    """Return a NEW DataFrame with risk columns added (input is not mutated)."""
    out = df.copy()
    cols = {"risk_score": [], "risk_level": [], "priority_score": [], "condition": [],
            "days_to_maintenance": []}
    for _, row in out.iterrows():
        sig = signals_from_row(row.to_dict())
        res = compute_risk(sig, row["category"], row["criticality"], cfg)
        cols["risk_score"].append(res["score"])
        cols["risk_level"].append(res["level"])
        cols["priority_score"].append(res["priority"])
        cols["condition"].append(res["condition"])
        cols["days_to_maintenance"].append(sig["days_to_maintenance"])
    out["risk_score"] = pd.Series(cols["risk_score"], index=out.index, dtype="float64")
    out["risk_level"] = pd.Series(cols["risk_level"], index=out.index, dtype="object")
    out["priority_score"] = pd.Series(cols["priority_score"], index=out.index, dtype="float64")
    out["condition"] = pd.Series(cols["condition"], index=out.index, dtype="object")
    out["days_to_maintenance"] = pd.array(cols["days_to_maintenance"], dtype="Int64") if len(out) else \
        pd.array([], dtype="Int64")
    return out


def rank_by_priority(df):
    """Sort by priority, then risk, then asset_id (stable tie-break; missing values last)."""
    return df.sort_values(["priority_score", "risk_score", "asset_id"],
                          ascending=[False, False, True], na_position="last")


# ----------------------------------------------------------------------------
# FHIR R4 demonstration (illustrative; structural self-check only)
# ----------------------------------------------------------------------------
FHIR_ID_RE = re.compile(r"^[A-Za-z0-9\-\.]{1,64}$")
URN_UUID_RE = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DEVICE_STATUS = {"active", "inactive", "entered-in-error", "unknown"}
DEVICEMETRIC_CATEGORY = {"measurement", "setting", "calculation", "unspecified"}
OBSERVATION_STATUS = {"registered", "preliminary", "final", "amended", "corrected",
                      "cancelled", "entered-in-error", "unknown"}
# Target resource types allowed by FHIR R4 for these reference elements
ALLOWED_REF_TARGETS = {
    ("Observation", "subject"): {"Patient", "Group", "Device", "Location"},
    ("Observation", "device"): {"Device", "DeviceMetric"},
    ("DeviceMetric", "source"): {"Device"},
}
REQUIRED_ELEMENTS = {"Device": [], "DeviceMetric": ["type", "category"],
                     "Observation": ["status", "code"]}


def _urn(asset_key, kind):
    """Deterministic, valid UUID URN (UUIDv5) for a resource within the demo bundle."""
    return "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:anomalix:demo:{asset_key}:{kind}"))


def build_fhir_bundle(row, generated_at):
    """
    Illustrative FHIR R4 collection Bundle: Device + DeviceMetric + Observations.
    References inside the bundle use the entries' urn:uuid fullUrls. Synthetic values are
    tagged with a LOCAL code. Output is NOT validated; see fhir_sanity_checks().
    """
    asset_id = clean_text(row.get("asset_id"), "unknown")
    rid = re.sub(r"[^A-Za-z0-9\-\.]", "-", asset_id.lower())[:40].strip("-") or "asset"
    meta = {"tag": [{"system": LOCAL_CODE_SYSTEM, "code": "synthetic-demo-data"}]}
    device_url = _urn(rid, "device")

    device = {
        "resourceType": "Device", "id": f"{rid}", "meta": meta,
        "identifier": [{"system": LOCAL_ID_SYSTEM, "value": asset_id}],
        # Demo mapping only: prototype 'Maintenance' status -> Device.status 'inactive'.
        "status": "inactive" if clean_text(row.get("status")) == "Maintenance" else "active",
        "type": {"text": clean_text(row.get("category"), "Unknown")},
    }
    vendor = clean_text(row.get("vendor"))
    if vendor and vendor != "Unknown":
        device["manufacturer"] = vendor
    name = clean_text(row.get("asset_name"))
    if name and name != "Unknown":
        device["deviceName"] = [{"name": name, "type": "user-friendly-name"}]
    loc = " / ".join(x for x in (clean_text(row.get("location")), clean_text(row.get("department")))
                     if x and x != "Unknown")
    if loc:
        device["location"] = {"display": loc}       # display-only; no Location resource is created
    entries = [(device_url, device)]

    temp = sanitize_signals({"temperature_c": row.get("temperature_c")})["temperature_c"]
    if temp is not None:
        metric_url = _urn(rid, "temp-metric")
        entries.append((metric_url, {
            "resourceType": "DeviceMetric", "id": f"{rid}-temp-metric", "meta": meta,
            "type": {"coding": [{"system": LOCAL_CODE_SYSTEM, "code": "simulated-temperature"}],
                     "text": "Simulated temperature (local code, not a standard terminology)"},
            "unit": {"coding": [{"system": "http://unitsofmeasure.org", "code": "Cel"}]},
            "source": {"reference": device_url},
            "category": "measurement",
        }))
        entries.append((_urn(rid, "temp-obs"), {
            "resourceType": "Observation", "id": f"{rid}-temp-obs", "meta": meta,
            "status": "preliminary",
            "code": {"coding": [{"system": LOCAL_CODE_SYSTEM, "code": "simulated-temperature"}],
                     "text": "Simulated temperature (demo)"},
            "subject": {"reference": device_url},
            "effectiveDateTime": generated_at,
            "device": {"reference": metric_url},
            "valueQuantity": {"value": temp, "unit": "C",
                              "system": "http://unitsofmeasure.org", "code": "Cel"},
        }))
    health = sanitize_signals({"health_score": row.get("health_score")})["health_score"]
    if health is not None:
        entries.append((_urn(rid, "health-obs"), {
            "resourceType": "Observation", "id": f"{rid}-health-obs", "meta": meta,
            "status": "preliminary",
            "code": {"coding": [{"system": LOCAL_CODE_SYSTEM, "code": "asset-health-score"}],
                     "text": "Anomalix demo asset health score (local code, not a standard clinical code)"},
            "subject": {"reference": device_url},
            "effectiveDateTime": generated_at,
            "valueQuantity": {"value": health, "unit": "points"},
        }))
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:anomalix:demo:{rid}:bundle")),
        "type": "collection",
        "timestamp": generated_at,
        "entry": [{"fullUrl": url, "resource": res} for url, res in entries],
    }


def _iter_references(node, path=""):
    """Yield (element_name, reference_string) for every Reference-like dict."""
    if isinstance(node, dict):
        for key, val in node.items():
            if key == "reference" and isinstance(val, str):
                yield path, val
            else:
                yield from _iter_references(val, key)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_references(item, path)


def fhir_sanity_checks(bundle):
    """
    Project-specific STRUCTURAL self-check (id format, urn:uuid validity, required elements,
    code-set values, reference targets, finite numbers). This is NOT FHIR conformance
    validation: it does not check full cardinality, data types, terminology bindings,
    profiles or invariants. Use an official validator for that.
    """
    checks = []
    entries = bundle.get("entry", [])
    checks.append(("Bundle", bundle.get("resourceType") == "Bundle" and bundle.get("type") == "collection",
                   "resourceType Bundle, type collection"))
    urls = [e.get("fullUrl") for e in entries]
    url_map = {}
    for e in entries:
        url_map[e.get("fullUrl")] = e.get("resource", {}).get("resourceType")
    checks.append(("Bundle fullUrls", all(isinstance(u, str) and URN_UUID_RE.match(u) for u in urls)
                   and len(set(urls)) == len(urls), "all unique, valid lowercase urn:uuid values"))
    for e in entries:
        res = e.get("resource", {})
        rt = res.get("resourceType", "?")
        label = f"{rt}/{res.get('id', '?')}"
        problems = []
        if not FHIR_ID_RE.match(str(res.get("id", ""))):
            problems.append("id format")
        problems += [f"missing {k}" for k in REQUIRED_ELEMENTS.get(rt, []) if k not in res]
        if rt == "Device" and res.get("status") not in DEVICE_STATUS:
            problems.append("Device.status value")
        if rt == "DeviceMetric" and res.get("category") not in DEVICEMETRIC_CATEGORY:
            problems.append("DeviceMetric.category value")
        if rt == "Observation":
            if res.get("status") not in OBSERVATION_STATUS:
                problems.append("Observation.status value")
            qty = res.get("valueQuantity")
            if qty is not None and to_float(qty.get("value")) is None:
                problems.append("valueQuantity.value not finite")
        for element, ref in _iter_references(res):
            allowed = ALLOWED_REF_TARGETS.get((rt, element))
            target = url_map.get(ref)
            if target is None:
                problems.append(f"{element} reference not found in bundle")
            elif allowed is not None and target not in allowed:
                problems.append(f"{element} -> {target} not an allowed R4 target")
        checks.append((label, not problems, "OK" if not problems else "; ".join(problems)))
    return checks


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def refresh_demo_data():
    clean, warns = validate_assets(build_sample_assets())
    st.session_state["assets_df"] = clean
    st.session_state["data_warnings"] = warns
    st.session_state["generated_at"] = utc_now_str()


def init_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("role", None)
    if "assets_df" not in st.session_state:
        refresh_demo_data()
    st.session_state.setdefault("generated_at", utc_now_str())
    st.session_state.setdefault("work_orders", [])
    st.session_state.setdefault("wo_counter", 0)


def do_logout():
    st.session_state["authenticated"] = False
    st.session_state["role"] = None


def _eq(a, b):
    """Constant-time comparison that is safe for non-ASCII input."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ----------------------------------------------------------------------------
# Login screen
# ----------------------------------------------------------------------------
def render_login():
    st.markdown(f"<div class='anx-login-logo'>{logo_img(72)}</div>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;color:{BRAND['navy']};'>Anomalix+ Enterprise CMMS</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748B;'>Prototype demo sign-in "
                "&nbsp;|&nbsp; <span dir='rtl'>تسجيل الدخول التجريبي</span></p>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.warning("Prototype-only authentication with shared demo credentials. "
                   "It is not production-grade authentication or role-based authorization.")
        with st.form("login_form"):
            role = st.selectbox("Department / role (display label only)", list(DEMO_CREDENTIALS.keys()))
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            exp = DEMO_CREDENTIALS[role]
            if _eq(user.strip(), exp["user"]) and _eq(pw, exp["pass"]):
                st.session_state["authenticated"] = True
                st.session_state["role"] = role
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        with st.expander("Demo credentials (for presentation use)"):
            st.table(pd.DataFrame(
                [{"Role": r, "Username": c["user"], "Password": c["pass"], "Workspace focus": ROLE_FOCUS[r]}
                 for r, c in DEMO_CREDENTIALS.items()]))


# ----------------------------------------------------------------------------
# Tab renderers
# ----------------------------------------------------------------------------
def render_dashboard(scored):
    st.subheader("Executive Dashboard")
    st.caption(f"Synthetic demo data generated at {st.session_state['generated_at']} (UTC). "
               "Risk scores and priorities are illustrative outputs of a rules-based simulation.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    due = scored["days_to_maintenance"].dropna()
    overdue = int((due < 0).sum())
    soon = int(((due >= 0) & (due <= MAINT_WARNING_DAYS)).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registered assets", len(scored))
    c2.metric("Average health score", fmt(scored["health_score"].mean()))
    c3.metric("High illustrative risk", int((scored["risk_level"] == "High").sum()))
    c4.metric("Maintenance overdue", overdue)
    c5.metric("Due within 14 days", soon)
    n_ins = int((scored["risk_level"] == "Insufficient data").sum())
    if n_ins:
        st.warning(f"{n_ins} asset(s) have too little valid data for a risk score.")
    st.caption("MTTR, uptime, failure rates and savings are intentionally not shown: "
               "the prototype has no real operational data to support them.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Assets by operational status**")
        st.bar_chart(scored["status"].value_counts())
    with right:
        st.markdown("**Assets by criticality**")
        order = [c for c in ["Critical", "High", "Medium", "Low"] if c in set(scored["criticality"])]
        other = [c for c in scored["criticality"].unique() if c not in order]
        st.bar_chart(scored["criticality"].value_counts().reindex(order + other))

    st.markdown("**Health score vs. illustrative risk score**")
    st.bar_chart(scored.set_index("asset_name")[["health_score", "risk_score"]])

    st.markdown("**Assets requiring review (ranked by illustrative maintenance priority)**")
    st.caption("Priority = illustrative risk score x criticality factor (capped at 100). "
               "Risk and priority are different numbers; ties are broken by risk score.")
    rank = rank_by_priority(scored)[
        ["asset_id", "asset_name", "department", "criticality", "condition", "health_score",
         "risk_score", "risk_level", "priority_score", "days_to_maintenance"]]
    st.dataframe(rank, hide_index=True)


def render_registry(scored, cfg):
    st.subheader("Enterprise Asset Registry")
    st.caption("In-memory sample records for demonstration. Changes are not persisted to any database.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    f1, f2, f3, f4 = st.columns(4)
    cat = f1.multiselect("Category", sorted(scored["category"].unique()),
                         default=sorted(scored["category"].unique()), key="f_cat")
    dep = f2.multiselect("Department", sorted(scored["department"].unique()),
                         default=sorted(scored["department"].unique()), key="f_dep")
    sta = f3.multiselect("Status", sorted(scored["status"].unique()),
                         default=sorted(scored["status"].unique()), key="f_sta")
    cri = f4.multiselect("Criticality", sorted(scored["criticality"].unique()),
                         default=sorted(scored["criticality"].unique()), key="f_cri")
    q = st.text_input("Search by asset ID or name", key="f_search").strip().lower()

    mask = (scored["category"].isin(cat) & scored["department"].isin(dep)
            & scored["status"].isin(sta) & scored["criticality"].isin(cri))
    if q:
        mask &= (scored["asset_id"].str.lower().str.contains(q, regex=False, na=False)
                 | scored["asset_name"].str.lower().str.contains(q, regex=False, na=False))
    filtered = scored[mask]
    if filtered.empty:
        st.info("No assets match the current filters.")
        return

    cols = ["asset_id", "asset_name", "category", "department", "location", "vendor",
            "criticality", "status", "health_score", "condition", "risk_score", "risk_level",
            "priority_score", "next_maintenance_due", "days_to_maintenance"]
    view = filtered[cols].copy()
    view["next_maintenance_due"] = view["next_maintenance_due"].map(
        lambda d: "" if d is None or pd.isna(d) else str(d))
    st.dataframe(view, hide_index=True)
    export = csv_safe(view).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Export filtered registry (CSV)", data=export,
                       file_name="Anomalix_Demo_Assets.csv", mime="text/csv", key="dl_csv")

    st.markdown("**Asset detail**")
    pick = st.selectbox("Select asset", filtered["asset_id"].tolist(), key="detail_asset")
    row = scored[scored["asset_id"] == pick].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Current condition", f"{row['condition']} ({fmt(row['health_score'])})")
    d2.metric("Illustrative failure risk", fmt(row["risk_score"]), row["risk_level"], delta_color="off")
    d3.metric("Illustrative priority", fmt(row["priority_score"]))
    d4.metric("Maintenance due", str(row["next_maintenance_due"]) if row["next_maintenance_due"] else "n/a")
    st.write(f"**Location:** {row['location']}  |  **Department:** {row['department']}  |  "
             f"**Vendor:** {row['vendor']}  |  **Criticality:** {row['criticality']}")
    res = compute_risk(signals_from_row(row.to_dict()), row["category"], row["criticality"], cfg)
    st.write(explain(res))
    st.dataframe(res["contributions"], hide_index=True)


SIM_SPECS = {   # key -> (label, min, max, step, is_int)
    "stress": ("Simulated operating stress (0-100)", 0, 100, 1, True),
    "errors": ("Simulated error events, last 30 days", 0, 60, 1, True),
    "days": ("Simulated days until maintenance due (negative = overdue)", -90, 180, 1, True),
}


def reset_sim(asset_id, keys, defaults):
    """Callback: restore all scenario sliders (and the saved copy) of one asset to its baseline."""
    for k, key in keys.items():
        st.session_state[key] = defaults[k]
    st.session_state.setdefault("sim_store", {})[asset_id] = dict(defaults)


def render_simulator(scored, cfg):
    st.subheader("Scenario Simulator")
    st.warning("All values below are SIMULATED. Moving a slider changes only this demo's calculation "
               "and never affects a real device.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    ids = scored["asset_id"].tolist()
    names = dict(zip(scored["asset_id"], scored["asset_name"]))
    asset_id = st.selectbox("Asset", ids, key="sim_asset",
                            format_func=lambda a: f"{a} - {names.get(a, '')}")
    row = scored[scored["asset_id"] == asset_id].iloc[0]
    base = sanitize_signals(signals_from_row(row.to_dict()))

    lo, hi = base["temp_band_min"], base["temp_band_max"]
    t_lo, t_hi = (lo - 15.0, hi + 30.0) if lo is not None else (-10.0, 80.0)
    mid = (lo + hi) / 2 if lo is not None else 25.0
    temp_default = clamp(round((base["temperature_c"] if base["temperature_c"] is not None else mid) * 2) / 2,
                         t_lo, t_hi)
    defaults = {
        "stress": int(clamp(round(base["operating_stress"] if base["operating_stress"] is not None else 50), 0, 100)),
        "errors": int(clamp(round(base["error_events_30d"] if base["error_events_30d"] is not None else 0), 0, 60)),
        "days": int(clamp(base["days_to_maintenance"] if base["days_to_maintenance"] is not None else 30, -90, 180)),
        "temp": float(temp_default),
    }
    keys = {k: f"sim_{k}_{asset_id}" for k in defaults}
    bounds = {"stress": (0, 100), "errors": (0, 60), "days": (-90, 180), "temp": (t_lo, t_hi)}
    # Slider widget state is dropped by Streamlit while a widget is not rendered (e.g. after
    # switching assets), so each asset's scenario is also kept in a plain session dict.
    store = st.session_state.setdefault("sim_store", {}).setdefault(asset_id, {})
    for k, key in keys.items():       # seed from the saved copy / baseline, clamp into current range
        raw = st.session_state[key] if key in st.session_state else store.get(k, defaults[k])
        v = clamp(to_float(raw, defaults[k]), *bounds[k])
        st.session_state[key] = float(v) if k == "temp" else int(round(v))

    s1, s2 = st.columns(2)
    with s1:
        st.slider(SIM_SPECS["stress"][0], 0, 100, key=keys["stress"])
        st.slider(SIM_SPECS["errors"][0], 0, 60, key=keys["errors"])
    with s2:
        st.slider("Simulated temperature reading (C)", float(t_lo), float(t_hi), step=0.5, key=keys["temp"])
        st.slider(SIM_SPECS["days"][0], -90, 180, key=keys["days"])
    st.button("Reset scenario", on_click=reset_sim, args=(asset_id, keys, defaults),
              key=f"sim_reset_{asset_id}")
    for k, key in keys.items():       # save current values for this asset
        store[k] = st.session_state[key]

    scenario = dict(base)
    scenario.update({
        "operating_stress": float(st.session_state[keys["stress"]]),
        "error_events_30d": float(st.session_state[keys["errors"]]),
        "temperature_c": float(st.session_state[keys["temp"]]),
        "days_to_maintenance": int(st.session_state[keys["days"]]),
    })
    before = compute_risk(base, row["category"], row["criticality"], cfg)
    after = compute_risk(scenario, row["category"], row["criticality"], cfg)

    supplied = [lbl for lbl, k in (("temperature", "temperature_c"), ("stress", "operating_stress"),
                                   ("error events", "error_events_30d"),
                                   ("maintenance date", "days_to_maintenance")) if base[k] is None]
    if supplied:
        st.info("Baseline had no valid " + ", ".join(supplied) + "; the slider default is a placeholder, "
                "so baseline and scenario may differ in data coverage.")

    delta = None if np.isnan(after["score"]) or np.isnan(before["score"]) else \
        fmt(after["score"] - before["score"], signed=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline risk (simulated)", fmt(before["score"]), before["level"], delta_color="off")
    m2.metric("Scenario risk (simulated)", fmt(after["score"]), delta, delta_color="inverse")
    m3.metric("Scenario category", after["level"])
    st.write(f"**Current health condition:** {after['condition']} ({fmt(base['health_score'])}) "
             "- unchanged by the sliders; future-failure risk is a separate, illustrative estimate.")
    st.write("**Explanation:** " + explain(after))
    st.info("**Suggested next step:** " + recommend(after, row["category"], row["criticality"]))
    st.caption(f"Illustrative maintenance priority (risk x criticality factor "
               f"{after['criticality_factor']:.2f}): baseline {fmt(before['priority'])} -> "
               f"scenario {fmt(after['priority'])}.")

    cmp_df = pd.DataFrame({
        "Baseline": before["contributions"].set_index("factor")["points"],
        "Scenario": after["contributions"].set_index("factor")["points"],
    }).fillna(0.0)
    if not cmp_df.empty:
        st.markdown("**Risk points by factor (baseline vs. scenario)**")
        st.bar_chart(cmp_df)
    st.dataframe(after["contributions"], hide_index=True)


def create_work_order(asset_id, asset_name, priority, team, status, notes, risk):
    st.session_state["wo_counter"] += 1
    st.session_state["work_orders"].append({
        "work_order_id": f"WO-{st.session_state['wo_counter']:03d}",
        "asset_id": asset_id, "asset_name": asset_name, "priority": priority,
        "team": team, "status": status, "notes": notes[:500],
        "risk_at_creation (simulated)": None if risk is None or np.isnan(risk) else round(float(risk), 1),
        "created_utc": utc_now_str(),
    })


def apply_wo_status():
    """Callback for the 'Update status' button (runs before the next rerun)."""
    wid = st.session_state.get("wo_update_id")
    new = st.session_state.get("wo_update_status")
    if new not in WO_STATUSES:
        st.session_state["wo_flash"] = ("error", "Invalid status.")
        return
    for o in st.session_state["work_orders"]:
        if o["work_order_id"] == wid:
            o["status"] = new
            st.session_state["wo_flash"] = ("success", f"{wid} set to '{new}' (demo record).")
            return
    st.session_state["wo_flash"] = ("error", "Work order not found.")


def render_impact(scored, cfg):
    st.subheader("Impact Analysis & Maintenance Workflow")
    st.caption("Illustrative maintenance priority = illustrative risk score x criticality factor "
               "(Critical 1.25, High 1.10, Medium 1.00, Low 0.90), capped at 100. Criticality never "
               "changes the risk score itself. Both numbers are demonstration values.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    ranked = rank_by_priority(scored)
    review = ranked[ranked["risk_level"].isin(["Moderate", "High", "Insufficient data"])
                    | (ranked["condition"] == "Poor")]
    if review.empty:
        st.success("No assets currently flagged for review by the simulated rules.")
    for _, row in review.iterrows():
        res = compute_risk(signals_from_row(row.to_dict()), row["category"], row["criticality"], cfg)
        with st.expander(f"{row['asset_name']} ({row['asset_id']}) - {row['department']} | "
                         f"risk {fmt(row['risk_score'])} ({row['risk_level']}) | "
                         f"health {fmt(row['health_score'])} ({row['condition']}) | {row['criticality']}"):
            st.write("**Why:** " + explain(res))
            st.write("**Suggested action:** " + recommend(res, row["category"], row["criticality"]))
            st.write(f"Illustrative priority: {fmt(row['priority_score'])}  |  Vendor for reference: {row['vendor']}")

    st.markdown("---")
    st.markdown("### Simulated work-order workflow")
    st.caption("Prototype records held in session memory only. They are not saved to a database "
               "and are cleared when the browser session ends.")
    flash = st.session_state.pop("wo_flash", None)
    if flash:
        (st.success if flash[0] == "success" else st.error)(flash[1])

    wo_asset = st.selectbox("Asset for new work order", ranked["asset_id"].tolist(), key="wo_asset")
    arow = scored[scored["asset_id"] == wo_asset].iloc[0]
    suggested = suggest_wo_priority(arow["priority_score"])
    suggested_team = WO_TEAMS[0] if arow["category"] == "Medical Equipment" else WO_TEAMS[1]
    st.caption(f"Suggested priority from the simulation: {suggested}")
    with st.form(f"wo_form_{wo_asset}"):
        pr = st.selectbox("Priority", WO_PRIORITIES, index=WO_PRIORITIES.index(suggested),
                          key=f"wo_pr_{wo_asset}_{suggested}")
        team = st.selectbox("Assigned team (demo)", WO_TEAMS, index=WO_TEAMS.index(suggested_team),
                            key=f"wo_team_{wo_asset}")
        stat = st.selectbox("Initial status (simulated)", WO_STATUSES[:2], key=f"wo_stat_{wo_asset}")
        notes = st.text_area("Notes", value="Inspection requested based on simulated risk score.",
                             max_chars=500, key=f"wo_notes_{wo_asset}")
        if st.form_submit_button("Create demo work order"):
            create_work_order(wo_asset, arow["asset_name"], pr, team, stat, notes.strip(), arow["risk_score"])
            # Rerun so other tabs (e.g. Role Workspace, rendered earlier in the script) show the new record.
            st.session_state["wo_flash"] = ("success", "Demo work order created (in-memory prototype record).")
            st.rerun()

    orders = st.session_state["work_orders"]
    if not orders:
        st.info("No demo work orders yet.")
        return
    odf = pd.DataFrame(orders)
    st.markdown("**Open work orders**")
    open_df = odf[odf["status"] != "Completed"]
    if open_df.empty:
        st.info("All demo work orders are completed.")
    else:
        st.dataframe(open_df, hide_index=True)

    status_of = dict(zip(odf["work_order_id"], odf["status"]))
    u1, u2 = st.columns(2)
    u1.selectbox("Work order", odf["work_order_id"].tolist(), key="wo_update_id",
                 format_func=lambda w: f"{w} (currently {status_of.get(w, '?')})")
    u2.selectbox("New status", WO_STATUSES, key="wo_update_status")
    st.button("Update status", on_click=apply_wo_status, key="wo_update_btn")
    with st.expander("All work orders (including completed)"):
        st.dataframe(odf, hide_index=True)


def render_fhir(scored):
    st.subheader("HL7 FHIR Interoperability Demo (illustrative)")
    st.warning("Illustrative FHIR R4-style JSON. It has NOT been validated against the FHIR specification "
               "or any profile, and nothing is sent to an EHR/HIS. Local code systems are placeholders "
               "and the demo status mapping is an assumption.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    pick = st.selectbox("Select asset", scored["asset_id"].tolist(), key="fhir_asset")
    row = scored[scored["asset_id"] == pick].iloc[0]
    bundle = build_fhir_bundle(row.to_dict(), st.session_state["generated_at"])
    st.markdown("**FHIR R4-style Bundle (Device + DeviceMetric + Observation)**")
    st.json(bundle)
    st.download_button("Download JSON", data=json.dumps(bundle, indent=2, allow_nan=False).encode("utf-8"),
                       file_name=f"{pick}_fhir_demo.json", mime="application/json", key="dl_fhir")

    risk = to_float(row["risk_score"])
    prio = to_float(row["priority_score"])
    st.markdown("**Anomalix-specific output (proprietary, NOT FHIR)**")
    st.json({"asset_id": pick, "illustrative_risk_score": risk, "risk_level": row["risk_level"],
             "illustrative_priority_score": prio, "method": "rules-based simulation",
             "note": "Kept outside FHIR resources deliberately."})

    with st.expander("Structural self-check (NOT FHIR conformance validation)"):
        st.caption("Checks id format, urn:uuid fullUrls, required elements, a few code values, and "
                   "reference targets. It does not check full cardinality, data types, terminology "
                   "bindings, profiles or invariants.")
        for name, ok, msg in fhir_sanity_checks(bundle):
            st.write(f"{'PASS' if ok else 'FAIL'} - {name}: {msg}")
    with st.expander("What a real integration would still need"):
        st.markdown(
            "- Validation with an official FHIR validator against R4 and the target profiles\n"
            "- Agreed terminology (device and measurement codes) instead of local placeholder codes\n"
            "- A FHIR server or interface engine, authentication (e.g., OAuth 2.0 / SMART), and audit logging\n"
            "- Real data feeds from devices or monitoring systems, with data-quality checks\n"
            "- Security, privacy and regulatory review by the hospital (including applicable Saudi requirements)")


# ----------------------------------------------------------------------------
# Role workspaces (stakeholder lenses over the same synthetic data)
# ----------------------------------------------------------------------------
REVIEW_COLS = ["asset_id", "asset_name", "department", "criticality", "condition",
               "risk_score", "risk_level", "priority_score", "days_to_maintenance"]


def thermal_status(row):
    """Compare the (simulated) temperature with the asset's assumed band."""
    s = sanitize_signals(signals_from_row(row.to_dict()))
    t, lo, hi = s["temperature_c"], s["temp_band_min"], s["temp_band_max"]
    if t is None or lo is None or hi is None:
        return "n/a"
    if t > hi:
        return f"Above band (+{t - hi:.1f} C)"
    if t < lo:
        return f"Below band (-{lo - t:.1f} C)"
    return "Within band"


def runtime_used_pct(row):
    s = sanitize_signals(signals_from_row(row.to_dict()))
    if s["runtime_hours"] is None or s["rated_hours"] is None:
        return float("nan")
    return round(100.0 * s["runtime_hours"] / s["rated_hours"], 1)


def suggested_step(row, cfg):
    res = compute_risk(signals_from_row(row.to_dict()), row["category"], row["criticality"], cfg)
    return recommend(res, row["category"], row["criticality"])


def _maintenance_counts(df):
    due = df["days_to_maintenance"].dropna()
    return int((due < 0).sum()), int(((due >= 0) & (due <= MAINT_WARNING_DAYS)).sum())


def view_executive(scored, cfg):
    overdue, soon = _maintenance_counts(scored)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered assets", len(scored))
    c2.metric("High illustrative risk", int((scored["risk_level"] == "High").sum()))
    c3.metric("Maintenance overdue / due in 14 d", f"{overdue} / {soon}")
    c4.metric("Average health score", fmt(scored["health_score"].mean()))
    st.markdown("**Top assets to review first (illustrative priority)**")
    st.dataframe(rank_by_priority(scored).head(3)[REVIEW_COLS], hide_index=True)
    exposed = sorted(scored.loc[scored["risk_level"].isin(["Moderate", "High"]), "department"].unique())
    st.write("**Departments with Moderate/High illustrative-risk assets:** "
             + (", ".join(exposed) if exposed else "none in the current demo data"))
    st.markdown("**Illustrative maintenance priority by asset**")
    st.bar_chart(scored.set_index("asset_name")["priority_score"])
    st.caption("Priority = risk score x criticality factor. Costs and savings are not shown: "
               "the prototype has no real financial or operational data.")


def view_ehealth(scored, cfg):
    it = scored[scored["category"] == "IT Infrastructure"]
    watch = scored[scored["criticality"].isin(["Critical", "High"])
                   & (scored["risk_level"].isin(["Moderate", "High", "Insufficient data"])
                      | (scored["days_to_maintenance"].fillna(10_000) < 0))]
    c1, c2, c3 = st.columns(3)
    c1.metric("IT infrastructure assets", len(it))
    c2.metric("Critical IT assets", int((it["criticality"] == "Critical").sum()))
    c3.metric("Service-continuity watch-list", len(watch))
    st.markdown("**IT infrastructure assets**")
    if it.empty:
        st.info("No IT infrastructure assets in the current data.")
    else:
        st.dataframe(rank_by_priority(it)[REVIEW_COLS + ["location"]], hide_index=True)
    st.markdown("**Service-continuity watch-list** (High/Critical criticality with Moderate+ risk or overdue maintenance)")
    if watch.empty:
        st.success("No assets currently on the watch-list under the simulated rules.")
    else:
        st.dataframe(rank_by_priority(watch)[REVIEW_COLS], hide_index=True)
    st.caption("Dependency mapping between assets and digital services is not modelled in this prototype. "
               "The FHIR tab shows an illustrative, unvalidated payload; real interoperability needs "
               "validation, agreed terminology and a hospital integration project.")


def view_governance(scored, cfg):
    raw = st.session_state["assets_df"]
    fields = NUM_COLS + ["next_maintenance_due"]
    comp = pd.DataFrame({
        "field": fields,
        "completeness_pct": [round(100.0 * float(raw[f].notna().mean()), 1) if len(raw) else float("nan")
                             for f in fields]})
    cov_rows, bad_crit, bad_cat = [], 0, 0
    for _, row in scored.iterrows():
        res = compute_risk(signals_from_row(row.to_dict()), row["category"], row["criticality"], cfg)
        cov_rows.append({"asset_id": row["asset_id"], "data_coverage_pct": round(100 * res["coverage"]),
                         "missing_inputs": ", ".join(res["missing"]) or "-", "risk_level": res["level"]})
        bad_crit += normalise_criticality(row["criticality"]) is None
        bad_cat += row["category"] not in DEFAULT_WEIGHTS
    warns = st.session_state.get("data_warnings", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall field completeness", f"{fmt(comp['completeness_pct'].mean())}%")
    c2.metric("Load-time validation warnings", len(warns))
    c3.metric("Unrecognised criticality / category", f"{bad_crit} / {bad_cat}")
    c4.metric("Assets with insufficient data", int((scored["risk_level"] == "Insufficient data").sum()))
    left, right = st.columns(2)
    with left:
        st.markdown("**Completeness by field**")
        st.dataframe(comp, hide_index=True)
    with right:
        st.markdown("**Risk-score data coverage by asset**")
        st.dataframe(pd.DataFrame(cov_rows), hide_index=True)
    if warns:
        with st.expander("Validation warnings"):
            for w in warns:
                st.write("- " + w)
    st.markdown("**Traceability**")
    st.write(f"Source: synthetic in-memory sample data, generated {st.session_state['generated_at']} (UTC). "
             "Each score is reproducible from its listed inputs and the configurable weights "
             "(see the Registry detail view and the Methodology tab). Duplicate or blank asset IDs are "
             "dropped at load; invalid numbers are treated as missing and reported above.")
    st.caption("This view supports data-quality review. It is not a compliance assessment, and no "
               "regulatory conformance is claimed.")


def view_support(scored, cfg):
    orders = st.session_state.get("work_orders", [])
    open_orders = [o for o in orders if o["status"] != "Completed"]
    overdue, soon = _maintenance_counts(scored)
    alerts = scored[scored["risk_level"].isin(["Moderate", "High", "Insufficient data"])
                    | (scored["days_to_maintenance"].fillna(10_000) <= MAINT_WARNING_DAYS)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proactive alerts", len(alerts))
    c2.metric("Maintenance overdue", overdue)
    c3.metric("Due in 14 days", soon)
    c4.metric("Open demo work orders", len(open_orders))
    st.markdown("**Alert queue (ordered by illustrative priority)**")
    if alerts.empty:
        st.success("No alerts under the simulated rules.")
    else:
        q = rank_by_priority(alerts)[["asset_id", "asset_name", "location", "risk_level",
                                      "priority_score", "days_to_maintenance"]].copy()
        q["suggested_next_step"] = [suggested_step(scored[scored["asset_id"] == a].iloc[0], cfg)
                                    for a in q["asset_id"]]
        st.dataframe(q, hide_index=True)
    st.markdown("**Maintenance schedule (soonest / most overdue first)**")
    sched = scored.sort_values("days_to_maintenance", na_position="last")[
        ["asset_id", "asset_name", "next_maintenance_due", "days_to_maintenance", "status"]].copy()
    sched["next_maintenance_due"] = sched["next_maintenance_due"].map(
        lambda d: "" if d is None or pd.isna(d) else str(d))
    st.dataframe(sched, hide_index=True)
    st.markdown("**Open demo work orders**")
    if open_orders:
        st.dataframe(pd.DataFrame(open_orders), hide_index=True)
    else:
        st.info("No open demo work orders. Create them in the 'Impact & Work Orders' tab.")


def view_infrastructure(scored, cfg):
    rows = []
    for _, row in scored.iterrows():
        s = sanitize_signals(signals_from_row(row.to_dict()))
        band = f"{s['temp_band_min']:.0f}-{s['temp_band_max']:.0f}" if s["temp_band_min"] is not None else "n/a"
        rows.append({"asset_id": row["asset_id"], "asset_name": row["asset_name"], "location": row["location"],
                     "temperature_c (simulated)": s["temperature_c"], "assumed_band_c": band,
                     "thermal_status": thermal_status(row), "error_events_30d": s["error_events_30d"],
                     "runtime_used_pct": runtime_used_pct(row),
                     "days_to_maintenance": row["days_to_maintenance"], "risk_level": row["risk_level"]})
    ops = pd.DataFrame(rows)
    outside = int(ops["thermal_status"].str.startswith(("Above", "Below")).sum())
    overdue, _ = _maintenance_counts(scored)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets outside assumed temperature band", outside)
    c2.metric("Maintenance overdue", overdue)
    c3.metric("Highest recent error count", fmt(ops["error_events_30d"].max(), 0))
    st.markdown("**Operating conditions (simulated readings)**")
    st.dataframe(ops, hide_index=True)
    st.markdown("**Share of rated service life used (%)**")
    st.bar_chart(ops.set_index("asset_name")["runtime_used_pct"])
    st.caption("Temperature bands and rated hours are illustrative assumptions; replace them with "
               "manufacturer specifications before any real use. No live equipment is connected.")


def view_quality(scored, cfg):
    med = scored[scored["category"] == "Medical Equipment"]
    if med.empty:
        st.info("No medical equipment in the current data.")
        return
    overdue, _ = _maintenance_counts(med)
    flagged = med[med["criticality"].isin(["Critical", "High"])
                  & med["risk_level"].isin(["Moderate", "High", "Insufficient data"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Medical equipment assets", len(med))
    c2.metric("Moderate/High illustrative risk", int(med["risk_level"].isin(["Moderate", "High"]).sum()))
    c3.metric("Maintenance overdue", overdue)
    c4.metric("Condition Fair or Poor", int(med["condition"].isin(["Fair", "Poor"]).sum()))
    st.markdown("**Medical equipment (ranked by illustrative priority)**")
    st.dataframe(rank_by_priority(med)[REVIEW_COLS + ["status"]], hide_index=True)
    if flagged.empty:
        st.success("No High/Critical-criticality medical equipment with Moderate+ risk under the simulated rules.")
    else:
        for _, row in rank_by_priority(flagged).iterrows():
            st.warning(f"{row['asset_name']} ({row['asset_id']}, {row['department']}): "
                       f"risk {fmt(row['risk_score'])} ({row['risk_level']}), criticality {row['criticality']}. "
                       "Review by qualified biomedical engineering and clinical personnel before any "
                       "operational decision.")
    st.caption("Illustrative simulation only. It does not replace clinical or biomedical engineering judgement "
               "and makes no patient-safety determination.")


ROLE_VIEWS = {
    "Executive Management": view_executive,
    "E-Health Management": view_ehealth,
    "Data Quality & Governance": view_governance,
    "Technical Support & Applications": view_support,
    "IT Infrastructure & Biomed": view_infrastructure,
    "Quality & Patient Safety": view_quality,
}


def render_role_workspace(scored, cfg):
    role = st.session_state.get("role") or "Executive Management"
    st.subheader(f"Role Workspace - {role}")
    st.caption(ROLE_FOCUS.get(role, "") + " Role workspaces are display lenses for the demo, not access "
               "control: every role can open every tab. All figures derive from the synthetic demo data.")
    if scored.empty:
        st.info("No valid assets available.")
        return
    ROLE_VIEWS.get(role, view_executive)(scored, cfg)


def render_methodology():
    st.subheader("Methodology, Configuration & Validation Notes")
    st.markdown(
        "**What this is:** a transparent, rules-based risk *simulation*. It is **not** a trained machine-learning "
        "model, has no labeled failure history, and makes no accuracy claims.\n\n"
        "**Risk score (0-100, illustrative):** each factor gets a 0-1 sub-score; the score is the weighted "
        "average x 100. Missing or invalid inputs are excluded and weights re-normalised; if valid inputs cover "
        f"less than {MIN_COVERAGE:.0%} of the weight, no score is produced.\n\n"
        "**Maintenance priority (illustrative):** risk score x criticality factor, capped at 100. Criticality "
        "never changes the risk score. Current health condition (Good / Fair / Poor) is shown separately "
        "from future risk.\n\n"
        "**Temperature:** judged against an assumed per-asset band (illustrative values); real limits must come "
        "from manufacturer specifications.")
    st.markdown("**Thresholds**")
    t1, t2 = st.columns(2)
    t1.slider("Low risk below", 5, 50, DEFAULT_LOW_MAX, key="cfg_low_max")
    t2.slider("High risk from", 30, 90, DEFAULT_HIGH_MIN, key="cfg_high_min")
    st.caption("If 'High from' is not above 'Low below', High is set to Low + 1.")
    st.markdown("**Factor weights** (auto-normalised to 100%)")
    cols = st.columns(len(DEFAULT_WEIGHTS))
    for col, (cat, dflt) in zip(cols, DEFAULT_WEIGHTS.items()):
        with col:
            st.markdown(f"*{cat}*")
            for f, label in FACTORS.items():
                st.slider(label, 0.0, 1.0, dflt[f], 0.05, key=f"w_{cat}_{f}")
    st.markdown(
        "**Simulated:** all asset signals, risk scores, priorities, work orders, timestamps, FHIR payloads.\n\n"
        "**Not validated:** risk logic against real failures, FHIR conformance, security/compliance posture, "
        "any live device or EHR connection.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Anomalix+ Enterprise CMMS (Demo)", layout="wide")
    inject_css()
    init_state()
    if not st.session_state.get("authenticated"):
        render_login()
        return

    with st.sidebar:
        st.markdown(f"### Signed in: {st.session_state.get('role') or 'Demo user'}")
        st.caption("Prototype demo. Authentication is for presentation only; production security and "
                   "regulatory compliance have not been implemented or validated.")
        if st.button("Refresh demo data", key="btn_refresh"):
            refresh_demo_data()
            st.rerun()
        st.button("Sign out", on_click=do_logout, key="btn_logout")

    st.markdown(
        f"<div class='anx-header'><div class='anx-brand'>{logo_img(56)}<div>"
        "<h1>Anomalix+ Enterprise CMMS</h1>"
        "<p>Hospital asset intelligence prototype - predictive maintenance concepts (simulated)</p>"
        "</div></div>"
        "<p class='anx-ar'>نموذج أولي لإدارة أصول المستشفيات والموثوقية التشغيلية</p></div>",
        unsafe_allow_html=True)

    for w in st.session_state.get("data_warnings", []):
        st.warning(w)

    cfg = get_config()
    scored = score_dataframe(st.session_state["assets_df"], cfg)

    tabs = st.tabs(["Role Workspace", "Executive Dashboard", "Enterprise Asset Registry", "Scenario Simulator",
                    "Impact & Work Orders", "HL7 FHIR Demo", "Methodology"])
    with tabs[0]:
        render_role_workspace(scored, cfg)
    with tabs[1]:
        render_dashboard(scored)
    with tabs[2]:
        render_registry(scored, cfg)
    with tabs[3]:
        render_simulator(scored, cfg)
    with tabs[4]:
        render_impact(scored, cfg)
    with tabs[5]:
        render_fhir(scored)
    with tabs[6]:
        render_methodology()

    st.markdown("---")
    st.caption("Anomalix+ Enterprise CMMS - Demo prototype built with Python & Streamlit. "
               "All data is synthetic; no live equipment or EHR integration.")


if __name__ == "__main__":
    main()
