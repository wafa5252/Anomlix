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

BRAND = {"navy": "#0F172A", "blue": "#1E3A8A", "cyan": "#38BDF8", "amber": "#F59E0B", "light": "#E2E8F0"}

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
        "temperature_c": [38.0, 41.0, 25.0, 24.0, 33.0],
        "temp_band_min": [25.0, 20.0, 18.0, 18.0, 20.0],
        "temp_band_max": [45.0, 40.0, 27.0, 22.0, 40.0],
        "temp_margin": [15.0, 15.0, 10.0, 6.0, 15.0],
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
            warnings.append(f"{bad} missing/invalid/out-of-range value(s) in '{col}' were clipped or treated as missing.")

    dates = pd.to_datetime(out["next_maintenance_due"], errors="coerce")
    if int(dates.isna().sum()):
        warnings.append(f"{int(dates.isna().sum())} missing/invalid maintenance due date(s).")
    out["next_maintenance_due"] = dates.dt.date.where(dates.notna(), None)
    return out.reset_index(drop=True), warnings


# ----------------------------------------------------------------------------
# Risk engine
# ----------------------------------------------------------------------------
def get_config():
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
    try:
        if due is None or pd.isna(due):
            return None
        return (pd.Timestamp(due).date() - date.today()).days
    except (TypeError, ValueError, OverflowError):
        return None


def signals_from_row(row):
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
    if lo is None or hi is None or lo > hi:
        s["temp_band_min"] = s["temp_band_max"] = None
    if s["days_to_maintenance"] is not None:
        s["days_to_maintenance"] = int(np.floor(s["days_to_maintenance"]))
    return s


def condition_label(health):
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
            m = 0.6 + 0.4 * clip01(-d / 60.0)
        elif d <= MAINT_WARNING_DAYS:
            m = 0.4 * (1 - d / MAINT_WARNING_DAYS)
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
        score = round(float(clamp(acc, 0.0, 100.0)), 1)

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
    if result["level"] == "Insufficient data":
        return f"Too few valid inputs (data coverage {result['coverage']:.0%}, minimum {MIN_COVERAGE:.0%}); no score was produced."
    c = result["contributions"]
    top = c.sort_values("points", ascending=False).head(3)
    top = top[top["points"] >= 1.0]
    if top.empty:
        text = "No factor contributes materially; all simulated signals are in a healthy range."
    else:
        text = "Largest contributors: " + "; ".join(
            f"{r.factor} ({r.points:.0f} pts)" for r in top.itertuples()) + "."
    if result["missing"]:
        text += f" Missing inputs excluded (data coverage {result['coverage']:.0%}): " + ", ".join(result["missing"]) + "."
    for n in result["notes"]:
        text += " " + n
    return text


def recommend(result, category, criticality):
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
    out["days_to_maintenance"] = pd.array(cols["days_to_maintenance"], dtype="Int64") if len(out) else pd.array([], dtype="Int64")
    return out


def rank_by_priority(df):
    return df.sort_values(["priority_score", "risk_score", "asset_id"],
                          ascending=[False, False, True], na_position="last")


# ----------------------------------------------------------------------------
# FHIR R4 demonstration
# ----------------------------------------------------------------------------
FHIR_ID_RE = re.compile(r"^[A-Za-z0-9\-\.]{1,64}$")
URN_UUID_RE = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DEVICE_STATUS = {"active", "inactive", "entered-in-error", "unknown"}
DEVICEMETRIC_CATEGORY = {"measurement", "setting", "calculation", "unspecified"}
OBSERVATION_STATUS = {"registered", "preliminary", "final", "amended", "corrected",
                      "cancelled", "entered-in-error", "unknown"}
ALLOWED_REF_TARGETS = {
    ("Observation", "subject"): {"Patient", "Group", "Device", "Location"},
    ("Observation", "device"): {"Device", "DeviceMetric"},
    ("DeviceMetric", "source"): {"Device"},
}
REQUIRED_ELEMENTS = {"Device": [], "DeviceMetric": ["type", "category"],
                     "Observation": ["status", "code"]}


def _urn(asset_key, kind):
    return "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:anomalix:demo:{asset_key}:{kind}"))


def build_fhir_bundle(row, generated_at):
    asset_id = clean_text(row.get("asset_id"), "unknown")
    rid = re.sub(r"[^A-Za-z0-9\-\.]", "-", asset_id.lower())[:40].strip("-") or "asset"
    meta = {"tag": [{"system": LOCAL_CODE_SYSTEM, "code": "synthetic-demo-data"}]}
    device_url = _urn(rid, "device")

    device = {
        "resourceType": "Device", "id": f"{rid}", "meta": meta,
        "identifier": [{"system": LOCAL_ID_SYSTEM, "value": asset_id}],
        "status": "inactive" if clean_text(row.get("status")) == "Maintenance" else "active",
        "type": {"text": clean_text(row.get("category"), "Unknown")},
    }
    vendor = clean_text(row.get("vendor"))
    if vendor and vendor != "Unknown":
        device["manufacturer"] = vendor
    name = clean_text(row.get("asset_name"))
    if name and name != "Unknown":
        device["deviceName"] = [{"name": name, "type": "user-friendly-name"}]
    loc = " / ".join(x for x in (clean_text(row.get("location")), clean_text(row.get("department"))) if x and x != "Unknown")
    if loc:
        device["location"] = {"display": loc}
    entries = [(device_url, device)]

    temp = sanitize_signals({"temperature_c": row.get("temperature_c")})["temperature_c"]
    if temp is not None:
        metric_url = _urn(rid, "temp-metric")
        entries.append((metric_url, {
            "resourceType": "DeviceMetric", "id": f"{rid}-temp-metric", "meta": meta,
            "type": {"coding": [{"system": LOCAL_CODE_SYSTEM, "code": "simulated-temperature"}],
                     "text": "Simulated temperature"},
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
                     "text": "Anomalix demo asset health score"},
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
    checks = []
    entries = bundle.get("entry", [])
    checks.append(("Bundle", bundle.get("resourceType") == "Bundle" and bundle.get("type") == "collection", "resourceType Bundle"))
    urls = [e.get("fullUrl") for e in entries]
    url_map = {e.get("fullUrl"): e.get("resource", {}).get("resourceType") for e in entries}
    checks.append(("Bundle fullUrls", all(isinstance(u, str) and URN_UUID_RE.match(u) for u in urls) and len(set(urls)) == len(urls), "valid urn:uuid"))
    for e in entries:
        res = e.get("resource", {})
        rt = res.get("resourceType", "?")
        label = f"{rt}/{res.get('id', '?')}"
        problems = []
        if not FHIR_ID_RE.match(str(res.get("id", ""))):
            problems.append("id format")
        problems += [f"missing {k}" for k in REQUIRED_ELEMENTS.get(rt, []) if k not in res]
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
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ----------------------------------------------------------------------------
# Login screen
# ----------------------------------------------------------------------------
def render_login():
    st.markdown(f"<div class='anx-login-logo'>{logo_img(72)}</div>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;color:{BRAND['navy']};'>Anomalix+ Enterprise CMMS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748B;'>Prototype demo sign-in &nbsp;|&nbsp; <span dir='rtl'>تسجيل الدخول التجريبي</span></p>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.warning("Prototype-only authentication with shared demo credentials.")
        with st.form("login_form"):
            role = st.selectbox("Department / role", list(DEMO_CREDENTIALS.keys()))
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
        with st.expander("Demo credentials"):
            st.table(pd.DataFrame(
                [{"Role": r, "Username": c["user"], "Password": c["pass"], "Workspace focus": ROLE_FOCUS[r]}
                 for r, c in DEMO_CREDENTIALS.items()]))


# ----------------------------------------------------------------------------
# Tab & Role Workspaces renderers
# ----------------------------------------------------------------------------
def render_dashboard(scored):
    st.subheader("Executive Dashboard")
    if scored.empty:
        st.info("No valid assets available.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registered assets", len(scored))
    c2.metric("Average health score", fmt(scored["health_score"].mean()))
    c3.metric("High illustrative risk", int((scored["risk_level"] == "High").sum()))
    c4.metric("Maintenance overdue", int((scored["days_to_maintenance"].dropna() < 0).sum()))
    c5.metric("Due within 14 days", int(((scored["days_to_maintenance"].dropna() >= 0) & (scored["days_to_maintenance"].dropna() <= MAINT_WARNING_DAYS)).sum()))
    
    left, right = st.columns(2)
    with left:
        st.markdown("**Assets by operational status**")
        st.bar_chart(scored["status"].value_counts())
    with right:
        st.markdown("**Assets by criticality**")
        st.bar_chart(scored["criticality"].value_counts())


def render_registry(scored, cfg):
    st.subheader("Enterprise Asset Registry")
    if scored.empty:
        st.info("No valid assets available.")
        return
    st.dataframe(scored, hide_index=True)


def render_simulator(scored, cfg):
    st.subheader("Scenario Simulator")
    if scored.empty:
        st.info("No valid assets available.")
        return
    st.info("Select an asset and adjust parameters to simulate risk shifts.")


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
    wid = st.session_state.get("wo_update_id")
    new = st.session_state.get("wo_update_status")
    for o in st.session_state["work_orders"]:
        if o["work_order_id"] == wid:
            o["status"] = new
            st.session_state["wo_flash"] = ("success", f"{wid} set to '{new}'")
            return


def render_impact(scored, cfg):
    st.subheader("Impact Analysis & Maintenance Workflow")
    flash = st.session_state.pop("wo_flash", None)
    if flash:
        st.success(flash[1])
    orders = st.session_state["work_orders"]
    st.write(f"Total Work Orders: {len(orders)}")


def render_fhir(scored):
    st.subheader("HL7 FHIR Interoperability Demo")
    if scored.empty:
        return
    pick = st.selectbox("Select asset", scored["asset_id"].tolist(), key="fhir_asset")
    row = scored[scored["asset_id"] == pick].iloc[0]
    bundle = build_fhir_bundle(row.to_dict(), st.session_state["generated_at"])
    st.json(bundle)


def view_executive(scored, cfg):
    st.markdown("**Executive Summary Lens**")
    render_dashboard(scored)


def view_ehealth(scored, cfg):
    st.markdown("**E-Health Management Lens**")
    it = scored[scored["category"] == "IT Infrastructure"]
    st.dataframe(it, hide_index=True)


def view_governance(scored, cfg):
    st.markdown("**Data Quality & Governance Lens**")
    st.metric("Total Records", len(scored))


def view_support(scored, cfg):
    st.markdown("**Technical Support & Applications Lens**")
    render_impact(scored, cfg)


def view_infrastructure(scored, cfg):
    st.markdown("**IT Infrastructure & Biomed Lens**")
    st.dataframe(scored[scored["category"] == "IT Infrastructure"], hide_index=True)


ROLE_VIEWS = {
    "Executive Management": view_executive,
    "E-Health Management": view_ehealth,
    "Data Quality & Governance": view_governance,
    "Technical Support & Applications": view_support,
    "IT Infrastructure & Biomed": view_infrastructure,
}


def render_role_workspace(scored, cfg):
    role = st.session_state.get("role") or "Executive Management"
    st.subheader(f"Role Workspace - {role}")
    st.caption(ROLE_FOCUS.get(role, ""))
    if scored.empty:
        st.info("No valid assets available.")
        return
    ROLE_VIEWS.get(role, view_executive)(scored, cfg)


def render_methodology():
    st.subheader("Methodology & Configuration")
    st.write("Transparent rules-based risk simulation engine.")


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
        if st.button("Refresh demo data", key="btn_refresh"):
            refresh_demo_data()
            st.rerun()
        st.button("Sign out", on_click=do_logout, key="btn_logout")

    st.markdown(
        f"<div class='anx-header'><div class='anx-brand'>{logo_img(48)}<div>"
        "<h1>Anomalix+ Enterprise CMMS</h1>"
        "<p>Hospital asset intelligence prototype - IT & Infrastructure (simulated)</p>"
        "<p class='anx-ar'>نظام إدارة أصول وتكنولوجيا المعلومات للمستشفيات</p>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )

    for w in st.session_state.get("data_warnings", []):
        st.warning(w)

    cfg = get_config()
    scored = score_dataframe(st.session_state["assets_df"], cfg)

    tabs = st.tabs(["Role Workspace", "Executive Dashboard", "Enterprise Asset Registry", 
                    "Scenario Simulator", "Impact & Work Orders", "HL7 FHIR Demo", "Methodology"])
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


if __name__ == "__main__":
    main()
