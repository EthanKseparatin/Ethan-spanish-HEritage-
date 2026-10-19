from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU", "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current mortality patterns persist."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, expressed in current US dollars; it is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Number of immigrants minus emigrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicide victims per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.")
}
CONTEXTS = {
    "Business": ("exponential", "Revenue", "US dollars", "time", "years", "Revenue may grow or shrink; the first derivative is the change in revenue per year."),
    "Engineering": ("polynomial", "Position", "meters", "time", "seconds", "The first derivative of position is velocity and the second derivative is acceleration."),
    "Medical": ("exponential", "Concentration", "mg/L", "time", "hours", "A concentration curve can represent change over time; it is illustrative, not medical advice."),
    "Sports science": ("polynomial", "Position", "meters", "time", "seconds", "Position changes over time; its derivative is velocity and its second derivative is acceleration."),
    "Biology": ("exponential", "Population", "organisms", "time", "days", "The function can represent population size; real populations are limited by resources."),
    "Bioengineering": ("trigonometric", "Signal amplitude", "volts", "time", "seconds", "A periodic signal can model repeated motion or measurements."),
    "Economic forecasting": ("polynomial", "Economic indicator", "index units", "time", "years", "The curve describes a trend, not a guaranteed forecast or a causal explanation."),
    "Weather": ("trigonometric", "Temperature", "°C", "time", "hours", "A periodic function can approximate cycles; actual weather is more complex."),
    "Custom": ("polynomial", "Output", "units", "Input", "units", "Describe the meaning and limitations of your custom model.")
}
WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=20000"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🌎 Hispanic Heritage Data Lab")
st.caption("Historical data, curve fitting, and interactive function analysis for Algebra 2, Precalculus, and Calculus.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    url = WB_BASE.format(country=country_code, indicator=indicator_code)
    response = requests.get(url, params={"date": f"{start_year}:{end_year}"}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank API returned an unexpected or empty response.")
    rows = []
    for item in payload[1]:
        if item and item.get("date") and item.get("value") is not None:
            try:
                rows.append({"year": int(item["date"]), "value": float(item["value"])})
            except (TypeError, ValueError):
                continue
    df = pd.DataFrame(rows, columns=["year", "value"]).drop_duplicates("year").sort_values("year")
    return df.reset_index(drop=True), response.url

def validate_year_value(df: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    out = df.copy()
    if not {"year", "value"}.issubset(out.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out["year"].isna().any(): errors.append("Every year must be an integer.")
    else:
        if not np.all(np.equal(out["year"], np.floor(out["year"]))): errors.append("Years must be integers.")
        if out["year"].duplicated().any(): errors.append("Years must be unique; remove duplicate years.")
    if out["value"].isna().any(): errors.append("Every value must be numeric; blanks are not valid observations.")
    out = out.dropna(subset=["year", "value"]).copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values("year").reset_index(drop=True)
    if indicator_name == "Life expectancy at birth" and ((out["value"] < 0) | (out["value"] > 120)).any():
        errors.append("Life expectancy values must be between 0 and 120 years.")
    if indicator_name == "Unemployment, national estimate" and ((out["value"] < 0) | (out["value"] > 100)).any():
        errors.append("Unemployment percentages must be between 0 and 100.")
    if indicator_name == "Crude birth rate" and (out["value"] < 0).any():
        errors.append("Crude birth rate cannot be negative.")
    if indicator_name == "Intentional homicide rate" and (out["value"] < 0).any():
        errors.append("Homicide rate cannot be negative.")
    return out, errors

def safe_parse_function(text: str, symbol: sp.Symbol) -> sp.Expr:
    """Parse familiar notation while restricting available names and operations."""
    text = text.strip().replace("^", "**")
    text = text.replace("×", "*").replace("÷", "/")
    allowed = {"x": symbol, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "E": sp.E}
    if not text or len(text) > 250:
        raise ValueError("Enter a non-empty expression shorter than 250 characters.")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s^]+", text):
        raise ValueError("Use standard math characters, x, and supported functions (sin, cos, tan, exp, log, sqrt, abs).")
    try:
        expr = sp.sympify(text, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse this expression. Check parentheses, operators, and function names.") from exc
    if expr.free_symbols - {symbol}:
        raise ValueError("Only x may be used as a variable.")
    return expr

def lambdify_expr(expr: sp.Expr, symbol: sp.Symbol):
    fn = sp.lambdify(symbol, expr, modules=["numpy"])
    def evaluate(x):
        with np.errstate(all="ignore"):
            y = np.asarray(fn(x), dtype=float)
        return y
    return evaluate

def finite_eval(fn, x):
    try:
        y = fn(x)
        y = float(np.asarray(y).reshape(-1)[0])
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan

def derivative_roots(expr, symbol, lo, hi):
    d1 = sp.diff(expr, symbol)
    roots = []
    try:
        candidates = sp.solve(sp.Eq(d1, 0), symbol)
        for root in candidates:
            if root.is_real is True:
                value = float(root)
                if lo <= value <= hi and np.isfinite(value):
                    roots.append(value)
    except Exception:
        pass
    # Numeric scan supplements symbolic roots for functions with difficult roots.
    try:
        f1 = lambdify_expr(d1, symbol)
        xs = np.linspace(lo, hi, 1000)
        ys = np.array([finite_eval(f1, x) for x in xs])
        for i in range(len(xs)-1):
            if np.isfinite(ys[i]) and np.isfinite(ys[i+1]) and ys[i] * ys[i+1] < 0:
                a, b = xs[i], xs[i+1]
                for _ in range(45):
                    mid = (a+b)/2
                    ym = finite_eval(f1, mid)
                    if not np.isfinite(ym): break
                    if ys[i] * ym <= 0: b = mid
                    else: a = mid
                roots.append((a+b)/2)
    except Exception:
        pass
    return sorted(set(round(r, 7) for r in roots)), d1

def classify_calculus(expr, symbol, lo, hi):
    f = lambdify_expr(expr, symbol)
    roots, d1 = derivative_roots(expr, symbol, lo, hi)
    d2 = sp.diff(expr, symbol, 2)
    f1, f2 = lambdify_expr(d1, symbol), lambdify_expr(d2, symbol)
    crit = []
    for x in roots:
        delta = max(1e-5, (hi-lo)*1e-5)
        left, right = finite_eval(f1, max(lo, x-delta)), finite_eval(f1, min(hi, x+delta))
        y = finite_eval(f, x)
        second = finite_eval(f2, x)
        kind = "critical point (no extremum established)"
        if x > lo + 1e-7 and x < hi - 1e-7:
            if np.isfinite(left) and np.isfinite(right):
                if left > 0 and right < 0: kind = "local maximum (first-derivative test)"
                elif left < 0 and right > 0: kind = "local minimum (first-derivative test)"
            if kind.startswith("critical point") and np.isfinite(second):
                if second < -1e-7: kind = "local maximum (second-derivative test)"
                elif second > 1e-7: kind = "local minimum (second-derivative test)"
        crit.append({"x": x, "y": y, "type": kind})
    # Concavity change is checked on either side of each candidate.
    infl = []
    try:
        candidates = sp.solve(sp.Eq(d2, 0), symbol)
    except Exception:
        candidates = []
    for r in candidates:
        try:
            x = float(r)
        except Exception:
            continue
        if not lo < x < hi: continue
        delta = max(1e-5, (hi-lo)*1e-5)
        left, right = finite_eval(f2, x-delta), finite_eval(f2, x+delta)
        if np.isfinite(left) and np.isfinite(right) and left*right < 0:
            infl.append({"x": x, "y": finite_eval(f, x), "type": "inflection point (concavity changes)"})
    # Domain warning for common rational/log functions; sampled gaps also reveal undefined intervals.
    xs = np.linspace(lo, hi, 1500)
    ys = np.array([finite_eval(f, x) for x in xs])
    gaps = int(np.sum(~np.isfinite(ys)))
    return crit, infl, d1, d2, gaps

def make_chart(df, model=None, symbol=None, domain=None, sample_step=1, query_points=None,
               show_f=True, show_d1=False, show_d2=False, tangent=None, secant=None, title="Model"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if not df.empty:
        ax.scatter(df["year"], df["value"], label="Observed data", s=24, zorder=4)
        ax.axvline(df["year"].min(), linestyle=":", linewidth=1, label="Historical boundary")
        ax.axvline(df["year"].max(), linestyle=":", linewidth=1)
    if model is not None and domain is not None:
        lo, hi = domain
        xs = np.linspace(lo, hi, 1200)
        if show_f:
            ys = np.array([finite_eval(model, x) for x in xs])
            finite = np.isfinite(ys)
            # Plot separate contiguous runs so discontinuities are not bridged.
            start = None
            for i, ok in enumerate(finite):
                if ok and start is None: start = i
                if start is not None and (not ok or i == len(finite)-1):
                    end = i if not ok else i+1
                    if end-start > 1: ax.plot(xs[start:end], ys[start:end], label="Function / fitted curve")
                    start = None
        if sample_step and show_f:
            sx = np.arange(math.ceil(lo), math.floor(hi)+1, int(sample_step))
            sy = np.array([finite_eval(model, x) for x in sx])
            good = np.isfinite(sy)
            ax.scatter(sx[good], sy[good], s=18, marker="x", label=f"Model markers (every {sample_step} year/unit)")
    if tangent is not None and model is not None:
        x0, slope = tangent
        y0 = finite_eval(model, x0)
        if np.isfinite(y0) and np.isfinite(slope):
            line_x = np.linspace(x0-(domain[1]-domain[0])*.12, x0+(domain[1]-domain[0])*.12, 60)
            ax.plot(line_x, y0+slope*(line_x-x0), "--", label="Tangent line")
            ax.scatter([x0], [y0], s=45, label="Current point")
    if secant is not None and model is not None:
        a, b = secant
        ya, yb = finite_eval(model, a), finite_eval(model, b)
        if np.isfinite(ya) and np.isfinite(yb) and abs(b-a)>1e-12:
            ax.plot([a,b], [ya,yb], "--", label="Secant line")
    if query_points:
        qx, qy = [], []
        for x in query_points:
            y = finite_eval(model, x) if model is not None else np.nan
            if np.isfinite(y): qx.append(x); qy.append(y)
        if qx: ax.scatter(qx, qy, s=70, marker="*", label="Calculated estimates", zorder=6)
    ax.set_title(title)
    ax.set_xlabel("Input (year for World Bank data; otherwise selected input units)")
    ax.set_ylabel("Output (see selected units)")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def dataframe_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def make_xlsx(summary: dict, working: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        working.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for sheet, frame in [("Summary", pd.DataFrame([summary])), ("Working data", working), ("Calculated years", results)]:
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            for i, col in enumerate(frame.columns):
                width = min(42, max(12, len(str(col))+2, *(len(str(v)) for v in frame[col].head(80).fillna(""))))
                ws.set_column(i, i, width)
    return buf.getvalue()

def make_pdf(summary, working, results, fig) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=32, leftMargin=32, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Spacer(1, 8)]
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{str(k)}</b>: {str(v)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=130, bbox_inches="tight")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=7.5*inch, height=3.7*inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = working.head(18).round(5)
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e8f5")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Calculated-year results", styles["Heading2"]))
    if not results.empty:
        res = results.head(30)
        t2 = Table([list(res.columns)] + res.astype(str).values.tolist(), repeatRows=1)
        t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9e8f5")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(t2)
    doc.build(story)
    return buf.getvalue()

# Sidebar / source selection
with st.sidebar:
    st.header("Data and model")
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    country_name = st.selectbox("Featured country", list(COUNTRIES))
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    comparison_names = st.multiselect("Comparison countries (same indicator)", [c for c in COUNTRIES if c != country_name], default=[])
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment", 1, 10, 2)
    forward_horizon = st.slider("Forward display horizon", 0, 50, 10)
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

code, units, definition = INDICATORS[indicator_name]
current_year = date.today().year
start_year, end_year = current_year-70, current_year-1
st.subheader("1. Authentic historical data")
st.markdown(f"**Research question:** How has **{indicator_name.lower()}** changed in {country_name}, and how does it compare with selected countries? Edit the research question in your project write-up if your team's approved question differs.")
st.caption("Brazil is included as a Latin American comparison; it is not a Spanish-speaking country.")
source_url = f"https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{code}?format=json&date={start_year}:{end_year}"
st.write(f"**Definition:** {definition}")
st.write(f"**Units:** {units}")
st.markdown(f"**Official source:** [World Bank indicator API]({source_url}) · **Retrieval date:** {date.today().isoformat()} · **Requested years:** {start_year}–{end_year} (70 calendar years).")
st.caption("Coverage is based on non-null values returned by the API; the number of observations may be fewer than 70.")

try:
    source_df, resolved_url = fetch_world_bank(COUNTRIES[country_name], code, start_year, end_year)
    api_error = None
except Exception as exc:
    source_df = pd.DataFrame(columns=["year", "value"])
    resolved_url = source_url
    api_error = str(exc)
if api_error:
    st.error(f"Could not retrieve World Bank data: {api_error}. Check your internet connection, the API status, or the app logs. No replacement or random data has been created.")
elif source_df.empty:
    st.warning("The API returned no non-null observations for this country/indicator/time window.")
else:
    st.success(f"Source coverage: {source_df.year.min()}–{source_df.year.max()}, {len(source_df)} non-null observations.")

source_key = f"source_{country_name}_{indicator_name}"
if source_key not in st.session_state:
    st.session_state[source_key] = source_df.copy()
edit_key = f"edit_{country_name}_{indicator_name}"
if edit_key not in st.session_state:
    st.session_state[edit_key] = source_df.copy()
if st.button("Reset edited data to source"):
    st.session_state[edit_key] = source_df.copy()
    st.session_state.pop(f"editor_{country_name}_{indicator_name}", None)
    st.session_state.pop("calculated_results", None)
    st.rerun()

st.markdown("**Editable working table** — change years/values to explore a scenario. Keep it distinct from the original source data.")
edited = st.data_editor(st.session_state[edit_key], num_rows="dynamic", use_container_width=True, key=f"editor_{country_name}_{indicator_name}")
validated, validation_errors = validate_year_value(edited, indicator_name)
if validation_errors:
    for err in validation_errors: st.error(err)
data_changed = not validated.equals(source_df.reset_index(drop=True))
if data_changed:
    st.warning("USER-MODIFIED SCENARIO: working data differs from the original World Bank data.")
if len(validated) < 2:
    st.warning("At least two valid observations are needed to draw a trend. Polynomial fitting requires at least degree + 1 distinct years.")
st.caption(f"Validated working coverage: {validated.year.min() if len(validated) else '—'}–{validated.year.max() if len(validated) else '—'} · {len(validated)} rows.")

# Model fit
model = None
model_expr = None
metrics = {}
fit_error = None
base_year = int(validated["year"].min()) if len(validated) else current_year-1
if len(validated) >= degree + 1 and not validation_errors:
    x = validated["year"].to_numpy(dtype=float)
    y = validated["value"].to_numpy(dtype=float)
    base_year = int(np.min(x))
    t = x - base_year
    try:
        coef = np.polynomial.polynomial.polyfit(t, y, degree)
        model = lambda year, c=coef, b=base_year: np.polynomial.polynomial.polyval(np.asarray(year)-b, c)
        z = sp.Symbol("t")
        model_expr = sp.Add(*[sp.Float(float(coef[i]), 14)*z**i for i in range(len(coef))])
        yhat = np.asarray(model(x), dtype=float)
        metrics = {"R-squared (in-sample)": float(r2_score(y, yhat)) if len(y)>1 else float("nan"),
                   "MAE (in-sample)": float(mean_absolute_error(y, yhat)),
                   "Observation count": len(y), "Historical range": f"{int(np.min(x))}–{int(np.max(x))}",
                   "Mean observed value": float(np.mean(y)), "Base year": base_year, "Degree": degree}
    except Exception as exc:
        fit_error = str(exc)
elif len(validated) and not validation_errors:
    fit_error = f"Need at least degree + 1 = {degree+1} distinct valid years; currently have {len(validated)}."
if fit_error:
    st.warning(f"Polynomial model not available: {fit_error}")

if model is not None:
    st.subheader("2. Fit and interpret the historical trend")
    st.write(f"Using centered time **t = year − {base_year}**, the fitted polynomial is:")
    coeff_text = " + ".join([f"{float(c):.10g}" + ("" if i==0 else f"·t^{i}") for i,c in enumerate(coef)])
    st.code(f"f(t) = {coeff_text}")
    mcols = st.columns(4)
    for i, (k,v) in enumerate(metrics.items()):
        if k in ["R-squared (in-sample)", "MAE (in-sample)", "Observation count", "Historical range"]:
            mcols[["R-squared (in-sample)", "MAE (in-sample)", "Observation count", "Historical range"].index(k)].metric(k, f"{v:.5g}" if isinstance(v,(float,np.floating)) else str(v))
    st.caption("Fit metrics are in-sample. A high R-squared does not establish causation or guarantee accurate future predictions.")
    st.write("**Residuals:** actual value − fitted value.")
    residuals = validated.copy()
    residuals["fitted_value"] = yhat
    residuals["residual"] = y-yhat
    with st.expander("Show residual table and model summary"):
        st.dataframe(residuals, use_container_width=True)
        st.json(metrics)
    if len(validated) >= max(degree+3, 8):
        holdout_n = max(1, min(len(validated)//5, len(validated)-degree-1))
        train, test = validated.iloc[:-holdout_n], validated.iloc[-holdout_n:]
        if len(train) >= degree+1:
            hc = np.polynomial.polynomial.polyfit(train.year.to_numpy()-base_year, train.value.to_numpy(), degree)
            hy = np.polynomial.polynomial.polyval(test.year.to_numpy()-base_year, hc)
            st.write(f"**Chronological holdout MAE:** {mean_absolute_error(test.value, hy):.5g} {units} on the last {holdout_n} observations (separate from in-sample metrics).")
    st.info("Uncertainty: use the residual table and MAE as a description of typical in-sample error, not as a validated 95% prediction interval. Polynomial extrapolation can become implausible quickly; estimates outside the fitted year range are especially uncertain.")

    single_year = st.number_input("Single year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(2100, max(1950, end_year+1)), step=1, key="single_year")
    if st.button("Calculate Value"):
        yy = int(single_year)
        st.session_state["single_result"] = {"year": yy, "estimate": finite_eval(model, yy), "country": country_name, "indicator": indicator_name, "degree": degree, "data_signature": hash((country_name, indicator_name, degree, tuple(validated.itertuples(index=False, name=None))))}
    signature = hash((country_name, indicator_name, degree, tuple(validated.itertuples(index=False, name=None))))
    if st.session_state.get("single_result", {}).get("data_signature") != signature:
        st.session_state.pop("single_result", None)
    if "single_result" in st.session_state:
        sr = st.session_state["single_result"]
        inside = int(validated.year.min()) <= sr["year"] <= int(validated.year.max())
        st.write(f"**Calculated estimate for {sr['year']}:** {sr['estimate']:.8g} {units} — {'within' if inside else 'outside'} the fitted historical range ({validated.year.min()}–{validated.year.max()}). This is a fitted value, not an actual observation.")
        if not inside: st.warning("Extrapolation warning: this polynomial estimate may be implausible; it has not been clipped.")
    st.markdown("**Batch estimates**")
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    batch_years = []
    if batch_mode == "Year range":
        c1,c2,c3 = st.columns(3)
        with c1: batch_start = st.number_input("Start year", 1950, 2100, max(1950, int(validated.year.min())), key="batch_start")
        with c2: batch_end = st.number_input("End year", 1950, 2100, min(2100, int(validated.year.max())), key="batch_end")
        with c3: batch_step = st.slider("Range step", 1, 10, 1, key="batch_step")
        if batch_start <= batch_end: batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
        else: st.error("Start year must be less than or equal to end year.")
    else:
        custom = st.text_input("Comma-separated years", value=f"{max(1950, int(validated.year.min()))}, {min(2100, int(validated.year.max()))}")
        for token in custom.split(","):
            token = token.strip()
            if not token: continue
            if not re.fullmatch(r"\d{4}", token) or not 1950 <= int(token) <= 2100:
                st.warning(f"Invalid year '{token}'. Enter integer years from 1950 through 2100.")
            else: batch_years.append(int(token))
    batch_years = list(dict.fromkeys(batch_years))
    results = pd.DataFrame(columns=["year","estimate","units","range_status"])
    if batch_years:
        results = pd.DataFrame({"year": batch_years})
        results["estimate"] = [finite_eval(model, yy) for yy in batch_years]
        results["units"] = units
        results["range_status"] = results.year.apply(lambda yy: "within fitted range" if validated.year.min() <= yy <= validated.year.max() else "outside fitted range")
        st.dataframe(results, use_container_width=True)
    else:
        st.caption("Enter valid batch years to see estimates.")
else:
    results = pd.DataFrame(columns=["year","estimate","units","range_status"])

# Contextual function / calculus explorer
st.divider()
st.subheader("3. Interactive function and calculus lab")
ctx = st.selectbox("Context", list(CONTEXTS))
suggested_kind, suggested_y, suggested_yu, suggested_x, suggested_xu, ctx_help = CONTEXTS[ctx]
ctx_description = st.text_input("Context description", value=ctx_help)
c1,c2,c3 = st.columns(3)
with c1: x_name = st.text_input("Independent variable", value=suggested_x)
with c2: y_name = st.text_input("Dependent variable", value=suggested_y)
with c3: y_units = st.text_input("Dependent units", value=suggested_yu)
x_units = st.text_input("Independent-variable units", value=suggested_xu)
function_source = st.radio("Function source", ["Curve of best fit", "Suggested function", "Parameter builder", "Enter my own function"], horizontal=True)
x = sp.Symbol("x", real=True)
custom_lo, custom_hi = st.columns(2)
with custom_lo: interval_lo = st.number_input("Analysis interval start", value=0.0, key="analysis_lo")
with custom_hi: interval_hi = st.number_input("Analysis interval end", value=10.0, key="analysis_hi")
if interval_lo >= interval_hi:
    st.error("Analysis interval start must be less than end.")
    interval_lo, interval_hi = 0.0, 10.0

chosen_expr = None
function_label = ""
if function_source == "Curve of best fit":
    if model is not None:
        # Model is in t = year - base year; substitute t = x - base year.
        chosen_expr = sp.Add(*[sp.Float(float(coef[i]), 14)*(x-base_year)**i for i in range(len(coef))])
        function_label = "World Bank polynomial fit (x is calendar year)"
        if interval_lo == 0.0 and interval_hi == 10.0:
            st.caption("This curve uses calendar-year x values. Set the analysis interval to meaningful calendar years in the fitted range.")
    else:
        st.warning("The data-fit function is unavailable until valid data contain at least degree + 1 distinct years.")
elif function_source == "Suggested function":
    kind = st.selectbox("Suggested function type", ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"], index=["polynomial","rational","exponential","logarithmic","trigonometric"].index(suggested_kind))
    if kind == "Polynomial":
        a = st.number_input("a (quadratic coefficient)", value=1.0)
        b = st.number_input("b (linear coefficient)", value=-2.0)
        c = st.number_input("c (constant)", value=1.0)
        chosen_expr = a*x**2+b*x+c
    elif kind == "Rational":
        a = st.number_input("Numerator scale a", value=1.0)
        b = st.number_input("Denominator shift b", value=1.0)
        chosen_expr = a/(x-b)
    elif kind == "Exponential":
        a = st.number_input("Scale a", value=2.0)
        k = st.number_input("Exponent rate k", value=0.15)
        chosen_expr = a*sp.exp(k*x)
    elif kind == "Logarithmic":
        a = st.number_input("Scale a", value=1.0)
        b = st.number_input("Shift b (x+b must be positive)", value=1.0)
        chosen_expr = a*sp.log(x+b)
    else:
        a = st.number_input("Amplitude a", value=1.0)
        k = st.number_input("Frequency k", value=1.0)
        b = st.number_input("Vertical shift b", value=0.0)
        chosen_expr = a*sp.sin(k*x)+b
    function_label = f"Suggested {kind.lower()} function"
elif function_source == "Parameter builder":
    kind = st.selectbox("Builder family", ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    a = st.number_input("Parameter a", value=1.0, key="builder_a")
    b = st.number_input("Parameter b", value=1.0, key="builder_b")
    c = st.number_input("Parameter c", value=0.0, key="builder_c")
    if kind == "Polynomial": chosen_expr = a*x**2+b*x+c
    elif kind == "Rational": chosen_expr = a/(x-b)
    elif kind == "Exponential": chosen_expr = a*sp.exp(b*x)+c
    elif kind == "Logarithmic": chosen_expr = a*sp.log(x+b)+c
    else: chosen_expr = a*sp.sin(b*x)+c
    function_label = f"Parameter-built {kind.lower()} function"
else:
    expression_text = st.text_input("Enter f(x)", value="x^3 - 3*x")
    try:
        chosen_expr = safe_parse_function(expression_text, x)
        function_label = "User-entered function"
    except ValueError as exc:
        st.error(str(exc))

if chosen_expr is not None:
    st.markdown(f"**Selected function:** `{sp.sstr(chosen_expr)}`")
    f = lambdify_expr(chosen_expr, x)
    d1 = sp.diff(chosen_expr, x)
    d2 = sp.diff(chosen_expr, x, 2)
    f1, f2 = lambdify_expr(d1, x), lambdify_expr(d2, x)
    st.write(f"**Function:** $f(x) = {sp.latex(chosen_expr)}$")
    if course == "Calculus":
        st.write(f"**First derivative:** $f'(x) = {sp.latex(d1)}$")
        st.write(f"**Second derivative:** $f''(x) = {sp.latex(d2)}$")
    elif course == "Precalculus":
        st.write(f"Degree/coefficient structure: `{sp.sstr(chosen_expr)}`. Changing coefficients can alter steepness, intercepts, turning points, and concavity.")
    else:
        st.write("Focus: domain and range, end behavior, graph trends, and average rates of change.")

    # Function controls and synchronized point / interval
    left, right = st.columns(2)
    with left:
        current_x = st.slider("Current input x", float(interval_lo), float(interval_hi), float((interval_lo+interval_hi)/2), key="current_x")
    with right:
        rate_interval = st.slider("Average-rate interval [a, b]", float(interval_lo), float(interval_hi),
                                  (float(interval_lo), float(interval_hi)), key="rate_interval")
    a_int, b_int = rate_interval
    y_current = finite_eval(f, current_x)
    slope_current = finite_eval(f1, current_x)
    y_a, y_b = finite_eval(f, a_int), finite_eval(f, b_int)
    avg_rate = (y_b-y_a)/(b_int-a_int) if np.isfinite(y_a) and np.isfinite(y_b) and b_int != a_int else np.nan
    st.write(f"**At x = {current_x:.5g}:** f(x) = {y_current:.7g} {y_units}; f′(x) = {slope_current:.7g} {y_units}/{x_units}; f″(x) = {finite_eval(f2,current_x):.7g} {y_units}/{x_units}².")
    st.write(f"**Average rate of change on [{a_int:.5g}, {b_int:.5g}]:** {avg_rate:.7g} {y_units}/{x_units}" if np.isfinite(avg_rate) else "Average rate of change is undefined because an endpoint is not in the domain.")
    if ctx in ["Engineering", "Sports science"]:
        st.caption("When y represents position and x represents time, f′ is velocity and f″ is acceleration. Their units are distance/time and distance/time².")
    elif ctx in ["Business", "Economic forecasting"]:
        st.caption(f"The first derivative describes the local change in {y_name} per {x_units}; the second derivative describes whether that rate is increasing or decreasing. This model does not establish causes.")
    else:
        st.caption(ctx_description)
    show_f = st.checkbox("Show function", True)
    show_d1 = st.checkbox("Show first derivative", course == "Calculus")
    show_d2 = st.checkbox("Show second derivative", False)
    # Plot main function, derivative curves on a secondary panel to keep units clear.
    fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4.8))
    xs = np.linspace(interval_lo, interval_hi, 1200)
    if show_f:
        ys = np.array([finite_eval(f, xx) for xx in xs])
        ax_ctx.plot(xs, ys, label="f(x)")
    if show_d1:
        ys1 = np.array([finite_eval(f1, xx) for xx in xs])
        ax_ctx.plot(xs, ys1, label="f′(x)")
    if show_d2:
        ys2 = np.array([finite_eval(f2, xx) for xx in xs])
        ax_ctx.plot(xs, ys2, label="f″(x)")
    if np.isfinite(y_current):
        ax_ctx.scatter([current_x], [y_current], marker="o", s=55, label="Current point")
    if np.isfinite(slope_current) and np.isfinite(y_current):
        tangent_x = np.linspace(max(interval_lo,current_x-(interval_hi-interval_lo)*.12), min(interval_hi,current_x+(interval_hi-interval_lo)*.12), 60)
        ax_ctx.plot(tangent_x, y_current+slope_current*(tangent_x-current_x), "--", label="Tangent line")
    if np.isfinite(y_a) and np.isfinite(y_b):
        ax_ctx.plot([a_int,b_int],[y_a,y_b],"--",label="Secant line")
    ax_ctx.axhline(0, linewidth=.7)
    ax_ctx.set_xlabel(f"{x_name} ({x_units})")
    ax_ctx.set_ylabel(f"{y_name} / derivatives (units differ; see labels above)")
    ax_ctx.set_title(f"{ctx}: {function_label}")
    ax_ctx.grid(True, alpha=.25)
    ax_ctx.legend(fontsize=8)
    fig_ctx.tight_layout()
    st.pyplot(fig_ctx, use_container_width=True)
    st.caption("Curves are broken at sampled undefined points. If a narrow discontinuity is missed between samples, increase scrutiny of the expression and interval; plots are numerical approximations.")

    if course == "Calculus":
        crit, infl, d1, d2, gaps = classify_calculus(chosen_expr, x, interval_lo, interval_hi)
        st.markdown("**Critical points and extrema (numerical/symbolic checks on the selected interval)**")
        if crit:
            st.dataframe(pd.DataFrame(crit), use_container_width=True)
        else:
            st.write("No stationary critical points were identified in this interval. This does not rule out points where the derivative is undefined.")
        # Increasing/decreasing interval classification by sampling derivative sign.
        test_x = np.linspace(interval_lo, interval_hi, 21)
        signs = []
        for xx in test_x:
            val = finite_eval(f1, xx)
            signs.append("increasing" if val > 1e-7 else ("decreasing" if val < -1e-7 else "stationary/undefined"))
        st.write("**Sampled behavior:** " + ", ".join(f"{xx:.3g}: {sg}" for xx,sg in zip(test_x[::5], signs[::5])))
        st.markdown("**Inflection points**")
        if infl: st.dataframe(pd.DataFrame(infl), use_container_width=True)
        else: st.write("No concavity-changing inflection points were verified among the solved second-derivative zeros.")
        # Closed-interval extrema include endpoints and stationary candidates.
        candidates = [interval_lo, interval_hi] + [c["x"] for c in crit if np.isfinite(c["y"])]
        vals = [(xx, finite_eval(f, xx)) for xx in candidates]
        vals = [(xx, yy) for xx,yy in vals if np.isfinite(yy)]
        if vals:
            minp, maxp = min(vals, key=lambda p:p[1]), max(vals, key=lambda p:p[1])
            st.write(f"**Absolute extrema on the selected closed interval, among evaluated candidates:** min at x≈{minp[0]:.6g}, f≈{minp[1]:.6g}; max at x≈{maxp[0]:.6g}, f≈{maxp[1]:.6g}.")
        st.write(f"**Domain/discontinuity check:** {gaps} of 1,500 sampled x-values were undefined or non-finite in the selected interval. For rational/logarithmic functions, domain restrictions must be checked explicitly.")
        if "log" in str(chosen_expr): st.info("Logarithms require a positive argument; values outside that domain are undefined.")
        if "/" in str(chosen_expr): st.info("Rational expressions may be undefined where the denominator is zero. The plot avoids connecting sampled gaps.")
    elif course == "Precalculus":
        st.write(f"**Concavity clue:** f″(x) at the selected point is {finite_eval(f2,current_x):.6g}. A positive value indicates upward concavity locally; a negative value indicates downward concavity locally, when the derivative exists.")
        st.write("Turning points can occur where the slope changes sign. A zero derivative alone does not guarantee a maximum or minimum.")
    else:
        st.write(f"**Domain/range and end behavior:** inspect the plotted interval and the expression's restrictions. On this selected interval the sampled output range is {np.nanmin([finite_eval(f,xx) for xx in xs]):.6g} to {np.nanmax([finite_eval(f,xx) for xx in xs]):.6g}, when finite. Average rates compare output change with input change.")

    # Animation with manual play-state and rerun-driven frame
    st.markdown("**Simulation controls**")
    visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or concentration indicator", "Population / cell colony", "Thermometer / weather indicator", "Business / economic indicator"])
    play = st.toggle("Play automatic simulation", value=False)
    speed = st.slider("Playback speed (x)", 0.25, 3.0, 1.0, 0.25)
    frame_count = st.slider("Simulation frame", 0, 100, 50)
    if st.button("Reset simulation"):
        st.session_state["frame_count"] = 50
        st.session_state["current_x"] = float((interval_lo+interval_hi)/2)
        st.rerun()
    # The frame slider synchronizes the point by defining its input; play mode is a clear manual frame playback.
    sim_x = interval_lo + (interval_hi-interval_lo)*frame_count/100
    sim_y = finite_eval(f, sim_x)
    st.write(f"**Simulation ({visual}):** frame input x≈{sim_x:.4g} {x_units}; model output ≈{sim_y:.5g} {y_units}. Speed setting: {speed:g}×. The visual scale is normalized to the selected analysis interval and is illustrative, not a physical animation.")
    if play:
        st.info("Playback is enabled. Move the simulation frame slider to advance frames; this Streamlit-compatible baseline avoids background timers so the app remains stable on hosted deployments.")
    # lightweight contextual visual with an indicator moving according to normalized x and y
    fig_sim, ax_sim = plt.subplots(figsize=(8, 2.1))
    ax_sim.set_xlim(0, 100); ax_sim.set_ylim(0, 10)
    ax_sim.axis("off")
    pos = frame_count
    ax_sim.plot([5,95],[2,2],linewidth=2)
    ax_sim.scatter([5+0.9*pos],[2],s=220)
    ax_sim.text(50,8,visual,ha="center",fontsize=12)
    ax_sim.text(50,5,f"normalized position {pos}% · f(x) = {sim_y:.4g} {y_units}",ha="center")
    st.pyplot(fig_sim, use_container_width=True)

# Country comparison
st.divider()
st.subheader("4. Same-indicator country comparison")
comparison_frames = []
for cname in comparison_names:
    try:
        cdf, _ = fetch_world_bank(COUNTRIES[cname], code, start_year, end_year)
        cdf["country"] = cname
        comparison_frames.append(cdf)
    except Exception as exc:
        st.warning(f"{cname}: comparison data unavailable ({exc}).")
if comparison_frames:
    if not source_df.empty:
        main_cmp = source_df.copy()
        main_cmp["country"] = country_name
        comparison_frames.insert(0, main_cmp)
    comp = pd.concat(comparison_frames, ignore_index=True)
    fig_cmp, ax_cmp = plt.subplots(figsize=(10,4.8))
    for cname, grp in comp.groupby("country"):
        ax_cmp.plot(grp.year, grp.value, marker=".", linewidth=1.4, label=cname)
    ax_cmp.set_title(f"{indicator_name}: country comparison")
    ax_cmp.set_xlabel("Year"); ax_cmp.set_ylabel(units); ax_cmp.grid(True, alpha=.25); ax_cmp.legend()
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp, use_container_width=True)
    coverage = comp.groupby("country").agg(first_year=("year","min"), last_year=("year","max"), observations=("value","count")).reset_index()
    st.write("Different countries may have different missing years and coverage; gaps are not filled with invented values.")
    st.dataframe(coverage, use_container_width=True)
else:
    st.caption("Choose one or more comparison countries in the sidebar to plot comparisons.")

# Historical chart and downloads
st.divider()
st.subheader("5. Chart and exports")
if model is not None:
    chart_end = max(float(validated.year.max()), float(validated.year.max()+forward_horizon))
    plot_df = validated.copy()
    chart_fig = make_chart(plot_df, model=model, domain=(float(validated.year.min()), chart_end),
                           sample_step=sample_step, query_points=([int(st.session_state["single_result"]["year"])] if "single_result" in st.session_state else list(results["year"]) if not results.empty else []),
                           title=f"{country_name} — {indicator_name} ({units})")
else:
    chart_fig = make_chart(validated, title=f"{country_name} — {indicator_name} ({units})")
st.pyplot(chart_fig, use_container_width=True)
if printer_mode:
    st.markdown("### Printer-friendly summary")
    st.write(f"{country_name} · {indicator_name} · {units}")
    st.write(f"Source: World Bank API; retrieval date {date.today().isoformat()}.")
    st.write(f"Validated observations: {len(validated)}. User-modified scenario: {'Yes' if data_changed else 'No'}.")
    if model is not None:
        st.write(f"Degree {degree} centered polynomial, base year {base_year}; in-sample MAE {metrics['MAE (in-sample)']:.5g}, R² {metrics['R-squared (in-sample)']:.5g}.")
    st.write("Limits: missing data are not imputed; polynomial extrapolation is uncertain and does not establish causation.")

summary = {
    "app": APP_TITLE, "country": country_name, "country_code": COUNTRIES[country_name],
    "indicator": indicator_name, "indicator_code": code, "units": units,
    "definition": definition, "source_url": source_url, "retrieval_date": date.today().isoformat(),
    "requested_years": f"{start_year}-{end_year}", "actual_coverage": f"{validated.year.min()}-{validated.year.max()}" if len(validated) else "none",
    "observations": len(validated), "user_modified_scenario": data_changed,
    "course": course, "degree": degree, "base_year": base_year,
    "equation": f"f(t) = {coeff_text}" if model is not None else "unavailable",
    "metrics": str(metrics), "limitations": "Polynomial extrapolation is uncertain; in-sample metrics are not prediction intervals."
}
working_export = validated.copy()
if model is not None:
    working_export["fitted_value"] = [finite_eval(model, yy) for yy in working_export.year]
    working_export["residual"] = working_export.value - working_export.fitted_value
csv_bytes = dataframe_csv(working_export)
xlsx_bytes = make_xlsx(summary, working_export, results)
pdf_bytes = make_pdf(summary, working_export, results, chart_fig)
d1,d2,d3 = st.columns(3)
with d1: st.download_button("Download CSV", csv_bytes, file_name="historical_data.csv", mime="text/csv")
with d2: st.download_button("Download Excel workbook", xlsx_bytes, file_name="historical_data_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with d3: st.download_button("Download PDF report", pdf_bytes, file_name="historical_data_report.pdf", mime="application/pdf")

st.divider()
st.subheader("6. Source notes and mathematical limitations")
st.markdown("""
- Source: World Bank Indicators API. The API may return null values for years with no reported observation.
- Brazil is a Latin American comparison country, but Portuguese is its principal language; it is not a Spanish-speaking country.
- GDP per capita is GDP divided by population, not average wealth or household income. Crude birth rate is a rate, not a count. Net migration is a net measure, not an emigration count.
- The polynomial uses centered time, t = year − base year. Coefficients are fitted to the current validated working table. Editing the table creates a scenario; reset restores source values.
- R-squared and MAE are in-sample unless explicitly labeled chronological holdout. Neither establishes causation. Polynomial extrapolation is not guaranteed to be realistic, and no 95% prediction interval is claimed.
- The function lab supports polynomial, rational, exponential, logarithmic, and trigonometric expressions. Symbolic derivatives are exact for parsed symbolic expressions; roots, plot values, and interval checks may be numerical approximations.
- Numerical sampling can miss narrow domain gaps or complicated roots. For rational/logarithmic expressions, inspect denominators and logarithm arguments. This classroom tool is not a medical, engineering, or financial decision system.
""")
st.markdown(f"**Indicator source link:** [World Bank indicator page](https://data.worldbank.org/indicator/{code}) · **API URL used:** {resolved_url}")
