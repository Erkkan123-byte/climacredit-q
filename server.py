"""
ClimaCredit Q v6 - API for the Lovable app.

Start locally:   uvicorn server:app --reload
Docs:            http://127.0.0.1:8000/docs

Endpoints
  GET  /health          - is the server up
  GET  /parameters      - all assumptions (value + source), for the "Assumptions" panel
  GET  /regions         - the 19 regions (local banks) with loan mix, flood areas and drought sensitivity
  GET  /banks           - all 19 local banks: capital need today vs scenario (for a comparison table/map)
  GET  /euribor         - Euribor 1, 3, 6 and 12 months (live from the ECB, fallback from parametrar.json)
  GET  /scenarios       - transition scenarios with the PD multiplier per sector
  GET  /weather_years   - good / normal / bad harvest years and the harvest deviation per area and year
  GET  /sub_industries  - TOL sub-industries per sector with their relative risk
  POST /price           - interest rate for a loan: reference rate + bank margin + climate add-on, plus repayment plan
  POST /payment_plan    - repayment plan for any amount, rate and term
  POST /quantum         - run the quantum model (QAE on a simulator) for one bank

All rates are decimals (0.0525 = 5.25 %). Money in EUR unless the field name says meur.
"""
import json
import os
import time
from collections import OrderedDict
from typing import Optional, Dict, Any, Union

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from climacredit_model import (Model, load_params, payment_plan, SECTORS, NUTS2, NUTS2_NAMES,
                               REPAYMENT_TYPES)

app = FastAPI(title="ClimaCredit Q API", version="6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_MODEL = Model()
REGION_NAMES = {r: r.split(" ", 1)[1] for r in BASE_MODEL.regions}   # "MK14 South Ostrobothnia" -> "South Ostrobothnia"
NAME_TO_REGION = {v.lower(): k for k, v in REGION_NAMES.items()}

# ------------------------------------------------------------------ reference rates (ECB)
REFERENCE_RATES = {   # key: (fallback key, ECB series, label)
    "euribor_12m": ("12m", "EURIBOR1YD_", "12-month Euribor"),
    "euribor_6m": ("6m", "EURIBOR6MD_", "6-month Euribor"),
    "euribor_3m": ("3m", "EURIBOR3MD_", "3-month Euribor"),
    "euribor_1m": ("1m", "EURIBOR1MD_", "1-month Euribor"),
}
ECB_URL = "https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.{series}.HSTA?lastNObservations=1&format=jsondata"
_rate_cache: Dict[str, tuple] = {}
_ecb_down_until = 0.0


def reference_key(name: Optional[str]) -> str:
    """'euribor_12m', '12m', '12-month Euribor', '12 months' ... -> 'euribor_12m'."""
    if not name:
        return "euribor_12m"
    n = "".join(ch for ch in name.lower() if ch.isalnum())
    n = n.replace("euribor", "").replace("months", "m").replace("month", "m")
    table = {"12m": "euribor_12m", "12": "euribor_12m", "1y": "euribor_12m", "6m": "euribor_6m", "6": "euribor_6m",
             "3m": "euribor_3m", "3": "euribor_3m", "1m": "euribor_1m", "1": "euribor_1m"}
    if n not in table:
        raise HTTPException(400, f"reference_rate must be one of {list(REFERENCE_RATES)}")
    return table[n]


def reference_rate(key="euribor_12m", params=None):
    """Monthly average Euribor from the ECB Data Portal, cached 6 hours. Returns (value, period, source)."""
    global _ecb_down_until
    fb_key, series, _ = REFERENCE_RATES[key]
    cached = _rate_cache.get(key)
    if cached and time.time() - cached[0] < 6 * 3600:
        return cached[1], cached[2], "ECB Data Portal (cached)"
    if time.time() > _ecb_down_until:
        try:
            r = httpx.get(ECB_URL.format(series=series), timeout=6)
            r.raise_for_status()
            d = r.json()
            obs = next(iter(d["dataSets"][0]["series"].values()))["observations"]
            k = sorted(obs, key=int)[-1]
            value = float(obs[k][0]) / 100
            period = d["structure"]["dimensions"]["observation"][0]["values"][int(k)]["id"]
            _rate_cache[key] = (time.time(), value, period)
            return value, period, "ECB Data Portal"
        except Exception:
            _ecb_down_until = time.time() + 600          # do not wait for the ECB again for 10 minutes
    if cached:
        return cached[1], cached[2], "ECB Data Portal (older value)"
    fb = (params or BASE_MODEL.p)["euribor_fallback"]
    return float(fb[fb_key] if isinstance(fb, dict) else fb), "2026-09", "fallback (parametrar.json)"


# ------------------------------------------------------------------ helpers
_model_cache: "OrderedDict[str, Model]" = OrderedDict()


def get_model(overrides: Optional[Dict[str, Any]]) -> Model:
    """The base model, or a model with changed assumptions (the last 6 are kept in memory)."""
    if not overrides:
        return BASE_MODEL
    key = json.dumps(overrides, sort_keys=True)
    if key in _model_cache:
        _model_cache.move_to_end(key)
        return _model_cache[key]
    try:
        m = Model(load_params(overrides=overrides), cache_size=64)
        m.bank_metrics(m.regions[0], "Today")              # fails early on bad values
    except KeyError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not use these overrides: {e}")
    _model_cache[key] = m
    if len(_model_cache) > 6:
        _model_cache.popitem(last=False)
    return m


def region_key(name: str) -> str:
    if name in BASE_MODEL.regions:
        return name
    if name.lower() in NAME_TO_REGION:
        return NAME_TO_REGION[name.lower()]
    for k in BASE_MODEL.regions:                            # "MK14" also works
        if k.split(" ", 1)[0].lower() == name.lower():
            return k
    raise HTTPException(400, f"Unknown region '{name}'. Use one of: {sorted(REGION_NAMES.values())}")


def as_decimal(x: Optional[float]) -> Optional[float]:
    """Accept both 0.02 and 2 (meaning 2 %)."""
    if x is None:
        return None
    return x / 100 if abs(x) > 0.5 else x


def weather_label(m: Model, weather) -> str:
    y = m.weather_year(weather)
    if y is None:
        return "Normal year"
    for label, year in m.p["weather_years"].items():
        if year == y:
            return f"{label} (like {y})"
    return f"Weather of {y}"


def check_weather(m: Model, weather):
    try:
        return m.weather_year(weather)
    except ValueError as e:
        raise HTTPException(400, str(e))


def physical_risk(m: Model, reg: str, weather=None) -> dict:
    area = NUTS2[m.code[reg]]
    a = m.anomaly[area]
    worst = a.sort_values().head(3)
    return {
        "harvest_area": area, "harvest_area_name": NUTS2_NAMES[area],
        "harvest_volatility": round(float(m.harvest_sigma[area]), 4),
        "drought_sensitivity": round(float(m.drought_scale[reg]), 3),
        "worst_harvest_years": [{"year": int(y), "deviation": round(float(np.expm1(v)), 4)} for y, v in worst.items()],
        "best_harvest_year": {"year": int(a.idxmax()), "deviation": round(float(np.expm1(a.max())), 4)},
        "selected_year_harvest_deviation": round(float(np.expm1(m.harvest_deviation(reg, weather))), 4),
        "flood_risk_area": bool(m.p["flood_areas"].get(reg)),
        "flood_areas": m.p["flood_areas"].get(reg, []),
        "climate_sensitivity": dict(zip(SECTORS, [round(float(v), 4) for v in m.rho(reg)[1]])),
    }



def r6(x):
    return round(float(x), 6) if isinstance(x, (float, np.floating)) else x


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "version": "6.0"}


@app.get("/parameters")
def parameters():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parametrar.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@app.get("/regions")
def regions():
    m = BASE_MODEL
    return [{"id": r, "name": REGION_NAMES[r],
             "loan_mix": {s: round(float(m.shares.loc[r, s]), 3) for s in SECTORS},
             "loan_mix_basis": m.exposure_basis[r],
             "pd_today": {s: round(float(m.pd_today.loc[r, s]), 4) for s in SECTORS},
             "flood_risk_area": bool(m.p["flood_areas"].get(r)),
             "flood_areas": m.p["flood_areas"].get(r, []),
             "harvest_area": NUTS2_NAMES[NUTS2[m.code[r]]],
             "drought_sensitivity": round(float(m.drought_scale[r]), 3)} for r in m.regions]


@app.get("/banks")
def banks(scenario: Optional[str] = None, weather_year: Optional[str] = None):
    m = BASE_MODEL
    scen = scenario or m.p["main_scenario"]
    if scen not in m.p["scenarios"]:
        raise HTTPException(400, f"scenario must be one of {list(m.p['scenarios'])}")
    check_weather(m, weather_year)
    out = []
    for r in m.regions:
        t, s = m.bank_metrics(r, "Today"), m.bank_metrics(r, scen, weather_year)
        out.append({"region": REGION_NAMES[r], "id": r,
                    "agriculture_share": round(float(m.shares.loc[r, "Agriculture"]), 3),
                    "flood_risk_area": bool(m.p["flood_areas"].get(r)),
                    "drought_sensitivity": round(float(m.drought_scale[r]), 3),
                    "expected_loss_today_meur": round(t["expected_loss_meur"], 3),
                    "expected_loss_scenario_meur": round(s["expected_loss_meur"], 3),
                    "capital_today_meur": round(t["capital_meur"], 3),
                    "capital_scenario_meur": round(s["capital_meur"], 3),
                    "capital_increase_pct": round(s["capital_meur"] / t["capital_meur"] - 1, 4)})
    return {"scenario": scen, "weather_year": weather_label(m, weather_year),
            "loan_book_meur": m.p["bank_size_meur"],
            "banks": sorted(out, key=lambda x: -x["capital_increase_pct"])}


@app.get("/euribor")
def euribor_endpoint():
    out = {}
    for key, (_, _, label) in REFERENCE_RATES.items():
        v, period, source = reference_rate(key)
        out[key] = {"value": v, "period": period, "source": source, "label": label}
    v12 = out["euribor_12m"]
    # v5 fields kept for the existing app
    return {"euribor_12m": v12["value"], "period": v12["period"], "source": v12["source"], "rates": out}


@app.get("/scenarios")
def scenarios():
    m = BASE_MODEL
    return {"main_scenario": m.p["main_scenario"],
            "scenarios": [{"name": s, "eu_average_multiplier": v,
                           "pd_multiplier_by_sector": dict(zip(SECTORS, [round(float(x), 3) for x in m.transition_multiplier(s)]))}
                          for s, v in m.p["scenarios"].items()],
            "source": "ECB/ESAs Fit-for-55 climate scenario analysis (Nov 2024); sector split from the 2022 energy crisis."}


@app.get("/weather_years")
def weather_years():
    m = BASE_MODEL
    a = m.anomaly
    return {
        "options": [{"label": lbl, "year": yr,
                     "national_harvest_deviation": None if yr is None else round(float(np.expm1(a.loc[yr, "FI"])), 4),
                     "harvest_deviation_by_area": None if yr is None else
                     {NUTS2_NAMES[c]: round(float(np.expm1(a.loc[yr, c])), 4) for c in a.columns}}
                    for lbl, yr in m.p["weather_years"].items()],
        "all_years": {int(y): {NUTS2_NAMES[c]: round(float(np.expm1(a.loc[y, c])), 4) for c in a.columns}
                      for y in a.index},
        "pd_formula": f"farm PD x exp(-{m.p['weather_beta']} x log harvest deviation)",
        "source": "Eurostat apro_cpnhr (cereals, 2010-2024), deviation from each area's own trend.",
    }


@app.get("/sub_industries")
def sub_industries(sector: Optional[str] = None):
    t = BASE_MODEL.subindustries
    if sector:
        if sector not in SECTORS:
            raise HTTPException(400, f"sector must be one of {SECTORS}")
        t = t[t["sector"] == sector]
    return [{"sector": r.sector, "code": r.code, "name": r.name, "relative_risk": round(r.relative_risk, 3),
             "bankruptcies_2019_2024": r.bankruptcies_2019_2024, "firm_years_2019_2024": r.firm_years_2019_2024}
            for r in t.sort_values(["sector", "relative_risk"], ascending=[True, False]).itertuples()]


class PlanRequest(BaseModel):
    amount_eur: float = Field(500_000, gt=0)
    annual_rate: float = Field(0.06, description="0.06 = 6 % (6 is also accepted)")
    years: float = Field(5, gt=0, le=40)
    repayment_type: str = Field("annuity", description="annuity | equal_principal | bullet")


@app.post("/payment_plan")
def plan(req: PlanRequest):
    if req.repayment_type not in REPAYMENT_TYPES:
        raise HTTPException(400, f"repayment_type must be one of {REPAYMENT_TYPES}")
    return payment_plan(req.amount_eur, as_decimal(req.annual_rate), req.years, req.repayment_type)


class PriceRequest(BaseModel):
    company_name: str = "Example Oy"
    sector: str = Field("Agriculture", description="Agriculture, Construction or Trade")
    region: str = Field("South Ostrobothnia", description="Region name, e.g. South Ostrobothnia or Uusimaa")
    amount_eur: float = Field(500_000, gt=0)
    equity_ratio: Optional[float] = Field(None, description="e.g. 0.30 for 30 %")
    net_debt_ebitda: Optional[float] = None
    payment_default: bool = False
    scenario: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = Field(None, description="Change assumptions, e.g. {'pd_agriculture_national': 0.05}")
    compare_regions: bool = True
    # new in v6
    bank_margin: Optional[float] = Field(None, description="Bank margin, 0.02 = 2 % (2 is also accepted). Default from parametrar.json")
    reference_rate: Optional[str] = Field("euribor_12m", description="euribor_12m | euribor_6m | euribor_3m | euribor_1m")
    reference_rate_value: Optional[float] = Field(None, description="Use this reference rate instead of the live Euribor, e.g. 0.025")
    weather_year: Optional[Union[int, str]] = Field(None, description="'Good year' | 'Normal year' | 'Bad year' | 'Very bad year' or a year 2010-2024")
    sub_industry: Optional[str] = Field(None, description="TOL 3-digit code from /sub_industries, e.g. '412'")
    loan_term_years: float = Field(5, gt=0, le=40)
    repayment_type: str = Field("annuity", description="annuity | equal_principal | bullet")
    include_matrix: bool = True


@app.post("/price")
def price(req: PriceRequest):
    if req.sector not in SECTORS:
        raise HTTPException(400, f"sector must be one of {SECTORS}")
    if req.repayment_type not in REPAYMENT_TYPES:
        raise HTTPException(400, f"repayment_type must be one of {REPAYMENT_TYPES}")
    m = get_model(req.overrides)
    scen = req.scenario or m.p["main_scenario"]
    if scen not in m.p["scenarios"]:
        raise HTTPException(400, f"scenario must be one of {list(m.p['scenarios'])}")
    reg = region_key(req.region)
    weather = req.weather_year
    check_weather(m, weather)
    ref_key = reference_key(req.reference_rate)
    if req.reference_rate_value is not None:
        ref, period, source = as_decimal(req.reference_rate_value), None, "entered by user"
    else:
        ref, period, source = reference_rate(ref_key, m.p)
    margin = as_decimal(req.bank_margin) if req.bank_margin is not None else m.p["bank_margin"]
    equity = req.equity_ratio / 100 if (req.equity_ratio is not None and req.equity_ratio > 1.5) else req.equity_ratio
    args = dict(sector=req.sector, amount_eur=req.amount_eur, equity_ratio=equity,
                net_debt_ebitda=req.net_debt_ebitda, payment_default=req.payment_default,
                sub_industry=req.sub_industry, euribor=ref, bank_margin=margin)
    try:
        main = m.price_loan(reg, scenario=scen, weather=weather, **args)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # good / bad year and scenario sensitivity (raw add-ons, can be negative in a good year)
    fmult, _ = m.firm_multiplier(equity, req.net_debt_ebitda, req.payment_default)
    smult, _ = m.subindustry_multiplier(req.sector, req.sub_industry)
    today = m._loan_risk(reg, "Today", None, req.sector, req.amount_eur / 1e6, fmult * smult)

    def addon(s, w):
        return m.climate_addon(reg, req.sector, req.amount_eur, s, w, fmult * smult, today=today)["total"]

    by_scenario = {s: round(max(addon(s, weather), 0.0), 5) for s in m.p["scenarios"] if s != "Today"}
    by_weather = {lbl: round(addon(scen, lbl), 5) for lbl in m.p["weather_years"]}
    matrix = ({s: {lbl: round(addon(s, lbl), 5) for lbl in m.p["weather_years"]} for s in m.p["scenarios"]}
              if req.include_matrix else None)

    # repayment plan with and without the climate add-on
    plan_with = payment_plan(req.amount_eur, main["total_rate"], req.loan_term_years, req.repayment_type)
    plan_without = payment_plan(req.amount_eur, main["total_rate"] - main["climate_addon"], req.loan_term_years,
                                req.repayment_type)

    own_part = min(max(main["climate_addon_own_risk"], 0.0), main["climate_addon"])
    bank_today = m.bank_metrics(reg, "Today")
    bank_scen = m.bank_metrics(reg, scen, weather)
    wl = weather_label(m, weather)
    sector_l = req.sector.lower()
    pr = physical_risk(m, reg, weather)
    expl = (f"The climate add-on of {main['climate_addon']*100:.2f} % has two parts. "
            f"1) The loan's own climate risk: the {sector_l} default probability rises from "
            f"{main['pd_today']*100:.1f} % today to {main['pd_scenario']*100:.1f} % in the '{scen}' scenario"
            f"{'' if wl == 'Normal year' else ' in a ' + wl.lower()}, which adds "
            f"{main['climate_addon_own_risk']*100:.2f} p.p. of expected loss. "
            f"2) Capital: the loan needs more of the {REGION_NAMES[reg]} bank's capital in that scenario, "
            f"which costs {main['climate_addon_concentration']*100:.2f} p.p.")
    if wl != "Normal year" and req.sector == "Agriculture":
        dev = pr["selected_year_harvest_deviation"]
        expl += (f" Weather: in {m.weather_year(weather)} the harvest in {pr['harvest_area_name']} was "
                 f"{abs(dev)*100:.0f} % {'below' if dev < 0 else 'above'} trend, which changes the add-on by "
                 f"{main['climate_addon_weather']*100:+.2f} p.p.")
    if main["climate_addon_raw"] < 0:
        expl += " The climate risk is lower than today, but the add-on is never negative (no discount)."

    out = {
        "company_name": req.company_name,
        "region_name": REGION_NAMES[reg],
        "reference_rate": ref_key, "reference_rate_label": REFERENCE_RATES[ref_key][2],
        "euribor_source": source, "euribor_period": period,
        "weather_label": wl,
        **{k: r6(v) for k, v in main.items()},
        "rate_breakdown": [
            {"part": REFERENCE_RATES[ref_key][2], "value": r6(ref)},
            {"part": "Bank margin", "value": r6(margin)},
            {"part": "Climate add-on: own risk", "value": r6(own_part)},
            {"part": "Climate add-on: capital", "value": r6(main["climate_addon"] - own_part)},
        ],
        "climate_addon_by_scenario": by_scenario,
        "climate_addon_by_weather": by_weather,
        "addon_matrix": matrix,
        "loan": {
            "term_years": req.loan_term_years, "repayment_type": req.repayment_type,
            "monthly_payment_first": plan_with["first_payment"], "monthly_payment_last": plan_with["last_payment"],
            "monthly_payment_first_without_climate": plan_without["first_payment"],
            "total_interest": plan_with["total_interest"],
            "total_interest_without_climate": plan_without["total_interest"],
            "climate_cost_total_eur": round(plan_with["total_interest"] - plan_without["total_interest"], 2),
            "total_paid": plan_with["total_paid"],
            "yearly": plan_with["yearly"],
        },
        "bank": {
            "loan_book_meur": m.p["bank_size_meur"],
            "loan_mix": {s: round(float(m.shares.loc[reg, s]), 3) for s in SECTORS},
            "loan_mix_basis": m.exposure_basis[reg],
            "flood_risk_area": pr["flood_risk_area"],
            "today": {k: round(v, 3) for k, v in bank_today.items()},
            "scenario": {k: round(v, 3) for k, v in bank_scen.items()},
            "capital_increase_pct": round(bank_scen["capital_meur"] / bank_today["capital_meur"] - 1, 4),
            "expected_loss_increase_pct": round(bank_scen["expected_loss_meur"] / bank_today["expected_loss_meur"] - 1, 4),
        },
        "physical_risk": pr,
        "explanation": expl,
    }
    if req.compare_regions:
        rows = []
        for r in m.regions:
            x = m.price_loan(r, scenario=scen, weather=weather, **args)
            rows.append({"region": REGION_NAMES[r], "climate_addon": round(x["climate_addon"], 5),
                         "flood_risk_area": bool(m.p["flood_areas"].get(r)),
                         "agriculture_share": round(float(m.shares.loc[r, "Agriculture"]), 3)})
        out["same_loan_other_regions"] = sorted(rows, key=lambda x: -x["climate_addon"])
    return out


class QuantumRequest(BaseModel):
    region: str = "South Ostrobothnia"
    scenario: Optional[str] = None
    weather_year: Optional[Union[int, str]] = None
    overrides: Optional[Dict[str, Any]] = None


@app.post("/quantum")
def quantum(req: QuantumRequest):
    import kvant_v6 as kv          # imported here so the other endpoints start fast
    m = get_model(req.overrides)
    scen = req.scenario or m.p["main_scenario"]
    if scen not in m.p["scenarios"]:
        raise HTTPException(400, f"scenario must be one of {list(m.p['scenarios'])}")
    reg = region_key(req.region)
    check_weather(m, req.weather_year)
    t0 = time.time()
    res = kv.run_one(m, reg, scen, req.weather_year)
    res.update({"region_name": REGION_NAMES[reg], "scenario": scen,
                "weather_label": weather_label(m, req.weather_year),
                "runtime_seconds": round(time.time() - t0, 1)})
    return res
