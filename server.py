"""
ClimaCredit Q – API for the Lovable app.

Start locally:   uvicorn server:app --reload
Docs:            http://127.0.0.1:8000/docs

Endpoints
  GET  /health        – is the server up
  GET  /parameters    – all assumptions (value + source), for the "Assumptions" panel
  GET  /regions       – the 19 regions (local banks)
  GET  /euribor       – 12-month Euribor (live from the ECB, fallback from parametrar.json)
  POST /price         – interest rate for a loan: Euribor + bank margin + climate add-on
  POST /quantum       – run the quantum model (QAE on a simulator) for one bank
"""
import time
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from climacredit_model import Model, load_params, SECTORS
import json, os

app = FastAPI(title="ClimaCredit Q API", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_MODEL = Model()
REGION_NAMES = {r: r.split(" ", 1)[1] for r in BASE_MODEL.regions}   # "MK15 Ostrobothnia" -> "Ostrobothnia"
NAME_TO_REGION = {v: k for k, v in REGION_NAMES.items()}

ECB_URL = ("https://data-api.ecb.europa.eu/service/data/FM/"
           "M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA?lastNObservations=1&format=jsondata")
_euribor_cache = {"t": 0.0, "value": None, "period": None}


def get_model(overrides: Optional[Dict[str, Any]]):
    if not overrides:
        return BASE_MODEL
    try:
        return Model(load_params(overrides=overrides))
    except KeyError as e:
        raise HTTPException(400, str(e))


def region_key(name: str) -> str:
    if name in BASE_MODEL.regions:
        return name
    if name in NAME_TO_REGION:
        return NAME_TO_REGION[name]
    raise HTTPException(400, f"Unknown region '{name}'. Use one of: {sorted(NAME_TO_REGION)}")


def euribor():
    """12-month Euribor (monthly average) from the ECB Data Portal, cached for 6 hours."""
    if _euribor_cache["value"] is not None and time.time() - _euribor_cache["t"] < 6 * 3600:
        return _euribor_cache["value"], _euribor_cache["period"], "ECB (cached)"
    try:
        r = httpx.get(ECB_URL, timeout=8)
        r.raise_for_status()
        d = r.json()
        obs = next(iter(d["dataSets"][0]["series"].values()))["observations"]
        key = sorted(obs, key=int)[-1]
        value = float(obs[key][0]) / 100
        period = d["structure"]["dimensions"]["observation"][0]["values"][int(key)]["id"]
        _euribor_cache.update(t=time.time(), value=value, period=period)
        return value, period, "ECB Data Portal"
    except Exception:
        return BASE_MODEL.p["euribor_fallback"], None, "fallback (parametrar.json)"


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/parameters")
def parameters():
    with open(os.path.join(os.path.dirname(__file__), "parametrar.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@app.get("/regions")
def regions():
    m = BASE_MODEL
    return [{"id": r, "name": REGION_NAMES[r],
             "loan_mix": {s: round(float(m.shares.loc[r, s]), 3) for s in SECTORS},
             "pd_today": {s: round(float(m.pd_today.loc[r, s]), 4) for s in SECTORS},
             "flood_risk_area": r in m.p["flood_regions"]} for r in m.regions]


@app.get("/euribor")
def euribor_endpoint():
    v, period, source = euribor()
    return {"euribor_12m": v, "period": period, "source": source}


class PriceRequest(BaseModel):
    company_name: str = "Example Oy"
    sector: str = Field("Agriculture", description="Agriculture, Construction or Trade")
    region: str = Field("Ostrobothnia", description="Region name, e.g. Ostrobothnia or Uusimaa")
    amount_eur: float = Field(500_000, gt=0)
    equity_ratio: Optional[float] = Field(None, description="e.g. 0.30 for 30 %")
    net_debt_ebitda: Optional[float] = None
    payment_default: bool = False
    scenario: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = Field(None, description="Change assumptions, e.g. {'pd_agriculture_national': 0.05}")
    compare_regions: bool = True


@app.post("/price")
def price(req: PriceRequest):
    if req.sector not in SECTORS:
        raise HTTPException(400, f"sector must be one of {SECTORS}")
    m = get_model(req.overrides)
    scen = req.scenario or m.p["main_scenario"]
    if scen not in m.p["scenarios"]:
        raise HTTPException(400, f"scenario must be one of {list(m.p['scenarios'])}")
    reg = region_key(req.region)
    eur, period, source = euribor()
    args = dict(sector=req.sector, amount_eur=req.amount_eur, equity_ratio=req.equity_ratio,
                net_debt_ebitda=req.net_debt_ebitda, payment_default=req.payment_default, euribor=eur)

    main = m.price_loan(reg, scenario=scen, **args)
    by_scenario = {s: round(m.price_loan(reg, scenario=s, **args)["climate_addon"], 5)
                   for s in m.p["scenarios"] if s != "Today"}
    bank_today = m.bank_metrics(reg, "Today")
    bank_scen = m.bank_metrics(reg, scen)
    out = {
        "company_name": req.company_name,
        "region_name": REGION_NAMES[reg],
        "euribor_source": source, "euribor_period": period,
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in main.items()},
        "climate_addon_by_scenario": by_scenario,
        "bank": {
            "loan_book_meur": m.p["bank_size_meur"],
            "loan_mix": {s: round(float(m.shares.loc[reg, s]), 3) for s in SECTORS},
            "flood_risk_area": reg in m.p["flood_regions"],
            "today": {k: round(v, 3) for k, v in bank_today.items()},
            "scenario": {k: round(v, 3) for k, v in bank_scen.items()},
        },
        "explanation": (
            f"The climate add-on has two parts: the loan's own climate risk "
            f"({main['climate_addon_own_risk']*100:.2f} p.p., higher expected loss because the "
            f"{req.sector.lower()} PD rises in the '{scen}' scenario) and the cost of the extra capital "
            f"the loan needs in the {REGION_NAMES[reg]} bank's loan book "
            f"({main['climate_addon_concentration']*100:.2f} p.p.)."),
    }
    if req.compare_regions:
        out["same_loan_other_regions"] = sorted(
            [{"region": REGION_NAMES[r],
              "climate_addon": round(m.price_loan(r, scenario=scen, **args)["climate_addon"], 5)}
             for r in m.regions], key=lambda x: -x["climate_addon"])
    return out


class QuantumRequest(BaseModel):
    region: str = "Ostrobothnia"
    scenario: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


@app.post("/quantum")
def quantum(req: QuantumRequest):
    import kvant_v5 as kv          # imported here so the price endpoint starts fast
    m = get_model(req.overrides)
    scen = req.scenario or m.p["main_scenario"]
    reg = region_key(req.region)
    t0 = time.time()
    L, prob = kv.scenario_losses(m, reg, scen)
    thr = kv.LARGE_LOSS_SHARE * m.p["bank_size_meur"]
    exact_p = float(prob[L >= thr].sum())
    p_q, p_lo, p_hi, calls1 = kv.qae(kv.circuit_large_loss(L, thr), epsilon=max(0.1 * exact_p, 1e-4))
    circ_el, lmax = kv.circuit_expected_loss(L)
    el_q, el_lo, el_hi, calls2 = kv.qae(circ_el, epsilon=0.002)
    return {
        "region_name": REGION_NAMES[reg], "scenario": scen,
        "qubits": 2 * kv.N_QUBITS_PER_FACTOR + 1,
        "large_loss_threshold_meur": thr,
        "p_large_loss_exact": exact_p, "p_large_loss_qae": p_q, "p_large_loss_ci": [p_lo, p_hi],
        "expected_loss_exact_meur": float((L * prob).sum()),
        "expected_loss_qae_meur": el_q * lmax, "expected_loss_ci_meur": [el_lo * lmax, el_hi * lmax],
        "oracle_calls": calls1 + calls2,
        "runtime_seconds": round(time.time() - t0, 1),
        "backend": "Qiskit Aer simulator (IterativeAmplitudeEstimation)",
    }
