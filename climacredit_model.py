"""
ClimaCredit Q v6 - classical risk model.

Many small loans per sector (the ASRF model behind Basel's capital rules):
in a given year the economy factor z_e and the climate factor z_c are drawn,
and a fraction p_k(z_e, z_c) of the loans in sector k defaults.
Bank loss L(z) = sum_k  exposure_k * LGD * p_k(z).

New in v6
  * Each local bank's loan book = the region's actual business debt (farm debt, SME interest costs),
    not the number of firms (which made agriculture look far too big in every region).
  * PD level from EU bank data; Finnish defaults (bankruptcies + restructurings) split it between
    sectors and regions, with credibility weighting so small regions are not driven by noise.
  * Physical climate risk beyond floods: good and bad harvest years (2019 / 2018 / 2021) per region,
    regional drought sensitivity, and all 18 significant flood risk areas (river and coastal).
  * The transition shock differs by sector (pattern of the 2022 energy crisis).
  * Sub-industry risk (TOL 3-digit), shrunk towards the sector (empirical Bayes).
  * Loan calculator: adjustable margin, reference rate, loan term and repayment plan.

All assumptions are read from parametrar.json.
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SECTORS = ["Agriculture", "Construction", "Trade"]
FI_CODE = {"Agriculture": "01_jordbruk", "Construction": "F_bygg", "Trade": "G_handel"}

# Eurostat NUTS 2 area of each region (harvest data exist per NUTS 2 area)
NUTS2 = {"MK01": "FI1B",
         "MK02": "FI1C", "MK05": "FI1C", "MK07": "FI1C", "MK08": "FI1C", "MK09": "FI1C",
         "MK04": "FI19", "MK06": "FI19", "MK13": "FI19", "MK14": "FI19", "MK15": "FI19",
         "MK10": "FI1D", "MK11": "FI1D", "MK12": "FI1D", "MK16": "FI1D", "MK17": "FI1D",
         "MK18": "FI1D", "MK19": "FI1D",
         "MK21": "FI"}                      # Åland: no regional harvest series -> whole country
NUTS2_NAMES = {"FI1B": "Helsinki-Uusimaa", "FI1C": "Southern Finland", "FI19": "Western Finland",
               "FI1D": "Northern and Eastern Finland", "FI": "Whole country"}
REPAYMENT_TYPES = ["annuity", "equal_principal", "bullet"]


# ---------------------------------------------------------------- parameters
def load_params(path=None, overrides=None):
    """Read parametrar.json -> dict of plain values. `overrides` replaces single values
    (a dict value is merged, e.g. {"scenarios": {"Run-on-brown": 1.8}})."""
    with open(path or os.path.join(HERE, "parametrar.json"), encoding="utf-8") as f:
        raw = json.load(f)
    p = {k: v["value"] for k, v in raw.items() if not k.startswith("_")}
    for k, v in (overrides or {}).items():
        if k not in p:
            raise KeyError(f"Unknown parameter: {k}")
        if isinstance(p[k], dict) and isinstance(v, dict):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


# ---------------------------------------------------------------- data
@lru_cache(maxsize=4)
def load_data(folder=HERE):
    """All input data (Statistics Finland, Eurostat). Returns a dict of DataFrames - do not modify."""
    def path(name):
        return os.path.join(folder, name)

    # 13ff bankruptcies and 13ww establishments per region and sector (used for regional differences)
    k = pd.read_excel(path("konkurser.xlsx"), header=None, skiprows=3)
    k.columns = ["year", "industry", "region", "bankruptcies"]
    k[["year", "industry"]] = k[["year", "industry"]].ffill()
    k = k[k["region"].notna() & pd.to_numeric(k["bankruptcies"], errors="coerce").notna()]
    k["year"] = k["year"].astype(int)
    k["bankruptcies"] = k["bankruptcies"].astype(int)
    k["sector"] = k["industry"].map({"Agriculture, forestry and fishing": "Agriculture",
                                     "Construction": "Construction", "Trade": "Trade"})
    k = k[k["sector"].notna()]

    f = pd.read_excel(path("foretag.xlsx"), header=None, skiprows=3)
    f.columns = ["year", "region", "industry", "firms", "turnover"]
    f[["year", "region"]] = f[["year", "region"]].ffill()
    f = f[f["industry"].notna()]
    f["year"] = f["year"].astype(int)
    f["firms"] = pd.to_numeric(f["firms"], errors="coerce")
    f["sector"] = f["industry"].map({
        "A Agriculture, forestry and fishing (01-03)": "Agriculture",
        "F Construction (41-43)": "Construction",
        "G Wholesale and retail trade; repair of motor vehicles and motorcycles (45-47)": "Trade"})
    f = f[f["sector"].notna()]

    return {
        "k_reg": k, "f_reg": f,
        "bankr": pd.read_csv(path("konkurser_bransch_2003_2025.csv"), index_col=0),      # 13fg
        "restr": pd.read_csv(path("saneringar_bransch_2003_2025.csv"), index_col=0),     # 13fp
        "firms": pd.read_csv(path("foretag_13w3.csv"), index_col=0),                     # 13w3
        "regions": pd.read_csv(path("regioner_2024.csv"), index_col=0),                  # 141b, 13z5, 13x2
        "debt_base": pd.read_csv(path("exponering_underlag_2024.csv"), index_col=0),     # 13z9, 13w2
        "sub": pd.read_csv(path("underbranscher_2018_2025.csv"), dtype={"kod": str}),    # 13fg, 13w3
        "harvest": pd.read_csv(path("skord_eurostat.csv")),                              # Eurostat
    }


def regional_bankruptcies(d, years=(2018, 2024)):
    """Bankruptcies and bankruptcy rate per region and sector, 2018-2024 (13ff / 13ww)."""
    k, f = d["k_reg"], d["f_reg"]
    x = k[k["year"].between(*years)].merge(f, on=["year", "region", "sector"])
    s = x.groupby(["region", "sector"])[["bankruptcies", "firms"]].sum()
    return s["bankruptcies"].unstack()[SECTORS], (s["bankruptcies"] / s["firms"]).unstack()[SECTORS]


def firm_shares(d, year=2024):
    """Share of establishments per sector in each region (v5 loan mix; now only a fallback)."""
    f = d["f_reg"]
    a = f[f["year"] == year].pivot(index="region", columns="sector", values="firms")[SECTORS]
    return a.div(a.sum(axis=1), axis=0)


def debt_weights(d):
    """National debt proxy per sector, MEUR of interest costs: farms (13z9), SME construction and trade (13w2)."""
    u = d["debt_base"]
    return np.array([float(u.loc["jordbruk_rantekostnader_Me", "hela_landet"]),
                     float(u.loc["bygg_smf_rantekostnader_1000e", "hela_landet"]) / 1000,
                     float(u.loc["handel_smf_rantekostnader_1000e", "hela_landet"]) / 1000])


def exposure_shares(d, region_keys):
    """Loan mix of each local bank = the region's business debt per sector.
    Agriculture: farms x debt per farm. Construction and trade: national SME interest costs split by the
    region's share of the sector's gross output. Åland has no such data -> firm mix re-weighted by debt."""
    reg, W = d["regions"], debt_weights(d)
    whole = reg.loc["SSS"]
    fs = firm_shares(d)
    nat_firm = fs.loc["WHOLE COUNTRY"].values
    rows, basis = {}, {}
    for key in region_keys:
        code = key.split(" ", 1)[0]
        if code in reg.index:
            r = reg.loc[code]
            share = np.array([
                r.antal_gardar * r.jordbruksskuld_per_gard_euro / (whole.antal_gardar * whole.jordbruksskuld_per_gard_euro),
                r.bygg_bruttoproduktion_1000e / whole.bygg_bruttoproduktion_1000e,
                r.handel_bruttoproduktion_1000e / whole.handel_bruttoproduktion_1000e])
            v = W * share
            basis[key] = "debt"
        else:
            v = fs.loc[key].values * (W / W.sum()) / nat_firm
            basis[key] = "firm counts re-weighted by national debt per firm (no regional debt data)"
        rows[key] = v / v.sum()
    return pd.DataFrame(rows, index=SECTORS).T, basis


def national_default_rates(d, years=range(2018, 2025)):
    """Construction and trade: (bankruptcies + restructurings) / firms, 2018-2024."""
    out = {}
    for s in ["Construction", "Trade"]:
        c = FI_CODE[s]
        b, r = d["bankr"].loc[years, c].sum(), d["restr"].loc[years, c].sum()
        n = d["firms"].loc[years, c].sum()
        out[s] = {"bankruptcies": int(b), "restructurings": int(r), "firm_years": int(n),
                  "bankruptcy_rate": b / n, "default_rate": (b + r) / n}
    return out


def harvest_anomalies(d, start=2010):
    """Cereal yield deviation from trend per NUTS 2 area (log, Eurostat): + good year, - bad year.
    Same period (2010-2024) for every area so they are comparable."""
    h = d["harvest"].copy()
    h["log_yield"] = np.log(h["produktion_1000t"] / h["areal_1000ha"])
    out = {}
    for area, g in h[h["ar"] >= start].groupby("region"):
        x = g["ar"].values.astype(float)
        b = np.polyfit(x, g["log_yield"].values, 1)
        out[area] = pd.Series(g["log_yield"].values - np.polyval(b, x), index=g["ar"].values)
    a = pd.DataFrame(out)
    a.index.name = "year"
    return a, a.std(ddof=2)


def transition_sensitivity(d):
    """How much each sector's bankruptcy rate rose in the 2022 energy crisis, relative to the whole
    economy (2018-19 -> 2023-24). 1.0 = same as the economy."""
    b, f = d["bankr"], d["firms"]
    cols = ["totalt"] + [FI_CODE[s] for s in SECTORS]
    pre = b.loc[2018:2019, cols].sum() / f.loc[2018:2019, cols].sum()
    post = b.loc[2023:2024, cols].sum() / f.loc[2023:2024, cols].sum()
    inc = post / pre
    return {s: float(inc[FI_CODE[s]] / inc["totalt"]) for s in SECTORS}


def subindustry_table(d, prior_strength=5, years=range(2019, 2025)):
    """Relative risk of each TOL 3-digit sub-industry within its sector (bankruptcies 2019-2024),
    shrunk towards 1 with a Poisson-gamma (empirical Bayes) prior. Firm-weighted average = 1."""
    u = d["sub"].copy()
    u["D"] = u[[f"konk_{a}" for a in years]].sum(axis=1)
    u["N"] = u[[f"foretag_{a}" for a in years]].sum(axis=1)
    rows = []
    for s in SECTORS:
        g = u[u["bransch"] == FI_CODE[s]]
        rate = g["D"].sum() / g["N"].sum()
        mu = g["N"] * rate                                   # expected bankruptcies at the sector rate
        w = mu / mu.sum()
        tau2 = max(float((w * (g["D"] / mu - 1) ** 2).sum() - (w / mu).sum()), 1e-3)
        alpha = max(1 / tau2, prior_strength)
        rr = (g["D"] + alpha) / (mu + alpha)
        rr = rr / ((g["N"] * rr).sum() / g["N"].sum())
        for (_, r), v in zip(g.iterrows(), rr):
            rows.append({"sector": s, "code": r["kod"], "name": r["namn_en"], "name_sv": r["namn"],
                         "bankruptcies_2019_2024": int(r["D"]), "firm_years_2019_2024": int(r["N"]),
                         "raw_relative_risk": float(r["D"] / r["N"] / rate), "relative_risk": float(v)})
    return pd.DataFrame(rows)


def payment_plan(amount_eur, annual_rate, years, repayment="annuity", payments_per_year=12):
    """Repayment plan at a constant rate. repayment: annuity (same payment every month),
    equal_principal (same amortisation, falling payments) or bullet (interest only, all principal at the end)."""
    if repayment not in REPAYMENT_TYPES:
        raise ValueError(f"repayment must be one of {REPAYMENT_TYPES}")
    n = max(int(round(years * payments_per_year)), 1)
    r = annual_rate / payments_per_year
    bal = float(amount_eur)
    if repayment == "annuity":
        pay = amount_eur * r / (1 - (1 + r) ** -n) if r > 0 else amount_eur / n
    pays, ints, prins, bals = [], [], [], []
    for k in range(1, n + 1):
        interest = bal * r
        if repayment == "annuity":
            principal = pay - interest
        elif repayment == "equal_principal":
            principal = amount_eur / n
        else:
            principal = amount_eur if k == n else 0.0
        principal = min(principal, bal)
        bal -= principal
        pays.append(interest + principal); ints.append(interest); prins.append(principal); bals.append(max(bal, 0.0))
    pays, ints, prins, bals = map(np.array, (pays, ints, prins, bals))
    yearly = []
    for y in range(int(np.ceil(n / payments_per_year))):
        sl = slice(y * payments_per_year, min((y + 1) * payments_per_year, n))
        yearly.append({"year": y + 1, "payments": round(float(pays[sl].sum()), 2),
                       "interest": round(float(ints[sl].sum()), 2), "principal": round(float(prins[sl].sum()), 2),
                       "balance_end": round(float(bals[sl][-1]), 2)})
    return {"repayment_type": repayment, "months": n, "annual_rate": annual_rate,
            "first_payment": round(float(pays[0]), 2), "last_payment": round(float(pays[-1]), 2),
            "total_interest": round(float(ints.sum()), 2), "total_paid": round(float(pays.sum()), 2),
            "yearly": yearly}


# ---------------------------------------------------------------- the model
class Model:
    def __init__(self, params=None, data_folder=HERE, grid_points=161, z_max=5.0, cache_size=160):
        self.p = params or load_params()
        d = self.data = load_data(data_folder)
        self.bankruptcy_counts, self.bankruptcy = regional_bankruptcies(d)
        self.regions = [r for r in self.bankruptcy.index if r != "WHOLE COUNTRY"]
        self.code = {r: r.split(" ", 1)[0] for r in self.regions}
        self.shares, self.exposure_basis = exposure_shares(d, self.regions)
        self.exposure = self.shares * self.p["bank_size_meur"]
        self.national_defaults = national_default_rates(d)
        self.pd_today = self._pd_today()
        self.anomaly, self.harvest_sigma = harvest_anomalies(d)
        self.drought_scale = self._drought_scale()
        self.transition_pattern = transition_sensitivity(d)
        self.transition_weight = self._transition_weights()
        self.subindustries = subindustry_table(d, self.p["subindustry_prior_strength"])
        # factor grid for exact classical results
        g = np.linspace(-z_max, z_max, grid_points)
        w = norm.pdf(g)
        w /= w.sum()
        self.ze, self.zc = np.meshgrid(g, g, indexing="ij")
        self.wgrid = np.outer(w, w)
        self._ze, self._zc, self._w = self.ze.ravel(), self.zc.ravel(), self.wgrid.ravel()
        self._bank_cache, self._cache_size = {}, cache_size

    # ------------------------------------------------------------ calibration
    def _pd_today(self):
        """PD today per region and sector (no climate scenario, normal weather)."""
        p = self.p
        n, br = self.bankruptcy_counts, self.bankruptcy
        z = p["regional_weight"] * n / (n + p["credibility_k"])           # credibility of the region's own data
        rel = 1 + z * (br / br.loc["WHOLE COUNTRY"] - 1)                  # regional relative risk
        dr = self.national_defaults
        W = debt_weights(self.data)
        wc, wt = W[1] / W.sum(), W[2] / W.sum()
        # one scale so that the debt-weighted average of construction and trade equals the EU average PD
        self.pd_calibration_scale = p["pd_construction_trade_average"] * (wc + wt) / (
            wc * dr["Construction"]["default_rate"] + wt * dr["Trade"]["default_rate"])
        self.pd_level = {
            "Agriculture": p["pd_agriculture_national"],
            "Construction": self.pd_calibration_scale * dr["Construction"]["default_rate"] * p["pd_scale_construction_trade"],
            "Trade": self.pd_calibration_scale * dr["Trade"]["default_rate"] * p["pd_scale_construction_trade"]}
        out = rel.copy()
        for s in SECTORS:
            out[s] = self.pd_level[s] * rel[s]
        return out.clip(upper=0.5)

    def _drought_scale(self):
        """Multiplier on the farm climate sensitivity per region: (local harvest volatility / national)^2,
        national = farm-debt-weighted average, partly shrunk (drought_regional_weight)."""
        reg, sig = self.data["regions"], self.harvest_sigma
        debt = {r: reg.loc[c, "antal_gardar"] * reg.loc[c, "jordbruksskuld_per_gard_euro"]
                for r, c in self.code.items() if c in reg.index}
        s2 = {r: float(sig[NUTS2[self.code[r]]]) ** 2 for r in self.regions}
        ref = sum(debt[r] * s2[r] for r in debt) / sum(debt.values())
        w = self.p["drought_regional_weight"]
        return {r: 1 + w * (s2[r] / ref - 1) for r in self.regions}

    def _transition_weights(self):
        """Sector weights of the transition shock. Debt-weighted average = 1, so the EU average is kept."""
        raw = np.array([max(self.transition_pattern[s] - 1, 0.0) for s in SECTORS])
        W = debt_weights(self.data)
        W = W / W.sum()
        pattern = raw / (W @ raw) if W @ raw > 0 else np.ones(len(SECTORS))
        lam = self.p["transition_sector_tilt"]
        return (1 - lam) + lam * pattern

    # ------------------------------------------------------------ scenario pieces
    def transition_multiplier(self, scenario):
        m = self.p["scenarios"][scenario]
        return np.maximum(1 + (m - 1) * self.transition_weight, 0.05)

    def weather_year(self, weather):
        """None / 'Normal year' -> None; a label from parametrar.json or a year 2010-2024 -> that year."""
        if weather is None or weather == "" or weather == 0:
            return None
        wy = self.p["weather_years"]
        if isinstance(weather, str):
            if weather in wy:
                return None if wy[weather] is None else int(wy[weather])
            if not weather.strip().isdigit():
                raise ValueError(f"Unknown weather year '{weather}'. Use one of {list(wy)} or a year "
                                 f"{self.anomaly.index.min()}-{self.anomaly.index.max()}.")
            weather = int(weather)
        y = int(weather)
        if y not in self.anomaly.index:
            raise ValueError(f"No harvest data for {y}. Use {self.anomaly.index.min()}-{self.anomaly.index.max()}.")
        return y

    def harvest_deviation(self, region, weather):
        y = self.weather_year(weather)
        return 0.0 if y is None else float(self.anomaly.loc[y, NUTS2[self.code[region]]])

    def weather_multiplier(self, region, weather):
        """PD multiplier from the weather year: farms only, exp(-beta x harvest deviation)."""
        m = np.ones(len(SECTORS))
        m[0] = np.exp(-self.p["weather_beta"] * self.harvest_deviation(region, weather))
        return m

    def flood_types(self, region):
        return sorted({a["type"] for a in self.p["flood_areas"].get(region, [])})

    def rho(self, region):
        """Factor loadings (economy, climate) per sector for a region."""
        p = self.p
        rm = np.array([p["rho_economy"][s] for s in SECTORS], dtype=float)
        rk = np.array([p["rho_climate"][s] for s in SECTORS], dtype=float)
        rk[0] *= self.drought_scale[region]
        types = self.flood_types(region)
        if "river" in types:
            rk += np.array([p["flood_rho_add"][s] for s in SECTORS])
        if "coastal" in types:
            rk += np.array([p["coastal_flood_rho_add"][s] for s in SECTORS])
        return rm, np.minimum(rk, 0.95 - rm)

    def scenario_shift(self, region, scenario, weather=None):
        """How far each sector's risk moves in the scenario, on the probit scale. In the Vasicek model this
        is exactly a shift of the systematic factor, so it is the consistent way to stress a PD. Calibrated
        so the sector's average firm gets PD x (transition multiplier) x (weather multiplier); riskier firms
        move less in relative terms, so stacked stresses do not explode."""
        lv = np.array([self.pd_level[s] for s in SECTORS], dtype=float)
        mult = self.transition_multiplier(scenario) * self.weather_multiplier(region, weather)
        return norm.ppf(np.minimum(lv * mult, 0.99)) - norm.ppf(lv)

    def pd_scenario(self, region, scenario, weather=None, pd_mult=1.0):
        """PD per sector in a scenario (pd_mult = firm and sub-industry multiplier on today's PD)."""
        base = np.clip(self.pd_today.loc[region].values * pd_mult, 1e-6, 0.99)
        return np.minimum(norm.cdf(norm.ppf(base) + self.scenario_shift(region, scenario, weather)), 0.99)

    # ------------------------------------------------------------ core formulas
    @staticmethod
    def cond_pd(pd_, rm, rk, ze, zc):
        """PD given the economy and climate factors (Vasicek with two factors)."""
        return norm.cdf((norm.ppf(pd_) - np.sqrt(rm) * ze - np.sqrt(rk) * zc) / np.sqrt(1 - rm - rk))

    def loss_grid(self, region, scenario, weather=None, ze=None, zc=None):
        """Bank loss L(z) in MEUR on the factor grid (or at the given factor values)."""
        ze = self.ze if ze is None else ze
        zc = self.zc if zc is None else zc
        pdv = self.pd_scenario(region, scenario, weather)
        rm, rk = self.rho(region)
        e = self.exposure.loc[region].values
        L = np.zeros(np.broadcast(ze, zc).shape, dtype=float)
        for i in range(len(SECTORS)):
            L = L + e[i] * self.p["lgd"] * self.cond_pd(pdv[i], rm[i], rk[i], ze, zc)
        return L

    @staticmethod
    def _var(values, weights, level):
        o = np.argsort(values)
        c = np.cumsum(weights[o])
        return float(values[o][min(np.searchsorted(c, level), len(c) - 1)])

    @staticmethod
    def _es(values, weights, level):
        """Expected shortfall: average loss in the worst (1 - level) of years (smooth in the inputs)."""
        o = np.argsort(values)[::-1]
        v, w = values[o], weights[o]
        tail = 1 - level
        c = np.cumsum(w)
        k = min(int(np.searchsorted(c, tail)), len(v) - 1)
        used = w[:k].sum()
        return float(((v[:k] * w[:k]).sum() + v[k] * (tail - used)) / tail)

    def _bank(self, region, scenario, weather=None):
        """Loss distribution of a bank (cached)."""
        key = (region, scenario, self.weather_year(weather))
        b = self._bank_cache.get(key)
        if b is None:
            L = self.loss_grid(region, scenario, weather).ravel()
            lvl = self.p["var_level"]
            b = {"L": L, "el": float(L @ self._w), "var": self._var(L, self._w, lvl), "es": self._es(L, self._w, lvl)}
            if len(self._bank_cache) >= self._cache_size:
                self._bank_cache.pop(next(iter(self._bank_cache)))
            self._bank_cache[key] = b
        return b

    def bank_metrics(self, region, scenario, weather=None):
        b = self._bank(region, scenario, weather)
        return {"expected_loss_meur": b["el"], "var_meur": b["var"], "es_meur": b["es"],
                "capital_meur": b["es"] - b["el"]}

    # ------------------------------------------------------------ firm and pricing
    def firm_multiplier(self, equity_ratio=None, net_debt_ebitda=None, payment_default=False):
        fa = self.p["firm_adjustment"]
        if payment_default:
            return fa["payment_default"], "payment default"
        if equity_ratio is None and net_debt_ebitda is None:
            return fa["satisfactory"], "no key ratios given"
        lo_e, hi_e = self.p["equity_ratio_limits"]
        lo_d, hi_d = self.p["net_debt_ebitda_limits"]
        grades = []
        if equity_ratio is not None:
            grades.append("strong" if equity_ratio > hi_e else "weak" if equity_ratio < lo_e else "satisfactory")
        if net_debt_ebitda is not None:
            grades.append("strong" if net_debt_ebitda < lo_d else "weak" if net_debt_ebitda > hi_d else "satisfactory")
        if "weak" in grades:
            g = "weak"
        elif all(x == "strong" for x in grades):
            g = "strong"
        else:
            g = "satisfactory"
        return fa[g], g

    def subindustry_multiplier(self, sector, code=None):
        """Relative risk of a TOL sub-industry (e.g. '412'); (1.0, None) if not given."""
        if code in (None, "", "all"):
            return 1.0, None
        t = self.subindustries
        row = t[(t["sector"] == sector) & (t["code"] == str(code).strip())]
        if row.empty:
            raise ValueError(f"Sub-industry '{code}' is not in {sector}. "
                             f"Use one of {list(t[t['sector'] == sector]['code'])}.")
        return float(row["relative_risk"].iloc[0]), str(row["name"].iloc[0])

    def _loan_risk(self, region, scenario, weather, sector, amount_meur, pd_mult):
        """Expected loss and extra capital (Delta ES - Delta EL) from adding one loan to the bank.
        The new loan is a single borrower: it defaults completely or not at all."""
        i = SECTORS.index(sector)
        rm, rk = self.rho(region)
        pd_firm = float(self.pd_scenario(region, scenario, weather, pd_mult)[i])
        b = self._bank(region, scenario, weather)
        L, w = b["L"], self._w
        q = self.cond_pd(pd_firm, rm[i], rk[i], self._ze, self._zc)
        a = amount_meur * self.p["lgd"]
        el1 = b["el"] + a * float(q @ w)
        # Only grid points that can lie in the worst 0.1 % after adding the loan (exact: the tail
        # threshold can only move up, so points with L < VaR - a can never enter it).
        m = L >= b["var"] - a
        vals = np.concatenate([L[m] + a, L[m]])
        wts = np.concatenate([w[m] * q[m], w[m] * (1 - q[m])])
        es1 = self._es(vals, wts, self.p["var_level"])
        return {"pd_firm": pd_firm,
                "el_rate": (el1 - b["el"]) / amount_meur,
                "capital_rate": ((es1 - el1) - (b["es"] - b["el"])) / amount_meur}

    def climate_addon(self, region, sector, amount_eur, scenario, weather=None, pd_mult=1.0, today=None):
        """Raw climate add-on (can be negative in a good year) = own risk + concentration, vs today."""
        amount = amount_eur / 1e6
        today = today or self._loan_risk(region, "Today", None, sector, amount, pd_mult)
        scen = self._loan_risk(region, scenario, weather, sector, amount, pd_mult)
        own = scen["el_rate"] - today["el_rate"]
        conc = self.p["cost_of_capital"] * (scen["capital_rate"] - today["capital_rate"])
        return {"own": own, "concentration": conc, "total": own + conc,
                "pd_today": today["pd_firm"], "pd_scenario": scen["pd_firm"], "today": today}

    def price_loan(self, region, sector, amount_eur, scenario=None, weather=None, equity_ratio=None,
                   net_debt_ebitda=None, payment_default=False, sub_industry=None, euribor=None,
                   bank_margin=None):
        """Interest rate = reference rate (Euribor) + bank margin + climate add-on."""
        p = self.p
        scenario = scenario or p["main_scenario"]
        fmult, grade = self.firm_multiplier(equity_ratio, net_debt_ebitda, payment_default)
        smult, sname = self.subindustry_multiplier(sector, sub_industry)
        mult = fmult * smult
        full = self.climate_addon(region, sector, amount_eur, scenario, weather, mult)
        if self.weather_year(weather) is None:
            transition = full
        else:
            transition = self.climate_addon(region, sector, amount_eur, scenario, None, mult, today=full["today"])
        add_on = max(full["total"], 0.0)
        euribor = p["euribor_fallback"]["12m"] if euribor is None else euribor
        margin = p["bank_margin"] if bank_margin is None else bank_margin
        i = SECTORS.index(sector)
        return {
            "region": region, "sector": sector, "amount_eur": amount_eur, "scenario": scenario,
            "weather_year": self.weather_year(weather),
            "sub_industry": sub_industry or None, "sub_industry_name": sname, "sub_industry_multiplier": smult,
            "firm_grade": grade, "firm_pd_multiplier": fmult,
            "pd_today": full["pd_today"], "pd_scenario": full["pd_scenario"],
            "transition_multiplier": float(self.transition_multiplier(scenario)[i]),
            "weather_multiplier": float(self.weather_multiplier(region, weather)[i]),
            "harvest_deviation": self.harvest_deviation(region, weather),
            "euribor": euribor, "bank_margin": margin,
            "climate_addon_own_risk": full["own"], "climate_addon_concentration": full["concentration"],
            "climate_addon_transition": transition["total"],
            "climate_addon_weather": full["total"] - transition["total"],
            "climate_addon_raw": full["total"],
            "climate_addon": add_on,
            "total_rate": euribor + margin + add_on,
        }


if __name__ == "__main__":
    m = Model()
    for reg in ["MK14 South Ostrobothnia", "MK01 Uusimaa"]:
        for sc in m.p["scenarios"]:
            print(reg, sc, {k: round(v, 3) for k, v in m.bank_metrics(reg, sc).items()})
    for reg in ["MK14 South Ostrobothnia", "MK01 Uusimaa"]:
        for w in m.p["weather_years"]:
            r = m.price_loan(reg, "Agriculture", 500_000, weather=w, equity_ratio=0.3, net_debt_ebitda=4)
            print(reg, w, "own %.3f%%  conc %.3f%%  weather %.3f%%  add-on %.3f%%  total %.2f%%" % (
                r["climate_addon_own_risk"] * 100, r["climate_addon_concentration"] * 100,
                r["climate_addon_weather"] * 100, r["climate_addon"] * 100, r["total_rate"] * 100))
