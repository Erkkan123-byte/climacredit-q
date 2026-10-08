"""
ClimaCredit Q – classical risk model.

Many small loans per sector (the ASRF model behind Basel's capital rules):
in a given year the economy factor z_e and the climate factor z_c are drawn,
and a fraction p_k(z_e, z_c) of the loans in sector k defaults.
Bank loss L(z) = sum_k  exposure_k * LGD * p_k(z).

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


# ---------------------------------------------------------------- parameters
def load_params(path=None, overrides=None):
    """Read parametrar.json -> dict of plain values. `overrides` replaces single values."""
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
    """Statistics Finland tables 13ff (bankruptcies) and 13ww (establishments)."""
    k = pd.read_excel(os.path.join(folder, "konkurser.xlsx"), header=None, skiprows=3)
    k.columns = ["year", "industry", "region", "bankruptcies"]
    k[["year", "industry"]] = k[["year", "industry"]].ffill()
    k = k[k["region"].notna() & pd.to_numeric(k["bankruptcies"], errors="coerce").notna()]
    k["year"] = k["year"].astype(int)
    k["bankruptcies"] = k["bankruptcies"].astype(int)
    k["sector"] = k["industry"].map({"Agriculture, forestry and fishing": "Agriculture",
                                     "Construction": "Construction", "Trade": "Trade"})
    k = k[k["sector"].notna()]

    f = pd.read_excel(os.path.join(folder, "foretag.xlsx"), header=None, skiprows=3)
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
    return k, f


def bankruptcy_rates(k, f):
    """Bankruptcies / firms per region and sector, 2018-2024."""
    x = k[k["year"].between(2018, 2024)].merge(f, on=["year", "region", "sector"])
    s = x.groupby(["region", "sector"])[["bankruptcies", "firms"]].sum()
    return (s["bankruptcies"] / s["firms"]).unstack()[SECTORS]


def region_shares(f, year=2024):
    """Share of firms per sector in each region -> the local bank's loan mix."""
    a = f[f["year"] == year].pivot(index="region", columns="sector", values="firms")[SECTORS]
    return a.div(a.sum(axis=1), axis=0)


# ---------------------------------------------------------------- model inputs
class Model:
    def __init__(self, params=None, data_folder=HERE):
        self.p = params or load_params()
        k, f = load_data(data_folder)
        self.bankruptcy = bankruptcy_rates(k, f)
        self.shares = region_shares(f).loc[self.bankruptcy.index]
        self.regions = [r for r in self.bankruptcy.index if r != "WHOLE COUNTRY"]
        self.pd_today = self._pd_today()
        self.exposure = self.shares * self.p["bank_size_meur"]
        # fine grid over (z_economy, z_climate) for exact classical results
        g = np.linspace(-6, 6, 241)
        w = norm.pdf(g); w /= w.sum()
        self.ze, self.zc = np.meshgrid(g, g, indexing="ij")
        self.wgrid = np.outer(w, w)

    # PD today per region and sector
    def _pd_today(self):
        p = self.p
        br = self.bankruptcy
        pd_ = br.copy()
        ratio = br["Agriculture"] / br.loc["WHOLE COUNTRY", "Agriculture"]
        pd_["Agriculture"] = p["pd_agriculture_national"] * (1 + p["regional_weight"] * (ratio - 1))
        for s in ["Construction", "Trade"]:
            pd_[s] = br[s] * p["pd_scale_construction_trade"]
        return pd_.clip(upper=0.5)

    def rho(self, region):
        """Factor loadings (economy, climate) per sector for a region."""
        p = self.p
        rm = np.array([p["rho_economy"][s] for s in SECTORS])
        rk = np.array([p["rho_climate"][s] for s in SECTORS])
        if region in p["flood_regions"]:
            rk = rk + np.array([p["flood_rho_add"][s] for s in SECTORS])
        return rm, rk

    def pd_scenario(self, region, scenario):
        return np.minimum(self.pd_today.loc[region].values * self.p["scenarios"][scenario], 0.99)

    # ------------------------------------------------------------ core formulas
    @staticmethod
    def cond_pd(pd_, rm, rk, ze, zc):
        """PD given the economy and climate factors (Vasicek with two factors)."""
        return norm.cdf((norm.ppf(pd_) - np.sqrt(rm) * ze - np.sqrt(rk) * zc) / np.sqrt(1 - rm - rk))

    def loss_grid(self, region, scenario, ze=None, zc=None):
        """Bank loss L(z) in MEUR on the factor grid."""
        ze = self.ze if ze is None else ze
        zc = self.zc if zc is None else zc
        pdv = self.pd_scenario(region, scenario)
        rm, rk = self.rho(region)
        exp_ = self.exposure.loc[region].values
        L = np.zeros_like(ze, dtype=float)
        for i in range(len(SECTORS)):
            L += exp_[i] * self.p["lgd"] * self.cond_pd(pdv[i], rm[i], rk[i], ze, zc)
        return L

    @staticmethod
    def _var(values, weights, level):
        o = np.argsort(values)
        c = np.cumsum(weights[o])
        return values[o][min(np.searchsorted(c, level), len(c) - 1)]

    @staticmethod
    def _es(values, weights, level):
        """Expected shortfall: average loss in the worst (1 - level) of years (smooth in the inputs)."""
        o = np.argsort(values)[::-1]
        v, w = values[o], weights[o]
        tail = 1 - level
        c = np.cumsum(w)
        k = np.searchsorted(c, tail)
        used = w[:k].sum()
        return float(((v[:k] * w[:k]).sum() + v[k] * (tail - used)) / tail)

    def bank_metrics(self, region, scenario):
        L = self.loss_grid(region, scenario).ravel()
        w = self.wgrid.ravel()
        el = float((L * w).sum())
        var = float(self._var(L, w, self.p["var_level"]))
        es = self._es(L, w, self.p["var_level"])
        return {"expected_loss_meur": el, "var_meur": var, "es_meur": es, "capital_meur": es - el}

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

    def _loan_risk(self, region, scenario, sector, amount_meur, firm_mult):
        """EL and extra capital (Delta ES - Delta EL) from adding one loan to the bank.
        The new loan is a single borrower: it defaults completely or not at all."""
        i = SECTORS.index(sector)
        lgd = self.p["lgd"]
        rm, rk = self.rho(region)
        pd_firm = min(self.pd_scenario(region, scenario)[i] * firm_mult, 0.99)
        Lp = self.loss_grid(region, scenario).ravel()
        w = self.wgrid.ravel()
        q = self.cond_pd(pd_firm, rm[i], rk[i], self.ze, self.zc).ravel()
        lvl = self.p["var_level"]
        # bank without the loan
        el0 = (Lp * w).sum()
        es0 = self._es(Lp, w, lvl)
        # bank with the loan: each grid point splits into "loan defaults" / "loan survives"
        vals = np.concatenate([Lp + amount_meur * lgd, Lp])
        wts = np.concatenate([w * q, w * (1 - q)])
        el1 = (vals * wts).sum()
        es1 = self._es(vals, wts, lvl)
        return {"pd_firm": pd_firm,
                "el_rate": (el1 - el0) / amount_meur,
                "capital_rate": ((es1 - el1) - (es0 - el0)) / amount_meur}

    def price_loan(self, region, sector, amount_eur, scenario=None, equity_ratio=None,
                   net_debt_ebitda=None, payment_default=False, euribor=None):
        """Interest rate = Euribor + bank margin + climate add-on."""
        p = self.p
        scenario = scenario or p["main_scenario"]
        amount = amount_eur / 1e6
        mult, grade = self.firm_multiplier(equity_ratio, net_debt_ebitda, payment_default)
        today = self._loan_risk(region, "Today", sector, amount, mult)
        scen = self._loan_risk(region, scenario, sector, amount, mult)
        cc = p["cost_of_capital"]
        own = scen["el_rate"] - today["el_rate"]                      # the loan's own climate risk
        conc = cc * (scen["capital_rate"] - today["capital_rate"])     # extra capital for the bank's portfolio
        add_on = max(own + conc, 0.0)
        euribor = p["euribor_fallback"] if euribor is None else euribor
        return {
            "region": region, "sector": sector, "amount_eur": amount_eur, "scenario": scenario,
            "firm_grade": grade, "firm_pd_multiplier": mult,
            "pd_today": today["pd_firm"], "pd_scenario": scen["pd_firm"],
            "euribor": euribor, "bank_margin": p["bank_margin"],
            "climate_addon_own_risk": own, "climate_addon_concentration": conc,
            "climate_addon": add_on,
            "total_rate": euribor + p["bank_margin"] + add_on,
        }


if __name__ == "__main__":
    m = Model()
    for reg in ["MK15 Ostrobothnia", "MK01 Uusimaa"]:
        for sc in m.p["scenarios"]:
            print(reg, sc, {k: round(v, 3) for k, v in m.bank_metrics(reg, sc).items()})
    for reg in ["MK15 Ostrobothnia", "MK01 Uusimaa"]:
        for amt in [500_000, 5_000_000, 20_000_000]:
            r = m.price_loan(reg, "Agriculture", amt, equity_ratio=0.3, net_debt_ebitda=4)
            print(reg, amt, "own %.3f%%  conc %.3f%%  add-on %.3f%%  total %.2f%%" % (
                r["climate_addon_own_risk"] * 100, r["climate_addon_concentration"] * 100,
                r["climate_addon"] * 100, r["total_rate"] * 100))
