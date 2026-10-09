"""
ClimaCredit Q v6 - quantum model (Qiskit).

Uses the same many-loans model as climacredit_model.py. Because each sector has many small
loans, the bank's loss is fully determined by the economy and climate factors. So the circuit
only needs:
  * n qubits for the economy factor and n qubits for the climate factor (normal distributions)
  * 1 objective qubit
QAE (Iterative Amplitude Estimation) then estimates
  1. P(large loss)  - probability that the bank loses at least 10 % of its loan book in a year
  2. Expected loss  - via amplitude encoding of the loss
  3. VaR            - by bisection over the threshold (as in Qiskit's credit risk tutorial)

New in v6: the weather year (good / bad harvest) can be chosen, and the rural bank is
South Ostrobothnia (the region with the largest share of farm debt, 59 %).

Run:   %run kvant_v6.py            (in Jupyter, same folder as parametrar.json and the data)
"""
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate, StatePreparation
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

from climacredit_model import Model

# ------------------------------------------------------------------ settings
N_QUBITS_PER_FACTOR = 4        # 4 -> 16 economy states x 16 climate states = 256 scenarios
Z_BOUND = 3.0                  # factor values from -3 to +3 standard deviations
LARGE_LOSS_SHARE = 0.10        # "large loss" = at least 10 % of the bank's loan book
BANKS = {"Rural bank (South Ostrobothnia)": "MK14 South Ostrobothnia", "City bank (Uusimaa)": "MK01 Uusimaa"}


def normal_distribution(n, bound):
    """n-qubit circuit that loads a standard normal distribution on 2^n points in [-bound, bound].
    Returns (circuit, values, probabilities). Same as qiskit_finance.NormalDistribution."""
    values = np.linspace(-bound, bound, 2 ** n)
    probs = np.exp(-values ** 2 / 2)
    probs = probs / probs.sum()
    qc = QuantumCircuit(n, name="N(0,1)")
    qc.append(StatePreparation(np.sqrt(probs)), range(n))
    return qc, values, probs


class TranspilingSampler(BaseSamplerV2):
    """Aer sampler that first translates circuits to basic gates (Aer does not know the Grover operator)."""
    def __init__(self, shots=2000, seed=1):
        self._base = AerSampler(default_shots=shots, seed=seed)
        self._backend = AerSimulator()

    def run(self, pubs, *, shots=None):
        new = []
        for pub in pubs:
            circ, rest = (pub[0], tuple(pub[1:])) if isinstance(pub, tuple) else (pub, ())
            new.append((transpile(circ, self._backend, optimization_level=1),) + rest)
        return self._base.run(new, shots=shots)


# ------------------------------------------------------------------ circuit pieces
def factor_grid(n=N_QUBITS_PER_FACTOR, bound=Z_BOUND):
    _, values, probs = normal_distribution(n, bound)
    return values, probs


def scenario_losses(model, region, scenario, weather=None, n=N_QUBITS_PER_FACTOR, bound=Z_BOUND):
    """Loss (MEUR) and probability for each of the 2^n x 2^n factor states.
    Index convention: state index = i_economy + 2^n * i_climate (qubit 0 = least significant)."""
    z, pz = factor_grid(n, bound)
    N = 2 ** n
    idx = np.arange(N * N)
    i_e, i_c = idx % N, idx // N
    L = model.loss_grid(region, scenario, weather, ze=z[i_e], zc=z[i_c])
    return L, pz[i_e] * pz[i_c]


def state_preparation(n, angles):
    """Economy + climate factors on 2n qubits, then rotate the objective qubit by angle(state)."""
    qc = QuantumCircuit(2 * n + 1)
    dist, _, _ = normal_distribution(n, Z_BOUND)
    qc.append(dist, range(0, n))
    qc.append(dist, range(n, 2 * n))
    qc.append(UCRYGate(list(angles)), [2 * n] + list(range(2 * n)))
    return qc


def circuit_large_loss(L, threshold, n=N_QUBITS_PER_FACTOR):
    """Objective qubit = 1 exactly in the factor states where the bank loses >= threshold."""
    return state_preparation(n, np.where(L >= threshold, np.pi, 0.0))


def circuit_expected_loss(L, n=N_QUBITS_PER_FACTOR):
    """Amplitude encoding: P(objective = 1 | state) = L(state) / L_max  ->  P(1) = E[L] / L_max."""
    lmax = L.max()
    return state_preparation(n, 2 * np.arcsin(np.sqrt(L / lmax))), lmax


def qae(circuit, epsilon, alpha=0.05, shots=2000):
    basis = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1)   # Aer needs basic gates
    problem = EstimationProblem(state_preparation=basis, objective_qubits=[circuit.num_qubits - 1])
    iae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=TranspilingSampler(shots))
    r = iae.estimate(problem)
    lo, hi = r.confidence_interval_processed
    return r.estimation_processed, lo, hi, r.num_oracle_queries


def mc_samples_needed(p, half_width, z=1.96):
    """Classical Monte Carlo samples for the same 95 % interval half-width (binomial)."""
    hw = max(half_width, 1e-12)
    return int(np.ceil(z ** 2 * p * (1 - p) / hw ** 2))


def qae_var(L, prob, level, n=N_QUBITS_PER_FACTOR):
    """VaR by bisection: smallest loss x with P(L > x) <= 1 - level, each P estimated with QAE."""
    xs = np.unique(L)
    lo_i, hi_i = 0, len(xs) - 1
    calls = 0
    while hi_i - lo_i > 1:
        mid = (lo_i + hi_i) // 2
        p_tail, *_, q = qae(circuit_large_loss(L, xs[mid] + 1e-9, n), epsilon=0.002)
        calls += q
        if p_tail <= 1 - level:
            hi_i = mid
        else:
            lo_i = mid
    return xs[hi_i], calls


def fine_tail(model, region, scenario, threshold, weather=None):
    L = model.loss_grid(region, scenario, weather)
    return float(model.wgrid[L >= threshold].sum())


# ------------------------------------------------------------------ run
def run_one(model, region, scenario, weather=None):
    """QAE for one bank: P(large loss) and expected loss, compared with the exact values. Used by the API."""
    L, prob = scenario_losses(model, region, scenario, weather)
    thr = LARGE_LOSS_SHARE * model.p["bank_size_meur"]
    exact_p = float(prob[L >= thr].sum())
    p_q, p_lo, p_hi, calls1 = qae(circuit_large_loss(L, thr), epsilon=max(0.1 * exact_p, 1e-4))
    circ_el, lmax = circuit_expected_loss(L)
    el_q, el_lo, el_hi, calls2 = qae(circ_el, epsilon=0.002)
    return {
        "qubits": 2 * N_QUBITS_PER_FACTOR + 1,
        "factor_states": int(4 ** N_QUBITS_PER_FACTOR),
        "large_loss_threshold_meur": thr,
        "p_large_loss_exact": exact_p, "p_large_loss_qae": float(p_q), "p_large_loss_ci": [float(p_lo), float(p_hi)],
        "expected_loss_exact_meur": float((L * prob).sum()),
        "expected_loss_qae_meur": float(el_q * lmax), "expected_loss_ci_meur": [float(el_lo * lmax), float(el_hi * lmax)],
        "oracle_calls": int(calls1 + calls2),
        "monte_carlo_samples_same_precision": mc_samples_needed(exact_p, (p_hi - p_lo) / 2)
                                              + mc_samples_needed(el_q, (el_hi - el_lo) / 2),
        "backend": "Qiskit Aer simulator (IterativeAmplitudeEstimation)",
    }


def run(scenarios=None, model=None, weather=None):
    model = model or Model()
    scenarios = scenarios or ["Today", model.p["main_scenario"]]
    rows = []
    for (name, region), scenario in [(b, s) for b in BANKS.items() for s in scenarios]:
        r = run_one(model, region, scenario, weather)
        threshold = r["large_loss_threshold_meur"]
        rows.append({
            "Bank": name, "Scenario": scenario, "Weather": weather or "Normal year", "Qubits": r["qubits"],
            "Large loss threshold (MEUR)": threshold,
            "P(large loss) exact %": round(r["p_large_loss_exact"] * 100, 3),
            "P(large loss) QAE %": round(r["p_large_loss_qae"] * 100, 3),
            "P(large loss), fine classical grid %": round(fine_tail(model, region, scenario, threshold, weather) * 100, 3),
            "Expected loss exact (MEUR)": round(r["expected_loss_exact_meur"], 2),
            "Expected loss QAE (MEUR)": round(r["expected_loss_qae_meur"], 2),
            "Expected loss, fine classical grid (MEUR)": round(model.bank_metrics(region, scenario, weather)["expected_loss_meur"], 2),
            "Oracle calls": r["oracle_calls"],
            "Monte Carlo samples for same precision": r["monte_carlo_samples_same_precision"],
        })
        print("Done:", name, "-", scenario, "-", weather or "Normal year")
    return pd.DataFrame(rows)


if __name__ == "__main__" or "get_ipython" in globals():
    quantum_results = run()
    print()
    print(quantum_results.T.to_string())
