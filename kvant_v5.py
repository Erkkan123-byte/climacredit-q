"""
ClimaCredit Q – quantum model (Qiskit).

Uses the same many-loans model as climacredit_model.py. Because each sector has many
small loans, the bank's loss is fully determined by the economy and climate factors.
So the circuit only needs:
  * n qubits for the economy factor and n qubits for the climate factor (normal distributions)
  * 1 objective qubit
QAE then estimates
  1. P(large loss)  – probability that the bank loses more than a threshold
  2. Expected loss  – via amplitude encoding of the loss
  3. VaR            – by bisection over the threshold (as in Qiskit's credit risk tutorial)

Run:   %run kvant_v5.py            (in Jupyter, same folder as parametrar.json and the data)
"""
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import StatePreparation

from climacredit_model import Model


def normal_distribution(n, bound):
    """n-qubit circuit that loads a standard normal distribution on 2^n points in [-bound, bound].
    Returns (circuit, values, probabilities). Same as qiskit_finance.NormalDistribution."""
    values = np.linspace(-bound, bound, 2 ** n)
    probs = np.exp(-values ** 2 / 2)
    probs = probs / probs.sum()
    qc = QuantumCircuit(n, name="N(0,1)")
    qc.append(StatePreparation(np.sqrt(probs)), range(n))
    return qc, values, probs

# ------------------------------------------------------------------ settings
N_QUBITS_PER_FACTOR = 4        # 4 -> 16 economy states x 16 climate states = 256 scenarios
Z_BOUND = 3.0                  # factor values from -3 to +3 standard deviations
LARGE_LOSS_SHARE = 0.10        # "large loss" = at least 10 % of the bank's loan book
BANKS = {"Rural bank (Ostrobothnia)": "MK15 Ostrobothnia", "City bank (Uusimaa)": "MK01 Uusimaa"}


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


def scenario_losses(model, region, scenario, n=N_QUBITS_PER_FACTOR, bound=Z_BOUND):
    """Loss (MEUR) and probability for each of the 2^n x 2^n factor states.
    Index convention: state index = i_economy + 2^n * i_climate (qubit 0 = least significant)."""
    z, pz = factor_grid(n, bound)
    N = 2 ** n
    idx = np.arange(N * N)
    i_e, i_c = idx % N, idx // N
    L = model.loss_grid(region, scenario, ze=z[i_e], zc=z[i_c])
    prob = pz[i_e] * pz[i_c]
    return L, prob


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


def qae_var(L, prob, level, n=N_QUBITS_PER_FACTOR):
    """VaR by bisection: smallest loss x with P(L > x) <= 1 - level, each P estimated with QAE."""
    xs = np.unique(L)
    lo_i, hi_i = 0, len(xs) - 1
    calls = 0
    while hi_i - lo_i > 1:
        mid = (lo_i + hi_i) // 2
        p_tail, *_ , q = qae(circuit_large_loss(L, xs[mid] + 1e-9, n), epsilon=0.002)
        calls += q
        if p_tail <= 1 - level:
            hi_i = mid
        else:
            lo_i = mid
    return xs[hi_i], calls


def fine_tail(model, region, scenario, threshold):
    L = model.loss_grid(region, scenario)
    return float(model.wgrid[L >= threshold].sum())


# ------------------------------------------------------------------ run
def run(scenarios=None, model=None):
    model = model or Model()
    scenarios = scenarios or ["Today", model.p["main_scenario"]]
    rows = []
    for (name, region), scenario in [(b, s) for b in BANKS.items() for s in scenarios]:
        L, prob = scenario_losses(model, region, scenario)
        threshold = LARGE_LOSS_SHARE * model.p["bank_size_meur"]

        exact_p = prob[L >= threshold].sum()
        p_q, p_lo, p_hi, p_calls = qae(circuit_large_loss(L, threshold), epsilon=max(0.1 * exact_p, 1e-4))

        exact_el = (L * prob).sum()
        circ_el, lmax = circuit_expected_loss(L)
        el_q, el_lo, el_hi, el_calls = qae(circ_el, epsilon=0.002)

        rows.append({
            "Bank": name, "Scenario": scenario, "Qubits": 2 * N_QUBITS_PER_FACTOR + 1,
            "Large loss threshold (MEUR)": threshold,
            "P(large loss) exact %": round(exact_p * 100, 3),
            "P(large loss) QAE %": round(p_q * 100, 3),
            "P(large loss), fine classical grid %": round(fine_tail(model, region, scenario, threshold) * 100, 3),
            "Expected loss exact (MEUR)": round(exact_el, 2),
            "Expected loss QAE (MEUR)": round(el_q * lmax, 2),
            "Expected loss, fine classical grid (MEUR)": round(model.bank_metrics(region, scenario)["expected_loss_meur"], 2),
            "Oracle calls": p_calls + el_calls,
        })
        print("Done:", name, "-", scenario)
    return pd.DataFrame(rows)


if __name__ == "__main__" or "get_ipython" in globals():
    quantum_results = run()
    print()
    print(quantum_results.T.to_string())
