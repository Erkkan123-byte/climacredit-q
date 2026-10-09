"""
ClimaCredit Q v6 – run the climate uncertainty model on REAL quantum hardware (IQM or IBM).

Full QAE needs thousands of two-qubit gates, which today's machines cannot run without
the noise taking over. So on hardware we run the small part that fits:
  * 2 qubits economy factor + 2 qubits climate factor (16 economy x climate scenarios)
  * 1 objective qubit, rotated so that P(objective = 1) = expected loss / maximum loss
We measure many times, correct for readout errors, and read off the bank's expected loss.
This is the "uncertainty model" of our quantum algorithm running on a real quantum computer.
The answer is compared with the exact value and with a noise-free simulator.

Usage (in Jupyter, same folder as the other files):
    %run kvant_hardware.py                      # noise-free simulator + IQM noisy simulator (no login needed)

    from kvant_hardware import run_on
    run_on("iqm", url="https://<your IQM Resonance server URL>", token="<your token>")
    run_on("ibm", token="<your IBM Quantum API key>", instance="<your instance CRN>")

Install (IQM and IBM need different Qiskit versions -> use separate environments):
    IQM:  pip install "iqm-client[qiskit]" qiskit-aer
    IBM:  pip install qiskit-ibm-runtime qiskit-aer
"""
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
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

N = 2                      # qubits per factor -> 4 x 4 = 16 scenarios (small enough for hardware)
Z_BOUND = 2.0
SHOTS = 4000
BANKS = {"Rural bank (South Ostrobothnia)": "MK14 South Ostrobothnia", "City bank (Uusimaa)": "MK01 Uusimaa"}


def build_circuit(model, region, scenario):
    dist, z, pz = normal_distribution(N, Z_BOUND)
    idx = np.arange(4 ** N)
    i_e, i_c = idx % 2 ** N, idx // 2 ** N
    L = model.loss_grid(region, scenario, ze=z[i_e], zc=z[i_c])
    prob = pz[i_e] * pz[i_c]
    lmax = float(L.max())

    qc = QuantumCircuit(2 * N + 1, 1)
    qc.append(dist, range(0, N))
    qc.append(dist, range(N, 2 * N))
    qc.append(UCRYGate(list(2 * np.arcsin(np.sqrt(L / lmax)))), [2 * N] + list(range(2 * N)))
    qc.measure(2 * N, 0)
    return qc, float((L * prob).sum()), lmax


def calibration_circuits():
    """Prepare |0> and |1> on the objective qubit to measure the readout error."""
    out = []
    for bit in (0, 1):
        c = QuantumCircuit(2 * N + 1, 1)
        if bit:
            c.x(2 * N)
        c.measure(2 * N, 0)
        out.append(c)
    return out


def _p_one(counts):
    shots = sum(counts.values())
    return sum(v for k, v in counts.items() if str(k).replace(" ", "")[-1] == "1") / shots


def _get_backend(kind, **login):
    if kind == "ideal":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    if kind == "iqm-fake":
        from iqm.qiskit_iqm import IQMFakeApollo          # noisy model of IQM's 20-qubit Apollo
        return IQMFakeApollo()
    if kind == "iqm":
        from iqm.qiskit_iqm import IQMProvider
        return IQMProvider(login["url"], token=login.get("token")).get_backend()
    if kind == "ibm-fake":
        from qiskit_ibm_runtime.fake_provider import FakeTorino   # noisy model of IBM's Heron r1
        return FakeTorino()
    if kind == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=login.get("token"),
                                       instance=login.get("instance"))
        return service.least_busy(operational=True, simulator=False)
    raise ValueError(kind)


def _run(backend, kind, circ, layout=None):
    tc = transpile(circ, backend, optimization_level=3, seed_transpiler=1, initial_layout=layout)
    two_q = sum(v for k, v in tc.count_ops().items() if k in ("cx", "cz", "ecr", "move"))
    if kind == "ibm":
        from qiskit_ibm_runtime import SamplerV2
        res = SamplerV2(mode=backend).run([tc], shots=SHOTS).result()[0]
        counts = res.data[list(res.data.keys())[0]].get_counts()
    else:
        counts = backend.run(tc, shots=SHOTS).result().get_counts()
    return _p_one(counts), two_q, tc


def _objective_physical_qubit(tc):
    return tc.layout.final_index_layout()[2 * N] if tc.layout is not None else 2 * N


def run_on(kind="ideal", scenarios=None, **login):
    model = Model()
    scenarios = scenarios or ["Today", model.p["main_scenario"]]
    backend = _get_backend(kind, **login)
    rows = []
    for name, region in BANKS.items():
        for sc in scenarios:
            circ, exact_el, lmax = build_circuit(model, region, sc)
            p, two_q, tc = _run(backend, kind, circ)
            # readout calibration on the same physical qubits
            layout = list(tc.layout.initial_index_layout())[: 2 * N + 1] if tc.layout is not None else None
            p0, _, _ = _run(backend, kind, calibration_circuits()[0], layout)   # P(read 1 | prepared 0)
            p1, _, _ = _run(backend, kind, calibration_circuits()[1], layout)   # P(read 1 | prepared 1)
            p_corr = float(np.clip((p - p0) / max(p1 - p0, 1e-6), 0, 1))
            rows.append({"Backend": kind, "Bank": name, "Scenario": sc,
                         "Expected loss exact (MEUR)": round(exact_el, 2),
                         "Measured, raw (MEUR)": round(p * lmax, 2),
                         "Measured, readout-corrected (MEUR)": round(p_corr * lmax, 2),
                         "Two-qubit gates": two_q})
    return pd.DataFrame(rows)


if __name__ == "__main__" or "get_ipython" in globals():
    out = [run_on("ideal")]
    try:
        out.append(run_on("iqm-fake"))
    except ImportError:
        print("IQM package not installed - skipping the IQM noisy simulator.")
    hardware_results = pd.concat(out, ignore_index=True)
    print(hardware_results.to_string(index=False))
