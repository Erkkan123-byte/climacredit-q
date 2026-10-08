# ClimaCredit Q – v5

Climate-adjusted loan pricing for local cooperative banks, with a quantum risk engine (Qiskit).

**Interest rate = 12-month Euribor + bank margin (2 %) + climate add-on**

The climate add-on has two parts:
1. **The loan's own climate risk** – higher expected loss because the sector's PD rises in an EU climate scenario.
2. **Concentration** – the cost of the extra capital the loan needs in the local bank's loan book (expected shortfall 99.9 %).

## Files

| File | What it is |
|---|---|
| `parametrar.json` | **All assumptions in one place**, each with a source. Change a number here and everything follows. |
| `climacredit_model.py` | Classical risk model: many small loans per sector (Basel ASRF), economy + climate factor, regional PD, flood regions, pricing. |
| `kvant_v5.py` | Quantum model: QAE on 9 qubits estimates P(large loss) and expected loss. Run with `%run kvant_v5.py`. |
| `kvant_hardware.py` | Runs the uncertainty model (5 qubits, ~20 two-qubit gates) on IQM or IBM hardware, or on IQM's noisy simulator. |
| `server.py` | API for the Lovable app. |
| `ClimaCredit_Q_v5.ipynb` | Notebook that walks through everything. |
| `konkurser.xlsx`, `foretag.xlsx` | Statistics Finland tables 13ff and 13ww. |

## Kör lokalt (på er dator)

```
pip install -r requirements.txt
uvicorn server:app --reload
```
Öppna http://127.0.0.1:8000/docs – där kan ni testa alla anrop direkt i webbläsaren.

## Lägg upp servern gratis (Render)

1. Skapa ett konto på github.com och ett nytt repository, t.ex. `climacredit-q`.
2. Ladda upp **alla filer i den här mappen** till repot (Add file → Upload files).
3. Gå till render.com, logga in med GitHub, välj **New → Blueprint** och välj repot. Render läser `render.yaml` själv.
4. Vänta tills det står *Live* (första gången 5–10 min). Ni får en adress som `https://climacredit-q-api.onrender.com`.
5. Testa: öppna `https://<er-adress>/docs`.
6. Ge adressen till Lovable (se `lovable_prompt.md`).

Obs: gratisservern somnar efter 15 minuter utan trafik och tar ~30 s att vakna. Öppna `/health` en minut före demon.

## Kvanthårdvara

IQM och IBM kräver olika Qiskit-versioner – använd två separata miljöer (t.ex. två conda-miljöer).

```
# IQM
pip install "iqm-client[qiskit]" qiskit-aer pandas scipy openpyxl
python -c "from kvant_hardware import run_on; print(run_on('iqm', url='<IQM server URL>', token='<token>'))"

# IBM
pip install qiskit-ibm-runtime qiskit-aer pandas scipy openpyxl
python -c "from kvant_hardware import run_on; print(run_on('ibm', token='<API key>', instance='<instance CRN>'))"
```
Utan inloggning: `%run kvant_hardware.py` kör en brusfri simulator och IQM:s brusiga modell av deras 20-qubit-maskin Apollo.
