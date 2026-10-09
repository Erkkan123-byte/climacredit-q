# ClimaCredit Q – v6

Climate-adjusted loan pricing for local cooperative banks, with a quantum risk engine (Qiskit).

**Interest rate = reference rate (Euribor) + bank margin + climate add-on**

The climate add-on has two parts:
1. **Own climate risk** – higher expected loss when the sector's default probability (PD) rises in an EU climate scenario (transition) and in a bad harvest year (physical risk).
2. **Capital** – the cost of the extra capital the loan needs in the local bank's loan book (expected shortfall 99.9 %).

## Nytt i v6

| | v5 | v6 |
|---|---|---|
| Bankens lånemix | antal företag (jordbruket alldeles för stort) | landskapets **företagsskulder** (gårdarnas skulder, SMF:s räntekostnader) |
| PD bygg/handel | konkursfrekvens rakt av | konkurser **+ saneringar**, nivå kalibrerad till EU-snittet 2,4 % |
| Regionala skillnader | hälften av skillnaden för jordbruk | credibility-vikt n/(n+100) för alla branscher |
| Fysisk klimatrisk | 8 översvämningsområden | alla **18** (14 älv, 4 kust) + **bra/dåliga skördeår** (2019, 2018, 2021) + torkkänslighet per landsdel |
| Omställningsscenariot | samma multiplikator för alla branscher | branschvis (mönstret från energikrisen 2022), till hälften |
| Scenario-stress | PD × multiplikator | samma för genomsnittsföretaget, men som förskjutning på probit-skalan (Vasicek) så att staplade stressar inte exploderar |
| Underbransch | – | TOL 3 siffror, krympt mot branschen (empirisk Bayes) |
| Lånekalkylator | – | **justerbar marginal**, Euribor 1/3/6/12 mån, löptid, amorteringssätt, betalningsplan |

## Files

| File | What it is |
|---|---|
| `parametrar.json` | **All assumptions in one place**, each with a source. Change a number here and everything follows. |
| `climacredit_model.py` | Classical risk model (Basel ASRF, economy + climate factor), calibration, weather years, pricing, payment plan. |
| `kvant_v6.py` | Quantum model: QAE on 9 qubits estimates P(large loss) and expected loss. Run with `%run kvant_v6.py`. |
| `kvant_hardware.py` | Runs the uncertainty model (5 qubits, ~20 two-qubit gates) on IQM or IBM hardware, or on IQM's noisy simulator. |
| `server.py` | API for the Lovable app. |
| `ClimaCredit_Q_v6.ipynb` | Notebook that walks through everything, with charts. |
| `LOVABLE_PROMPT_v6.txt` | Prompt for the calculator-style app. |
| `konkurser.xlsx`, `foretag.xlsx` | Statistics Finland 13ff and 13ww (bankruptcies and establishments per region). |
| `konkurser_bransch_2003_2025.csv`, `saneringar_bransch_2003_2025.csv`, `foretag_13w3.csv` | Bankruptcies, restructurings and firms per sector (13fg, 13fp, 13w3). |
| `regioner_2024.csv`, `exponering_underlag_2024.csv` | Farms, farm debt, gross output, SME interest costs (141b, 13z5, 13x2, 13z9, 13w2). |
| `underbranscher_2018_2025.csv` | Bankruptcies and firms per TOL 3-digit sub-industry. |
| `skord_eurostat.csv` | Cereal area and production per NUTS 2 area, 2000–2024 (Eurostat apro_cpnhr). |
| `konkurser_1986_2025.csv`, `utslapp_11ig.csv`, `omsattning_2022.csv` | Used in the notebook's robustness checks. |

## Uppdatera servern (GitHub + Render)

1. Gå till repot **Erkkan123-byte/climacredit-q** på github.com → **Add file → Upload files**.
2. Dra in **alla filer** från den här mappen (de som redan finns skrivs över). Skriv t.ex. "v6" som commit-meddelande → **Commit changes**.
3. Render bygger om automatiskt. Om inget händer: render.com → tjänsten **climacredit-q-api** → **Manual Deploy → Deploy latest commit**.
4. Vänta tills det står *Live* (3–8 min). Testa `https://climacredit-q-api.onrender.com/health` → ska visa `"version": "6.0"`.
5. Öppna `https://climacredit-q-api.onrender.com/docs` och prova `POST /price`.
6. Klistra in `LOVABLE_PROMPT_v6.txt` i Lovable.

Den gamla appen fortsätter fungera mot v6-servern (samma fält finns kvar), så inget går sönder medan ni bygger om.

Obs: gratisservern somnar efter 15 minuter utan trafik och tar ~30–50 s att vakna. Öppna `/health` en minut före demon.

## Kör lokalt

```
pip install -r requirements.txt
uvicorn server:app --reload
```
Öppna http://127.0.0.1:8000/docs.

## API i korthet

| Anrop | Vad det ger |
|---|---|
| `GET /regions` | 19 landskap: lånemix, PD, översvämningsområden, torkkänslighet |
| `GET /banks` | Alla 19 lokala banker: kapitalbehov idag mot scenario |
| `GET /euribor` | Euribor 1/3/6/12 mån live från ECB |
| `GET /scenarios` | EU-scenarierna med PD-multiplikator per bransch |
| `GET /weather_years` | Bra/normala/dåliga skördeår och skördeavvikelse per landsdel och år |
| `GET /sub_industries?sector=...` | Underbranscher med relativ risk |
| `POST /price` | Ränta, uppdelning, månadsbetalning, betalningsplan, bra/dåligt år, scenario-matris, jämförelse mellan landskap |
| `POST /payment_plan` | Betalningsplan för valfri summa, ränta och löptid |
| `POST /quantum` | QAE på simulator (9 qubits), 10–60 s |

## Kvanthårdvara

IQM och IBM kräver olika Qiskit-versioner – använd två separata miljöer.

```
# IQM
pip install "iqm-client[qiskit]" qiskit-aer pandas scipy openpyxl
python -c "from kvant_hardware import run_on; print(run_on('iqm', url='<IQM server URL>', token='<token>'))"

# IBM
pip install qiskit-ibm-runtime qiskit-aer pandas scipy openpyxl
python -c "from kvant_hardware import run_on; print(run_on('ibm', token='<API key>', instance='<instance CRN>'))"
```
Utan inloggning: `%run kvant_hardware.py` kör en brusfri simulator och IQM:s brusiga modell av deras 20-qubit-maskin Apollo.

## Källor

- ECB/ESAs: Fit-for-55 climate scenario analysis (Nov 2024) – PD 2,4 %, jordbruk 3,5 %, scenarierna.
- Statistikcentralen: 13ff, 13fg, 13fp, 13ww, 13w3, 141b, 13z5, 13x2, 13z9, 13w2, 11ig.
- Eurostat apro_cpnhr: spannmålsskörd per NUTS 2-område.
- Jord- och skogsbruksministeriet 19.12.2024: 18 betydande översvämningsområden 2025–2030.
- Luke: torkan 2018 (gårdarnas företagarinkomst −40 %), skörden 2021 (minsta sedan 1992). Meteorologiska institutet: varm och torr sommar 2021.
- ECB/EMMI: Euribor.
