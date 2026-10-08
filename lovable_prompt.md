# Prompt för Lovable

Klistra in allt under strecket i Lovable. Byt `https://YOUR-API-URL` mot adressen från Render.

---

Build a web app called **ClimaCredit Q** – climate-adjusted loan pricing for local cooperative banks, powered by a quantum risk engine. Audience: a bank's credit advisor and a hackathon jury (OP Financial Group). Language: English. Style: clean, professional fintech; dark navy + green accent; lots of white space; numbers large and readable.

All calculations happen in an existing API. Do NOT compute risk in the frontend – always call the API.

**API base URL:** `https://YOUR-API-URL` (store it in one constant so it is easy to change)

## Endpoints

1. `GET /regions` → array of `{ id, name, loan_mix: {Agriculture, Construction, Trade}, pd_today: {...}, flood_risk_area: bool }`. Use `name` in the region dropdown (19 Finnish regions).
2. `GET /parameters` → object of assumptions, each `{ value, source }`.
3. `GET /euribor` → `{ euribor_12m, period, source }`.
4. `POST /price` with JSON:
```json
{
  "company_name": "Maatila Oy",
  "sector": "Agriculture",            // "Agriculture" | "Construction" | "Trade"
  "region": "Ostrobothnia",            // region name from /regions
  "amount_eur": 500000,
  "equity_ratio": 0.30,                // optional, 0.30 = 30 %
  "net_debt_ebitda": 4.0,              // optional
  "payment_default": false,
  "scenario": "Run-on-brown",          // "Fit-for-55 baseline" | "Run-on-brown" | "Adverse"
  "overrides": { "pd_agriculture_national": 0.05 },   // optional, from the Assumptions panel
  "compare_regions": true
}
```
Response fields used: `euribor`, `euribor_source`, `bank_margin`, `climate_addon`, `climate_addon_own_risk`, `climate_addon_concentration`, `total_rate`, `firm_grade`, `pd_today`, `pd_scenario`, `climate_addon_by_scenario` (object scenario → add-on), `bank` (`loan_book_meur`, `loan_mix`, `flood_risk_area`, `today` and `scenario` each with `expected_loss_meur`, `es_meur`, `capital_meur`), `explanation`, `same_loan_other_regions` (array `{region, climate_addon}` sorted high→low). All rates are decimals (0.0567 = 5.67 %).
5. `POST /quantum` with `{ "region": "Ostrobothnia", "scenario": "Run-on-brown", "overrides": {...} }` → `{ qubits, large_loss_threshold_meur, p_large_loss_exact, p_large_loss_qae, p_large_loss_ci, expected_loss_exact_meur, expected_loss_qae_meur, expected_loss_ci_meur, oracle_calls, runtime_seconds, backend }`. Takes 5–15 s.

The free server sleeps; if the first call fails or takes long, show "Waking up the risk engine…" and retry automatically up to 3 times.

## Page layout

**Header:** "ClimaCredit Q" + subtitle "Climate-adjusted loan pricing for local banks · quantum risk engine".

**Left column – Loan application form**
- Company name (text)
- Sector (select: Agriculture, Construction, Trade)
- Region (select from /regions; show a small "flood risk area" badge next to regions where `flood_risk_area` is true)
- Loan amount (EUR, number with thousand separators, default 500 000)
- Equity ratio % (number, optional)
- Net debt / EBITDA (number, optional)
- Payment defaults (toggle)
- Climate scenario (segmented control: Fit-for-55 baseline / Run-on-brown / Adverse; default Run-on-brown)
- Button "Calculate rate"

**Right column – Result**
1. Big number: **Total interest rate X.XX %**
2. A horizontal stacked bar showing the build-up: 12-month Euribor (grey) + Bank margin 2.00 % (blue) + Climate add-on (green, split into "own climate risk" and "concentration in the bank's loan book" in two shades). Label each segment with its value.
3. Small cards: "Firm grade: satisfactory", "PD today → in scenario: 4.7 % → 7.3 %", "Euribor source".
4. The `explanation` text.
5. "Climate add-on by scenario": small bar chart from `climate_addon_by_scenario`.
6. "Same loan in other regions": horizontal bar chart of `same_loan_other_regions`, highlight the selected region.
7. "The local bank": loan mix as a donut (Agriculture/Construction/Trade), flood risk badge, and a small table: expected loss, expected shortfall 99.9 %, capital – today vs scenario (MEUR).

**Quantum panel (below the result)**
Title "Quantum risk engine". Short text: "The bank's tail risk is estimated with Quantum Amplitude Estimation (Qiskit) on a 9-qubit circuit: economy factor (4 qubits) + climate factor (4 qubits) + 1 objective qubit." Button "Run quantum calculation" → POST /quantum for the selected region and scenario. Show a spinner with "Running QAE on simulator…". Then show a two-column comparison: Exact vs QAE for "P(loss ≥ threshold)" and "Expected loss", with QAE confidence intervals, plus qubits, oracle calls and runtime. Add a note: "Today's quantum hardware is too noisy for full QAE; we ran the uncertainty model on IQM/IBM hardware separately. The quadratic speed-up applies to future fault-tolerant quantum computers."

**Assumptions panel (collapsible drawer, button "Assumptions" in the header)**
Load /parameters. Show sliders with the current value and the source text in small grey under each:
- Agriculture PD, national average (`pd_agriculture_national`): 1 %–10 %, step 0.1 %
- Regional weight (`regional_weight`): 0–1, step 0.05
- Construction & trade PD scale (`pd_scale_construction_trade`): 0.5–5, step 0.1
- LGD (`lgd`): 20 %–80 %
- Cost of capital (`cost_of_capital`): 5 %–20 %
- Bank margin (`bank_margin`): 0.5 %–4 %
- Bank loan book MEUR (`bank_size_meur`): 50–2000
- Scenario multipliers (`scenarios`): three sliders 1.0–3.0 for Fit-for-55 baseline, Run-on-brown, Adverse (send as `{"scenarios": {"Run-on-brown": 1.8}}`)
Only send the values the user changed, in `overrides`. Show a badge "Custom assumptions" when any override is active, and a "Reset to sources" button. Recalculate automatically 500 ms after a slider stops moving.

**Footer – "About the model"**: Data: Statistics Finland (bankruptcies 13ff, establishments 13ww). Climate scenarios: ECB/ESAs Fit-for-55 climate scenario analysis (Nov 2024). Risk model: Basel ASRF with an economy and a climate factor. Flood risk areas: Ministry of Agriculture and Forestry (2025–2030). Built for the OP Quantum x Hanken hackathon 2026.

Make it responsive (works on a laptop projector and a phone). Format percentages with 2 decimals and EUR with thousand separators.
