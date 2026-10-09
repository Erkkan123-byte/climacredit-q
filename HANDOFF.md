# ClimaCredit Q – överlämning (v6)

## Vad det är

En lånekalkylator för företagslån i en lokal OP-andelsbank:

**Ränta = Euribor (1/3/6/12 mån, live från ECB) + bankens marginal (justerbar, standard 2 %) + klimatpåslag**

Klimatpåslaget = (1) lånets egen klimatrisk: högre förväntad förlust när branschens PD stiger i ett EU-klimatscenario och i ett dåligt skördeår, plus (2) kostnaden för det extra kapital lånet kräver i den lokala bankens låneportfölj.

## Arkitektur

```
Lovable-app (frontend)  →  https://climacredit-q-api.onrender.com  (Python-server)
                                   ├─ riskmodell (climacredit_model.py)
                                   ├─ kvantmodell, QAE i Qiskit (kvant_v6.py)
                                   └─ data: Statistikcentralen, Eurostat + parametrar.json
```

Appen räknar ingenting själv, den anropar bara servern. Fälten finns i `LOVABLE_PROMPT_v6.txt` och på `/docs`.

## Nyckelsiffror (jordbrukslån 500 000 €, nöjaktigt företag, Run-on-brown, 12 mån Euribor 3,25 %, marginal 2 %)

| | Södra Österbotten (landsbygd) | Nyland (stad) |
|---|---|---|
| Jordbrukets andel av bankens lån | 59 % | 3 % |
| Klimatpåslag, normalt år | 2,17 % | 2,02 % |
| Klimatpåslag, bra år (som 2019) | 1,23 % | 0,87 % |
| Klimatpåslag, dåligt år (som 2018) | 3,00 % | 3,06 % |
| Klimatpåslag, mycket dåligt år (som 2021) | 3,45 % | 4,62 % |
| Bankens kapitalbehov idag → scenario | 53 → 65 M€ (+22 %) | 43 → 50 M€ (+16 %) |

Torka slår hårdast i södra Finland (Nyland −31 % skörd 2021), översvämning i Österbotten och Lappland.

Kvant: QAE ger 3,50 % sannolikhet för stor förlust (≥ 10 % av lånestocken) mot exakt 3,49 % för landsbygdsbanken i Run-on-brown; förväntad förlust 7,91 mot 7,91 M€.

## Viktigt

- Byt **inte** server-adress och räkna **inte** risk i frontenden – då stämmer siffrorna inte med kvantmodellen.
- Alla antaganden ändras i `parametrar.json` (GitHub: Erkkan123-byte/climacredit-q). Render bygger om vid ny commit (annars **Manual Deploy**).
- Bra/dåligt år är ett stresstest (vad händer om nästa år blir som 2018), inte priset banken skulle ta i ett normalt läge.
