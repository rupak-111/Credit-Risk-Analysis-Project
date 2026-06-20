# HK Credit Risk Analytics — End-to-End Portfolio

A complete credit risk analyst case study for a Hong Kong–based multinational
bank ("Pacific Crest Bank"). It takes a retail/SME lending book from raw data
all the way to a deployable credit scorecard, risk grades, expected loss, and a
stakeholder report — the full workflow a bank credit-risk team runs.

> **Synthetic data.** The dataset is generated locally and contains no real
> customer information. It is engineered to reproduce economically sensible
> relationships so the modelling behaves like a real portfolio.

## What's inside

| Path | Description |
|---|---|
| `data/generate_data.py` | Synthetic HKD retail credit portfolio generator (HKMA-style features: DSR, LTV, TU score) |
| `data/credit_portfolio.csv` | Generated dataset (12,000 facilities) |
| `notebooks/HK_Credit_Risk_Analysis.ipynb` | **Main deliverable** — executed end-to-end analysis with inline charts |
| `build_notebook.py` | Script that assembles the notebook from source cells |
| `outputs/` | All generated charts (PNG), scorecard, IV table, model metrics, risk grades |
| `RISK_REPORT.md` | Stakeholder risk report with findings and recommendations |

## Analysis workflow

1. **Business context** — HK MNC bank, HKMA regulatory framing (DSR, LTV).
2. **Data loading & dictionary**.
3. **Data quality & cleaning** — duplicates, implausible values, imputation.
4. **Exploratory data analysis** — default rates by segment, risk drivers,
   DSR bands, correlation.
5. **Feature engineering** — affordability headroom, loan-to-income, leverage.
6. **WOE / Information Value** — univariate predictive-power screening.
7. **Machine-learning models** — Logistic Regression, Random Forest, Gradient
   Boosting benchmarked on AUC / Gini / KS.
8. **Credit scorecard** — WOE → points (base 600, PDO 20), calibrated PDs.
9. **Risk tiers & expected loss** — five grades (A–E), EL = PD × LGD × EAD.
10. **Findings & recommendations** — lending policy, pricing, governance.

## Headline results

- Portfolio 12-month default rate: **13.6%**
- Scorecard discrimination: **AUC 0.826 · Gini 0.651 · KS 0.510**
- Risk grades rank-order from **1.6% (A)** to **46.6% (E)** realised default
- Auto-declining Grade E: **85.4% approval** captures **50% of defaults**

## How to run

Requires Python 3.10+ with: `pandas numpy scikit-learn statsmodels seaborn
matplotlib nbformat nbconvert`.

```bash
# 1. (re)generate the dataset
cd data && python3 generate_data.py && cd ..

# 2. build the notebook from source cells
python3 build_notebook.py

# 3. execute it end-to-end (writes charts + tables to outputs/)
cd notebooks
python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 HK_Credit_Risk_Analysis.ipynb
```

Or simply open `notebooks/HK_Credit_Risk_Analysis.ipynb` in Jupyter and run all
cells.

## Notes

- The dataset is regenerated deterministically (fixed seed) so results are
  reproducible.
- All preprocessing (imputation, scaling, WOE binning) is fit on the training
  split only — no leakage into the hold-out.
- Methodology is aligned to common HK practice and HKMA model-risk expectations
  (transparent scorecard, calibration, recommended PSI monitoring and
  fair-lending testing before any real deployment).
