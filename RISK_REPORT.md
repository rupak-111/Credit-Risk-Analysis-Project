# Credit Risk Assessment — Retail Lending Portfolio
### Pacific Crest Bank (Hong Kong) · Group Risk & Analytics · June 2026

---

## Executive Summary

This report presents an end-to-end credit risk assessment of a 12,000-facility
sample of Pacific Crest Bank's Hong Kong retail and SME lending book. The
portfolio carries a **12-month default rate of 13.6%**, concentrated in
small-business facilities and unsecured revolving credit.

We built and benchmarked three predictive models and a regulatory-style credit
scorecard. The governed **WOE scorecard achieves a Gini of 0.65 (AUC 0.826,
KS 0.51)** — matching the black-box challengers while remaining fully
transparent and auditable, the right trade-off for a regulated origination
decision under HKMA model-risk expectations.

The scorecard produces **well-calibrated probabilities of default** and
rank-orders cleanly into five risk grades, with realised default rates ranging
from **1.6% (Grade A) to 46.6% (Grade E)**. We recommend a hard 50% DSR cap,
adoption of the five-grade scorecard for origination, risk-based pricing, and
portfolio steering toward the secured book.

---

## 1. Portfolio Profile

| Metric | Value |
|---|---|
| Facilities analysed | 12,000 |
| 12-month default rate (90+ DPD) | 13.6% |
| Products | Personal Loan, Credit Card, Mortgage, SME Facility |
| Currency | HKD |
| Default definition | 90+ days past due within 12 months |

**Default rate by product:**

| Product | Default rate |
|---|---|
| SME Facility | 26.5% |
| Credit Card | 14.7% |
| Personal Loan | 11.6% |
| Mortgage | 8.5% |

Small-business facilities and unsecured revolving credit run materially hotter
than the secured mortgage book, consistent with their risk profile and the
HKMA's LTV discipline on residential lending.

---

## 2. Key Risk Drivers

Drivers were identified consistently across three independent lenses —
Information Value (univariate), machine-learning feature importance, and
scorecard points:

| Rank | Driver | Information Value | Strength |
|---|---|---|---|
| 1 | **TU credit score** | 1.24 | Very strong |
| 2 | **Recent delinquencies (24m)** | 0.65 | Very strong |
| 3 | **Any prior delinquency** | 0.42 | Strong |
| 4 | **DSR headroom vs 50% guide** | 0.38 | Strong |
| 5 | **Debt Servicing Ratio (DSR)** | 0.38 | Strong |
| 6 | **Product type** | 0.16 | Medium |

**Bureau behaviour and affordability dominate.** The TU score and recent
delinquency history are the strongest signals, followed closely by the DSR
family of affordability measures — direct empirical support for the HKMA's
emphasis on debt-servicing capacity.

---

## 3. Model Performance (Test Set, 25% Hold-out)

| Model | AUC | Gini | KS | Avg. Precision |
|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.684 | 0.516 | 0.532 |
| Random Forest (challenger) | 0.835 | 0.670 | 0.502 | 0.512 |
| Gradient Boosting (challenger) | 0.826 | 0.652 | 0.486 | 0.499 |
| **WOE Scorecard (deployed)** | **0.826** | **0.651** | **0.510** | — |

The non-linear challengers offer **no meaningful uplift** over the transparent
logistic / scorecard approach. For a regulated lending decision the scorecard
is the clear choice: interpretable, monotonic, governable, and statistically
on par with the best challenger.

---

## 4. Risk Grades, Calibration & Expected Loss

The scorecard PD bands into five grades. Realised default rates track model PDs
closely — evidence of good calibration — and expected-loss rates rank-order
monotonically (EL = PD × LGD × EAD, LGD = 45%):

| Grade | Facilities | Actual Default | Model PD | EAD (HK$) | Expected Loss (HK$) | EL Rate |
|---|---|---|---|---|---|---|
| **A (Prime)** | 609 | 1.6% | 2.1% | 391.2M | 3.4M | 0.87% |
| **B (Low)** | 1,117 | 5.7% | 5.0% | 723.9M | 16.2M | 2.24% |
| **C (Medium)** | 477 | 10.1% | 10.8% | 318.6M | 16.0M | 5.03% |
| **D (High)** | 359 | 22.8% | 21.4% | 299.6M | 29.2M | 9.75% |
| **E (Decline)** | 438 | 46.6% | 49.5% | 379.7M | 82.7M | 21.77% |

Grade E holds **~15% of facilities but a disproportionate share of expected
loss** (HK$82.7M, ~54% of total modelled EL). The EL-rate spread (0.87% →
21.77%) is the basis for both risk-based pricing and IFRS 9 provisioning.

---

## 5. Illustrative Origination Policy

A policy that **auto-declines Grade E** yields, on the hold-out sample:

| Metric | Value |
|---|---|
| Approval rate | 85.4% |
| Bad-capture (defaults avoided) | 50.0% |

Declining the worst ~15% of applications removes roughly half of all future
defaults. Grade D (next ~12%) should route to manual underwriting rather than
auto-approval, capturing further loss with human oversight.

---

## 6. Recommendations

**Lending policy**
1. **Hard 50% DSR cap** with a stressed-rate overlay; second-line sign-off for
   50–65% exceptions; auto-decline above 65%.
2. **Adopt the five-grade scorecard (A–E)** for origination — auto-decline
   Grade E, manual underwrite Grade D, auto-approve A–C within policy.
3. **Risk-based pricing** aligned to grade-level EL rates so margin covers
   provisioning, especially on the SME and card books.
4. **Portfolio steering** — sub-limits on high-EL unsecured / SME segments;
   grow the secured mortgage book where EL rates are lowest.

**Provisioning & capital**
5. Use grade-level expected loss as the input to **IFRS 9 ECL** staging and
   management overlays.

**Model governance (HKMA-aligned)**
6. Monitor stability with **PSI** on out-of-time samples; recalibrate at least
   annually.
7. Add **macroeconomic overlays** (unemployment, HIBOR, residential price
   index) for forward-looking ECL.
8. Conduct **fair-lending / disparate-impact testing** before deployment.
9. Stand up monitoring: monthly score distribution, override rates, early-arrears.

---

## 7. Methodology & Reproducibility

- **Data:** synthetic HKD retail credit portfolio generated by
  `data/generate_data.py` with an economically grounded data-generating process
  (leverage, bureau history, affordability, product mix). No real customer data.
- **Pipeline:** loading → QA & cleaning (duplicates, implausible values,
  imputation) → EDA → feature engineering → WOE/IV screening → ML benchmarking
  → WOE scorecard → calibration, risk grading & expected loss.
- **Leakage control:** all imputation, scaling and WOE binning are learned on
  the training split only and applied to the hold-out.
- **Full analysis:** `notebooks/HK_Credit_Risk_Analysis.ipynb` (executed, with
  inline charts). All figures and tables are saved under `outputs/`.

---

*Prepared by Group Risk & Analytics, Pacific Crest Bank (Hong Kong). Synthetic
data for demonstration of credit-risk methodology only; not a real customer
portfolio and not investment or lending advice.*
