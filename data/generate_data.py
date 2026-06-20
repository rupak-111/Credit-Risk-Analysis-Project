"""
Synthetic credit portfolio generator for a Hong Kong based multinational bank.

Produces a retail lending book (personal loans, credit cards, mortgages, SME
facilities) denominated in HKD with features that mirror the data a Hong Kong
credit risk team actually underwrites on:

  * TransUnion (TU) HK credit score
  * Debt Servicing Ratio (DSR) - a core HKMA supervisory metric
  * Loan-to-Value (LTV) for secured lending - capped under HKMA guidelines
  * Residential status (owner / private rental / public housing)
  * Employment sector and tenure

The default flag (12-month definition: 90+ days past due) is produced by a
transparent data-generating process so the downstream modelling recovers
economically sensible relationships, with enough noise to stay realistic.

Run:  python generate_data.py
Out:  credit_portfolio.csv  (one row per booked facility)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260620)
N = 12000


def _bounded(values, lo, hi):
    return np.clip(values, lo, hi)


def generate(n: int = N) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Applicant demographics
    # ------------------------------------------------------------------ #
    age = _bounded(RNG.normal(41, 11, n).round(), 21, 70).astype(int)

    gender = RNG.choice(["M", "F"], size=n, p=[0.52, 0.48])

    # Hong Kong districts grouped into three broad regions
    district = RNG.choice(
        ["HK Island", "Kowloon", "New Territories"],
        size=n,
        p=[0.24, 0.34, 0.42],
    )

    residential_status = RNG.choice(
        ["Owner", "Private Rental", "Public Housing", "Living with Family"],
        size=n,
        p=[0.42, 0.30, 0.18, 0.10],
    )

    employment_sector = RNG.choice(
        [
            "Finance & Insurance",
            "Trade & Logistics",
            "Professional Services",
            "Retail & F&B",
            "Construction",
            "Public Sector",
            "Self-Employed",
        ],
        size=n,
        p=[0.18, 0.16, 0.15, 0.17, 0.10, 0.12, 0.12],
    )

    # Employment tenure in years (self-employed / retail tend to be shorter)
    base_tenure = RNG.gamma(shape=2.0, scale=3.0, size=n)
    sector_tenure_adj = np.select(
        [
            employment_sector == "Public Sector",
            employment_sector == "Finance & Insurance",
            np.isin(employment_sector, ["Retail & F&B", "Self-Employed"]),
        ],
        [3.0, 1.5, -1.0],
        default=0.0,
    )
    employment_years = _bounded(base_tenure + sector_tenure_adj, 0, 35).round(1)

    # ------------------------------------------------------------------ #
    # 2. Income (monthly, HKD) - driven by age, sector, tenure
    # ------------------------------------------------------------------ #
    sector_income_mult = np.select(
        [
            employment_sector == "Finance & Insurance",
            employment_sector == "Professional Services",
            employment_sector == "Public Sector",
            employment_sector == "Construction",
            employment_sector == "Retail & F&B",
            employment_sector == "Self-Employed",
        ],
        [1.55, 1.35, 1.10, 0.95, 0.78, 1.05],
        default=1.0,
    )
    age_income_factor = 1 + (age - 21) * 0.012
    income_noise = RNG.lognormal(mean=0.0, sigma=0.35, size=n)
    monthly_income = _bounded(
        18000 * sector_income_mult * age_income_factor * income_noise,
        9000,
        260000,
    ).round(-2)

    # ------------------------------------------------------------------ #
    # 3. Product / facility
    # ------------------------------------------------------------------ #
    product_type = RNG.choice(
        ["Personal Loan", "Credit Card", "Mortgage", "SME Facility"],
        size=n,
        p=[0.34, 0.30, 0.24, 0.12],
    )

    # Loan amount depends on product and income
    annual_income = monthly_income * 12
    loan_amount = np.empty(n)
    for prod, lo_mult, hi_mult in [
        ("Personal Loan", 0.2, 1.2),
        ("Credit Card", 0.1, 0.6),
        ("Mortgage", 3.5, 9.0),
        ("SME Facility", 0.8, 4.0),
    ]:
        mask = product_type == prod
        mult = RNG.uniform(lo_mult, hi_mult, size=mask.sum())
        loan_amount[mask] = annual_income[mask] * mult
    loan_amount = _bounded(loan_amount, 20000, 18_000_000).round(-3)

    # Tenor in months
    tenor_months = np.select(
        [
            product_type == "Credit Card",
            product_type == "Personal Loan",
            product_type == "SME Facility",
            product_type == "Mortgage",
        ],
        [
            RNG.choice([12, 24], size=n),
            RNG.choice([12, 24, 36, 48, 60], size=n),
            RNG.choice([12, 24, 36, 60], size=n),
            RNG.choice([120, 180, 240, 300, 360], size=n),
        ],
    )

    # Interest rate (APR, %) - secured lending cheaper, cards expensive
    base_rate = np.select(
        [
            product_type == "Mortgage",
            product_type == "Personal Loan",
            product_type == "SME Facility",
            product_type == "Credit Card",
        ],
        [
            RNG.normal(3.4, 0.5, n),
            RNG.normal(7.5, 1.6, n),
            RNG.normal(8.5, 1.8, n),
            RNG.normal(28.0, 3.0, n),
        ],
    )
    interest_rate = _bounded(base_rate, 1.8, 36.0).round(2)

    # ------------------------------------------------------------------ #
    # 4. Bureau / leverage features
    # ------------------------------------------------------------------ #
    num_existing_loans = RNG.poisson(1.3, n)
    num_credit_cards = RNG.poisson(2.1, n) + 1

    # Existing monthly debt obligations (HKD)
    existing_debt_payment = _bounded(
        monthly_income * RNG.beta(2.0, 8.0, n),  # mostly modest, some heavy
        0,
        monthly_income * 0.7,
    ).round(-1)

    # New facility monthly payment (simple amortised approximation)
    monthly_rate = interest_rate / 100 / 12
    # avoid div by zero
    monthly_rate = np.where(monthly_rate == 0, 1e-6, monthly_rate)
    new_payment = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** tenor_months
        / ((1 + monthly_rate) ** tenor_months - 1)
    )
    # Revolving / facility products are not fully amortised: model a monthly
    # servicing burden rather than a 12-month paydown.
    #   Credit cards  ~ 4% minimum payment of balance
    #   SME facility  ~ 2% interest-and-fee servicing of the limit
    new_payment = np.where(
        product_type == "Credit Card", loan_amount * 0.04, new_payment
    )
    new_payment = np.where(
        product_type == "SME Facility", loan_amount * 0.02, new_payment
    )
    new_payment = new_payment.round(-1)

    # Debt Servicing Ratio = total monthly debt / monthly income (HKMA metric)
    dsr = _bounded(
        (existing_debt_payment + new_payment) / monthly_income, 0.02, 1.5
    ).round(3)

    # Loan-to-Value (only meaningful for secured products)
    ltv = np.where(
        np.isin(product_type, ["Mortgage", "SME Facility"]),
        _bounded(RNG.normal(0.62, 0.12, n), 0.20, 0.95),
        np.nan,
    ).round(3)

    # Past delinquencies in last 24 months
    delinq_24m = RNG.choice([0, 1, 2, 3, 4], size=n, p=[0.70, 0.16, 0.08, 0.04, 0.02])

    # Months since last delinquency (NaN if never)
    months_since_delinq = np.where(
        delinq_24m > 0,
        RNG.integers(1, 24, n),
        np.nan,
    )

    credit_history_years = _bounded(
        (age - 21) * RNG.uniform(0.4, 0.95, n), 0, 45
    ).round(1)

    # ------------------------------------------------------------------ #
    # 5. TU credit score (300-850) - synthesised from drivers then used
    #    as an observed feature.
    # ------------------------------------------------------------------ #
    score_linear = (
        680
        - delinq_24m * 38
        - (dsr - 0.4) * 120
        - num_existing_loans * 6
        + credit_history_years * 2.2
        + (employment_years * 1.5)
        + RNG.normal(0, 28, n)
    )
    tu_score = _bounded(score_linear, 300, 850).round().astype(int)

    # ------------------------------------------------------------------ #
    # 6. Default data-generating process (12-month, 90+ DPD)
    #    Logit driven by leverage, history, affordability, product, macro.
    # ------------------------------------------------------------------ #
    sector_risk = np.select(
        [
            employment_sector == "Public Sector",
            employment_sector == "Finance & Insurance",
            np.isin(employment_sector, ["Retail & F&B", "Construction"]),
            employment_sector == "Self-Employed",
        ],
        [-0.35, -0.15, 0.30, 0.40],
        default=0.0,
    )

    res_risk = np.select(
        [
            residential_status == "Owner",
            residential_status == "Public Housing",
            residential_status == "Living with Family",
        ],
        [-0.30, 0.10, 0.15],
        default=0.0,  # Private Rental baseline
    )

    product_risk = np.select(
        [
            product_type == "Mortgage",
            product_type == "Credit Card",
            product_type == "SME Facility",
        ],
        [-0.90, 1.05, 0.70],
        default=0.0,  # Personal Loan baseline
    )

    logit = (
        -3.05
        + 2.6 * (dsr - 0.45)
        + 0.55 * delinq_24m
        - 0.011 * (tu_score - 650)
        - 0.045 * (monthly_income / 10000 - 3).clip(-2, 6)
        - 0.03 * employment_years
        + 0.10 * num_existing_loans
        + 1.1 * np.nan_to_num((ltv - 0.6), nan=0.0)
        + sector_risk
        + res_risk
        + product_risk
        + 0.015 * (interest_rate - 8).clip(-6, 28)
        + RNG.normal(0, 0.45, n)  # unexplained heterogeneity
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_flag = (RNG.uniform(0, 1, n) < prob_default).astype(int)

    # ------------------------------------------------------------------ #
    # 7. Assemble
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(
        {
            "customer_id": [f"HK{1000000 + i}" for i in range(n)],
            "age": age,
            "gender": gender,
            "district": district,
            "residential_status": residential_status,
            "employment_sector": employment_sector,
            "employment_years": employment_years,
            "monthly_income_hkd": monthly_income.astype(int),
            "product_type": product_type,
            "loan_amount_hkd": loan_amount.astype(int),
            "tenor_months": tenor_months.astype(int),
            "interest_rate_pct": interest_rate,
            "num_existing_loans": num_existing_loans,
            "num_credit_cards": num_credit_cards,
            "existing_debt_payment_hkd": existing_debt_payment.astype(int),
            "new_payment_hkd": new_payment.astype(int),
            "dsr": dsr,
            "ltv": ltv,
            "delinq_24m": delinq_24m,
            "months_since_delinq": months_since_delinq,
            "credit_history_years": credit_history_years,
            "tu_credit_score": tu_score,
            "default_flag": default_flag,
        }
    )

    # ------------------------------------------------------------------ #
    # 8. Inject realistic data-quality issues for the cleaning step
    # ------------------------------------------------------------------ #
    # ~1.5% missing income (to be imputed)
    miss_income = RNG.choice(n, size=int(0.015 * n), replace=False)
    df.loc[miss_income, "monthly_income_hkd"] = np.nan

    # ~1% missing employment tenure
    miss_emp = RNG.choice(n, size=int(0.01 * n), replace=False)
    df.loc[miss_emp, "employment_years"] = np.nan

    # A handful of duplicate rows
    dups = df.sample(25, random_state=7)
    df = pd.concat([df, dups], ignore_index=True)

    # A few implausible ages to be caught in QA
    bad_age = RNG.choice(len(df), size=8, replace=False)
    df.loc[bad_age, "age"] = RNG.choice([0, 5, 999], size=8)

    return df


if __name__ == "__main__":
    data = generate()
    out = "credit_portfolio.csv"
    data.to_csv(out, index=False)
    rate = data["default_flag"].mean()
    print(f"Wrote {out}: {len(data):,} rows, {data.shape[1]} columns")
    print(f"Portfolio default rate: {rate:.2%}")
    print(data["product_type"].value_counts())
