import numpy as np
import pandas as pd


def engineer_features(df_raw):

    df = df_raw.copy()
    df = df.sort_values(["customer_id","month"]).reset_index(drop=True)

    NUMERIC_COLS = [
        "active_products_count",
        "credit_card_utilization",
        "total_monthly_obligation",
        "emi_amount",
        "days_to_emi",
        "emi_to_income_ratio",
        "salary_delay_days",
        "weekly_balance_change_pct",
        "atm_withdrawal_amount",
        "monthly_income"
    ]

    
    # LAG FEATURES (works for 2+ months)
    
    for col in NUMERIC_COLS:
        df[f"{col}_lag1"] = df.groupby("customer_id")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("customer_id")[col].shift(2)

    
    # DELTA FEATURES (works if lag exists)
    
    for col in NUMERIC_COLS:
        df[f"{col}_delta_1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_delta_2"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]

    
    # VOLATILITY (only if >=2 months)
    
    VOL_COLS = [
        "credit_card_utilization",
        "weekly_balance_change_pct",
        "monthly_income"
    ]

    for col in VOL_COLS:
        df[f"{col}_std_3m"] = (
            df.groupby("customer_id")[col]
              .rolling(3, min_periods=1)
              .std()
              .reset_index(level=0, drop=True)
        )

    
    # ROLLING FEATURES (min_periods=1 → safe)
    
    ROLLING_COLS = [
        "credit_card_utilization",
        "monthly_income",
        "emi_to_income_ratio",
        "weekly_balance_change_pct"
    ]

    for col in ROLLING_COLS:
        g = df.groupby("customer_id")[col]

        df[f"{col}_mean_3m"] = (
            g.rolling(3, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
        )

        df[f"{col}_max_3m"] = (
            g.rolling(3, min_periods=1)
             .max()
             .reset_index(level=0, drop=True)
        )

        df[f"{col}_slope_3m"] = (
            g.rolling(3, min_periods=2)
             .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
             .reset_index(level=0, drop=True)
        )

    
    # PERSISTENCE FLAGS (safe version)
    
    df["emi_high_persist_3m"] = (
        (df["emi_to_income_ratio"] > 0.40)
        .groupby(df["customer_id"])
        .rolling(3, min_periods=1)
        .sum()
        .ge(2)
        .reset_index(level=0, drop=True)
        .astype(int)
    )

    df["salary_delay_persist_3m"] = (
        (df["salary_delay_days"] > 5)
        .groupby(df["customer_id"])
        .rolling(3, min_periods=1)
        .sum()
        .ge(2)
        .reset_index(level=0, drop=True)
        .astype(int)
    )

    df["utilization_high_persist_3m"] = (
        (df["credit_card_utilization"] > 0.75)
        .groupby(df["customer_id"])
        .rolling(3, min_periods=1)
        .sum()
        .ge(2)
        .reset_index(level=0, drop=True)
        .astype(int)
    )

    
    # FILL REMAINING NaNs SAFELY
    
    df = df.fillna(0)

    return df