import pandas as pd
import numpy as np


def engineer_brfss_features(df):
    print("[Feature Engineering] BRFSS dataset...")
    original_cols = len(df.columns)

    if "Age" in df.columns:
        df["age_squared"] = df["Age"] ** 2
        df["age_bin"] = pd.cut(
            df["Age"], bins=[0, 35, 50, 65, 100],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(1)
        df["is_senior"] = (df["Age"] >= 65).astype(int)

    if "BMI" in df.columns:
        df["bmi_squared"] = df["BMI"] ** 2
        df["bmi_category"] = pd.cut(
            df["BMI"], bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(1)
        df["is_obese"] = (df["BMI"] >= 30).astype(int)

    if "PhysicalHealth" in df.columns and "MentalHealth" in df.columns:
        df["total_health_burden"] = df["PhysicalHealth"] + df["MentalHealth"]
        df["health_ratio"] = (
            df["PhysicalHealth"] / (df["MentalHealth"] + 1)
        )

    risk_cols = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                 "KidneyDisease", "SkinCancer", "Diabetic"]
    available_risk = [c for c in risk_cols if c in df.columns]
    if available_risk:
        df["risk_factor_count"] = df[available_risk].sum(axis=1)

    if "SleepTime" in df.columns:
        df["sleep_deviation"] = abs(df["SleepTime"] - 7)
        df["poor_sleep"] = (
            (df["SleepTime"] < 6) | (df["SleepTime"] > 9)
        ).astype(int)

    if "Age" in df.columns and "BMI" in df.columns:
        df["age_x_bmi"] = df["Age"] * df["BMI"]
    if "Age" in df.columns and "Smoking" in df.columns:
        df["age_x_smoking"] = df["Age"] * df["Smoking"]
    if "GenHealth" in df.columns and "PhysicalHealth" in df.columns:
        df["genhealth_x_physical"] = df["GenHealth"] * df["PhysicalHealth"]

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    new_cols = len(df.columns) - original_cols
    print(f"  [OK] Added {new_cols} features ({original_cols} -> {len(df.columns)})")
    return df


def engineer_clinical_features(df):
    print("[Feature Engineering] Clinical dataset...")
    original_cols = len(df.columns)

    if "Age" in df.columns:
        df["age_squared"] = df["Age"] ** 2
        df["is_senior"] = (df["Age"] >= 60).astype(int)

    if "trestbps" in df.columns:
        df["bp_category"] = pd.cut(
            df["trestbps"], bins=[0, 120, 130, 140, 300],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(1)

    if "chol" in df.columns:
        df["chol_category"] = pd.cut(
            df["chol"], bins=[0, 200, 240, 1000],
            labels=[0, 1, 2]
        ).astype(float).fillna(1)
        df["high_chol"] = (df["chol"] > 240).astype(int)

    if "thalch" in df.columns:
        df["hr_reserve"] = 220 - df["Age"] - df["thalch"] if "Age" in df.columns else 0

    if "oldpeak" in df.columns and "Age" in df.columns:
        df["oldpeak_x_age"] = df["oldpeak"] * df["Age"]

    risk_components = []
    if "Age" in df.columns:
        risk_components.append((df["Age"] > 55).astype(int))
    if "trestbps" in df.columns:
        risk_components.append((df["trestbps"] > 140).astype(int))
    if "chol" in df.columns:
        risk_components.append((df["chol"] > 240).astype(int))
    if "fbs" in df.columns:
        risk_components.append(df["fbs"].astype(int))
    if risk_components:
        df["clinical_risk_score"] = sum(risk_components)

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    new_cols = len(df.columns) - original_cols
    print(f"  [OK] Added {new_cols} features ({original_cols} -> {len(df.columns)})")
    return df


if __name__ == "__main__":
    import os, yaml

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    proc_dir = config["data"]["processed_dir"]

    brfss_train = os.path.join(proc_dir, "brfss_X_train.csv")
    if os.path.exists(brfss_train):
        df = pd.read_csv(brfss_train)
        df = engineer_brfss_features(df)
        df.to_csv(brfss_train, index=False)

    clin_train = os.path.join(proc_dir, "clinical_X_train.csv")
    if os.path.exists(clin_train):
        df = pd.read_csv(clin_train)
        df = engineer_clinical_features(df)
        df.to_csv(clin_train, index=False)

    print("[OK] Feature engineering complete!")
