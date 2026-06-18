import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import yaml
import warnings

warnings.filterwarnings("ignore")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_brfss_2020(raw_dir):
    path = os.path.join(raw_dir, "heart_2020_cleaned.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["HeartDisease"] = (df["HeartDisease"] == "Yes").astype(int)
    df["Sex"] = (df["Sex"] == "Male").astype(int)
    df["Smoking"] = (df["Smoking"] == "Yes").astype(int)
    df["AlcoholDrinking"] = (df["AlcoholDrinking"] == "Yes").astype(int)
    df["Stroke"] = (df["Stroke"] == "Yes").astype(int)
    df["DiffWalking"] = (df["DiffWalking"] == "Yes").astype(int)
    df["PhysicalActivity"] = (df["PhysicalActivity"] == "Yes").astype(int)
    df["Asthma"] = (df["Asthma"] == "Yes").astype(int)
    df["KidneyDisease"] = (df["KidneyDisease"] == "Yes").astype(int)
    df["SkinCancer"] = (df["SkinCancer"] == "Yes").astype(int)
    df["Diabetic"] = df["Diabetic"].map({
        "Yes": 2, "No": 0,
        "No, borderline diabetes": 1,
        "Yes (during pregnancy)": 1
    }).fillna(0).astype(int)
    gen_map = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    df["GenHealth"] = df["GenHealth"].map(gen_map).fillna(2).astype(int)
    age_map = {
        "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
        "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
        "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
        "80 or older": 82
    }
    df["Age"] = df["AgeCategory"].map(age_map).fillna(50).astype(int)
    race_dummies = pd.get_dummies(df["Race"], prefix="Race", dtype=int)
    df = pd.concat([df, race_dummies], axis=1)
    df.drop(columns=["AgeCategory", "Race"], inplace=True)
    df["source"] = "brfss_2020"
    print(f"  [OK] BRFSS 2020: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_brfss_2022(raw_dir):
    path = os.path.join(raw_dir, "heart_2022_no_nans.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["HeartDisease"] = (
        (df["HadHeartAttack"] == "Yes") | (df["HadAngina"] == "Yes")
    ).astype(int)
    df["Sex"] = (df["Sex"] == "Male").astype(int)
    df["Smoking"] = df["SmokerStatus"].isin(
        ["Current smoker - now smokes every day",
         "Current smoker - now smokes some days"]
    ).astype(int)
    df["AlcoholDrinking"] = (df["AlcoholDrinkers"] == "Yes").astype(int)
    df["Stroke"] = (df["HadStroke"] == "Yes").astype(int)
    df["DiffWalking"] = (df["DifficultyWalking"] == "Yes").astype(int)
    df["PhysicalActivity"] = (df["PhysicalActivities"] == "Yes").astype(int)
    df["Asthma"] = (df["HadAsthma"] == "Yes").astype(int)
    df["KidneyDisease"] = (df["HadKidneyDisease"] == "Yes").astype(int)
    df["SkinCancer"] = (df["HadSkinCancer"] == "Yes").astype(int)
    df["Diabetic"] = df["HadDiabetes"].map({
        "Yes": 2, "No": 0,
        "No, pre-diabetes or borderline diabetes": 1,
        "Yes, but only during pregnancy (female)": 1
    }).fillna(0).astype(int)
    gen_map = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    df["GenHealth"] = df["GeneralHealth"].map(gen_map).fillna(2).astype(int)
    age_map = {
        "Age 18 to 24": 21, "Age 25 to 29": 27, "Age 30 to 34": 32,
        "Age 35 to 39": 37, "Age 40 to 44": 42, "Age 45 to 49": 47,
        "Age 50 to 54": 52, "Age 55 to 59": 57, "Age 60 to 64": 62,
        "Age 65 to 69": 67, "Age 70 to 74": 72, "Age 75 to 79": 77,
        "Age 80 or older": 82
    }
    df["Age"] = df["AgeCategory"].map(age_map).fillna(50).astype(int)
    df["PhysicalHealth"] = df["PhysicalHealthDays"]
    df["MentalHealth"] = df["MentalHealthDays"]
    df["SleepTime"] = df["SleepHours"]
    race_map = {
        "White only, Non-Hispanic": "White",
        "Black only, Non-Hispanic": "Black",
        "Hispanic": "Hispanic",
        "Asian only, Non-Hispanic": "Asian",
        "American Indian/Alaskan Native only, Non-Hispanic": "American Indian/Alaskan Native",
        "Other race only, Non-Hispanic": "Other",
        "Multiracial, Non-Hispanic": "Other"
    }
    df["Race_mapped"] = df["RaceEthnicityCategory"].map(race_map).fillna("Other")
    race_dummies = pd.get_dummies(df["Race_mapped"], prefix="Race", dtype=int)
    df = pd.concat([df, race_dummies], axis=1)
    common_cols = [
        "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke",
        "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex",
        "PhysicalActivity", "GenHealth", "SleepTime", "Asthma",
        "KidneyDisease", "SkinCancer", "Diabetic", "Age"
    ]
    race_cols = [c for c in df.columns if c.startswith("Race_")]
    keep_cols = common_cols + race_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["source"] = "brfss_2022"
    print(f"  [OK] BRFSS 2022: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_clinical_heart(raw_dir):
    path = os.path.join(raw_dir, "heart.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.rename(columns={
        "ChestPainType": "cp", "RestingBP": "trestbps",
        "Cholesterol": "chol", "FastingBS": "fbs",
        "RestingECG": "restecg", "MaxHR": "thalch",
        "ExerciseAngina": "exang", "Oldpeak": "oldpeak",
        "ST_Slope": "slope"
    }, inplace=True)
    df["Sex"] = (df["Sex"] == "M").astype(int)
    df["exang"] = (df["exang"] == "Y").astype(int)
    cp_dummies = pd.get_dummies(df["cp"], prefix="cp", dtype=int)
    ecg_dummies = pd.get_dummies(df["restecg"], prefix="ecg", dtype=int)
    slope_dummies = pd.get_dummies(df["slope"], prefix="slope", dtype=int)
    df = pd.concat([df, cp_dummies, ecg_dummies, slope_dummies], axis=1)
    df.drop(columns=["cp", "restecg", "slope"], inplace=True)
    df.rename(columns={"HeartDisease": "target"}, inplace=True)
    df["source"] = "clinical_heart"
    print(f"  [OK] Clinical heart.csv: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_uci_heart(raw_dir):
    path = os.path.join(raw_dir, "heart_disease_uci.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["target"] = (df["num"] > 0).astype(int)
    df["Sex"] = (df["sex"] == "Male").astype(int)
    df["exang"] = df["exang"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
    df["fbs"] = df["fbs"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
    df.rename(columns={"age": "Age"}, inplace=True)
    cp_dummies = pd.get_dummies(df["cp"], prefix="cp", dtype=int)
    ecg_dummies = pd.get_dummies(df["restecg"], prefix="ecg", dtype=int)
    slope_dummies = pd.get_dummies(df["slope"], prefix="slope", dtype=int)
    thal_dummies = pd.get_dummies(df["thal"], prefix="thal", dtype=int)
    df = pd.concat([df, cp_dummies, ecg_dummies, slope_dummies, thal_dummies], axis=1)
    df.drop(columns=["id", "sex", "dataset", "cp", "restecg", "slope",
                      "thal", "num"], inplace=True)
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = "uci"
    print(f"  [OK] UCI multi-hospital: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_cleveland(raw_dir):
    path = os.path.join(raw_dir, "heart_cleveland_upload.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.rename(columns={"condition": "target", "age": "Age", "sex": "Sex",
                        "thalach": "thalch"}, inplace=True)
    df["source"] = "cleveland"
    print(f"  [OK] Cleveland: {len(df):,} rows, {len(df.columns)} cols")
    return df


def build_brfss_dataset(raw_dir, config):
    print("\n[Phase 2A] Loading BRFSS Survey Datasets...")
    df_2020 = load_brfss_2020(raw_dir)
    df_2022 = load_brfss_2022(raw_dir)

    frames = [d for d in [df_2020, df_2022] if d is not None]
    if not frames:
        print("  [ERROR] No BRFSS datasets found")
        return None

    common = set(frames[0].columns)
    for f in frames[1:]:
        common = common.intersection(f.columns)
    common = sorted(common)

    aligned = [f[common] for f in frames]
    combined = pd.concat(aligned, ignore_index=True)
    combined.fillna(0, inplace=True)

    print(f"  [OK] Combined BRFSS: {len(combined):,} rows, {len(combined.columns)} cols")
    return combined


def build_clinical_dataset(raw_dir, config):
    print("\n[Phase 2B] Loading Clinical Datasets...")
    df_heart = load_clinical_heart(raw_dir)
    df_uci = load_uci_heart(raw_dir)
    df_clev = load_cleveland(raw_dir)

    frames = [d for d in [df_heart, df_uci, df_clev] if d is not None]
    if not frames:
        print("  [ERROR] No clinical datasets found")
        return None

    common = set(frames[0].columns)
    for f in frames[1:]:
        common = common.intersection(f.columns)
    common = sorted(common)

    aligned = [f[common] for f in frames]
    combined = pd.concat(aligned, ignore_index=True)
    for col in combined.select_dtypes(include=[object]).columns:
        if col != "source":
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined.fillna(combined.median(numeric_only=True), inplace=True)

    print(f"  [OK] Combined Clinical: {len(combined):,} rows, {len(combined.columns)} cols")
    return combined


def preprocess_data(df, config, target_col="HeartDisease",
                    apply_smote=True, save=True, prefix="brfss"):
    print(f"\n[Preprocessing] {prefix} dataset ({len(df):,} rows)...")

    if "source" in df.columns:
        df = df.drop(columns=["source"])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    for col in X.select_dtypes(include=[object]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X.fillna(0, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)

    rs = config["data"]["random_state"]
    ts = config["data"]["test_size"]
    vs = config["data"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=ts, random_state=rs, stratify=y
    )
    val_ratio = vs / (1 - ts)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=rs, stratify=y_temp
    )

    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=feature_names, index=X_val.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    print(f"  Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    print(f"  Class balance (train): {dict(y_train.value_counts())}")

    if apply_smote:
        sm = SMOTE(random_state=rs)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=feature_names)
        y_train = pd.Series(y_train, name=target_col)
        print(f"  After SMOTE: train={len(X_train):,}, balance={dict(y_train.value_counts())}")

    if save:
        out_dir = config["data"]["processed_dir"]
        os.makedirs(out_dir, exist_ok=True)
        X_train.to_csv(os.path.join(out_dir, f"{prefix}_X_train.csv"), index=False)
        X_val.to_csv(os.path.join(out_dir, f"{prefix}_X_val.csv"), index=False)
        X_test.to_csv(os.path.join(out_dir, f"{prefix}_X_test.csv"), index=False)
        y_train.to_csv(os.path.join(out_dir, f"{prefix}_y_train.csv"), index=False)
        y_val.to_csv(os.path.join(out_dir, f"{prefix}_y_val.csv"), index=False)
        y_test.to_csv(os.path.join(out_dir, f"{prefix}_y_test.csv"), index=False)
        print(f"  [OK] Saved to {out_dir}/{prefix}_*.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    config = load_config()
    raw_dir = config["data"]["raw_dir"]

    df_brfss = build_brfss_dataset(raw_dir, config)
    if df_brfss is not None:
        preprocess_data(df_brfss, config, target_col="HeartDisease",
                        apply_smote=True, prefix="brfss")

    df_clinical = build_clinical_dataset(raw_dir, config)
    if df_clinical is not None:
        preprocess_data(df_clinical, config, target_col="target",
                        apply_smote=True, prefix="clinical")

    print("\n[OK] All datasets preprocessed successfully!")
