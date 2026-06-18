import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreprocessing:

    def test_brfss_2020_exists(self):
        assert os.path.exists("data/raw/heart_2020_cleaned.csv")

    def test_brfss_2022_exists(self):
        assert os.path.exists("data/raw/heart_2022_no_nans.csv")

    def test_clinical_heart_exists(self):
        assert os.path.exists("data/raw/heart.csv")

    def test_uci_heart_exists(self):
        assert os.path.exists("data/raw/heart_disease_uci.csv")

    def test_processed_files_exist(self):
        files = [
            "data/processed/brfss_X_train.csv",
            "data/processed/brfss_X_val.csv",
            "data/processed/brfss_X_test.csv",
            "data/processed/brfss_y_train.csv",
            "data/processed/brfss_y_val.csv",
            "data/processed/brfss_y_test.csv",
        ]
        for f in files:
            assert os.path.exists(f), f"Missing: {f}"

    def test_train_data_shape(self):
        X = pd.read_csv("data/processed/brfss_X_train.csv")
        y = pd.read_csv("data/processed/brfss_y_train.csv")
        assert len(X) == len(y)
        assert len(X) > 100000
        assert X.shape[1] >= 18

    def test_no_missing_values(self):
        X = pd.read_csv("data/processed/brfss_X_train.csv")
        assert X.isnull().sum().sum() == 0

    def test_binary_target(self):
        y = pd.read_csv("data/processed/brfss_y_train.csv").squeeze()
        assert set(y.unique()).issubset({0, 1})

    def test_class_balance_after_smote(self):
        y = pd.read_csv("data/processed/brfss_y_train.csv").squeeze()
        counts = y.value_counts()
        ratio = counts.min() / counts.max()
        assert ratio > 0.9


class TestFeatureEngineering:

    def test_brfss_features(self):
        from src.feature_engineering import engineer_brfss_features
        df = pd.DataFrame({
            "Age": [55, 30], "BMI": [28.5, 22.0],
            "PhysicalHealth": [5, 0], "MentalHealth": [3, 1],
            "Smoking": [1, 0], "AlcoholDrinking": [0, 0],
            "Stroke": [0, 0], "DiffWalking": [0, 0],
            "KidneyDisease": [0, 0], "SkinCancer": [0, 0],
            "Diabetic": [0, 0], "SleepTime": [7, 8],
            "GenHealth": [2, 3],
        })
        original_cols = len(df.columns)
        result = engineer_brfss_features(df)
        assert len(result.columns) > original_cols
        assert "age_squared" in result.columns
        assert "risk_factor_count" in result.columns
        assert result.isnull().sum().sum() == 0

    def test_clinical_features(self):
        from src.feature_engineering import engineer_clinical_features
        df = pd.DataFrame({
            "Age": [55, 60], "trestbps": [130, 160],
            "chol": [250, 200], "thalch": [150, 120],
            "oldpeak": [1.5, 2.0], "fbs": [1, 0],
        })
        result = engineer_clinical_features(df)
        assert "age_squared" in result.columns
        assert "clinical_risk_score" in result.columns


class TestConfig:

    def test_config_exists(self):
        assert os.path.exists("configs/config.yaml")

    def test_config_structure(self):
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        assert "data" in config
        assert "mlflow" in config
        assert "models" in config
        assert "tuning" in config
        assert "serving" in config
        assert "monitoring" in config


class TestMLflowIntegration:

    def test_mlflow_db_exists(self):
        assert os.path.exists("mlflow.db")

    def test_artifacts_exist(self):
        assert os.path.isdir("artifacts")


class TestServing:

    def test_serve_module_imports(self):
        assert os.path.exists("src/serve.py")

    def test_html_template_in_serve(self):
        with open("src/serve.py") as f:
            content = f.read()
        assert "predict-form" in content
        assert "/health" in content
        assert "/predict" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
