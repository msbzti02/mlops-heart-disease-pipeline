import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_shap_analysis(model, X_sample, feature_names, output_dir):
    import shap

    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    mlflow.log_artifact(path)
    print(f"    [OK] SHAP summary plot saved")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    mlflow.log_artifact(path)
    print(f"    [OK] SHAP bar plot saved")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1],
                data=X_sample.iloc[0].values,
                feature_names=feature_names
            ),
            show=False, max_display=15
        )
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_waterfall.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")
        mlflow.log_artifact(path)
        print(f"    [OK] SHAP waterfall plot saved")
    except Exception as e:
        print(f"    [SKIP] Waterfall plot: {e}")

    return shap_values


def run_lime_analysis(model, X_train_sample, X_explain, feature_names, output_dir):
    from lime.lime_tabular import LimeTabularExplainer

    print("  Computing LIME explanations...")

    explainer = LimeTabularExplainer(
        training_data=X_train_sample.values,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        mode="classification",
    )

    for i in range(min(3, len(X_explain))):
        try:
            exp = explainer.explain_instance(
                X_explain.iloc[i].values,
                model.predict_proba,
                num_features=15,
            )
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            path = os.path.join(output_dir, f"lime_sample_{i+1}.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close("all")
            mlflow.log_artifact(path)
            print(f"    [OK] LIME explanation for sample {i+1} saved")
        except Exception as e:
            print(f"    [SKIP] LIME sample {i+1}: {e}")

    try:
        lime_exp = explainer.explain_instance(
            X_explain.iloc[0].values,
            model.predict_proba,
            num_features=10,
        )
        lime_features = [f[0] for f in lime_exp.as_list()]
        print(f"    LIME top features: {lime_features[:5]}")
    except Exception:
        pass


def main():
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    proc_dir = config["data"]["processed_dir"]
    prefix = "brfss"

    X_train = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_test.csv"))

    from src.feature_engineering import engineer_brfss_features
    X_train = engineer_brfss_features(X_train)
    X_test = engineer_brfss_features(X_test)

    feature_names = X_train.columns.tolist()

    model_name = config["mlflow"]["registry_model_name"]
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.model_type = 'GradientBoosting_XGB_GPU'",
        order_by=["metrics.auc DESC"],
        max_results=1,
    )

    if not runs:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`mlflow.parentRunId` IS NULL",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )

    if not runs:
        print("[ERROR] No model runs found")
        return

    run_id = runs[0].info.run_id
    model_type = runs[0].data.params.get("model_type", "unknown")
    print(f"[INFO] Loading {model_type} from run {run_id}")

    try:
        model_obj = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    except Exception:
        model_obj = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    X_train_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)

    output_dir = "artifacts/explainability"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS (SHAP + LIME)")
    print(f"{'='*60}")

    with mlflow.start_run(run_name="Explainability_Analysis"):
        print("\n[1] SHAP Analysis...")
        try:
            run_shap_analysis(model_obj, X_sample, feature_names, output_dir)
        except Exception as e:
            print(f"  [ERROR] SHAP failed: {e}")

        print("\n[2] LIME Analysis...")
        try:
            run_lime_analysis(model_obj, X_train_sample, X_sample,
                              feature_names, output_dir)
        except Exception as e:
            print(f"  [ERROR] LIME failed: {e}")

        print(f"\n[OK] Explainability analysis complete!")


if __name__ == "__main__":
    main()
