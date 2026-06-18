import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import json

import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, matthews_corrcoef,
    classification_report
)
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inject_drift(X_batch, drift_magnitude=5.0):
    drifted = X_batch.copy()
    n_features = min(10, len(drifted.columns))
    drift_cols = drifted.columns[:n_features]
    for col in drift_cols:
        std = drifted[col].std()
        if std > 0:
            drifted[col] = drifted[col] + drift_magnitude * std
    return drifted


def ks_drift_test(ref_data, batch_data, threshold=0.05):
    drift_results = {}
    n_drifted = 0

    for col in ref_data.columns:
        if col not in batch_data.columns:
            continue
        stat, p_value = stats.ks_2samp(ref_data[col], batch_data[col])
        is_drifted = p_value < threshold
        drift_results[col] = {
            "statistic": float(stat), "p_value": float(p_value),
            "drifted": bool(is_drifted)
        }
        if is_drifted:
            n_drifted += 1

    total = len(drift_results)
    drift_pct = n_drifted / total if total > 0 else 0
    return {
        "feature_results": drift_results,
        "n_features_drifted": n_drifted,
        "total_features": total,
        "drift_percentage": drift_pct,
        "dataset_drift": drift_pct > 0.25,
    }


def main():
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    proc_dir = config["data"]["processed_dir"]
    prefix = "brfss"

    X_val = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_val.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_test.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, f"{prefix}_y_test.csv")).squeeze()

    from src.feature_engineering import engineer_brfss_features
    X_val = engineer_brfss_features(X_val)
    X_test = engineer_brfss_features(X_test)

    model_name = config["mlflow"]["registry_model_name"]
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    except Exception:
        print("[WARN] Could not load production model, using latest run")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`mlflow.parentRunId` IS NULL",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )
        if runs:
            model = mlflow.pyfunc.load_model(f"runs:/{runs[0].info.run_id}/model")
        else:
            print("[ERROR] No model found")
            return

    ref_data = X_val.sample(n=min(5000, len(X_val)), random_state=42)

    n_batches = config["monitoring"]["n_batches"]
    batch_size = min(config["monitoring"]["batch_size"], len(X_test) // n_batches)

    print(f"\n{'='*60}")
    print(f"MONITORING: {n_batches} batches, {batch_size:,} samples each")
    print(f"Drift threshold: {config['monitoring']['drift_threshold']*100:.0f}%")
    print(f"{'='*60}")

    os.makedirs("artifacts/monitoring", exist_ok=True)

    with mlflow.start_run(run_name="Monitoring_Pipeline") as parent_run:
        mlflow.log_param("n_batches", n_batches)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("drift_threshold", config["monitoring"]["drift_threshold"])
        mlflow.log_param("reference_samples", len(ref_data))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("drift_test", "Kolmogorov-Smirnov (KS)")
        mlflow.set_tag("stage", "monitoring")
        mlflow.set_tag("model_name", model_name)

        batch_metrics = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_test.iloc[start_idx:end_idx].copy()
            y_batch = y_test.iloc[start_idx:end_idx].copy()

            is_drift_batch = (i == n_batches - 1)
            if is_drift_batch:
                X_batch = inject_drift(X_batch)

            with mlflow.start_run(run_name=f"batch_{i+1}", nested=True):
                y_pred = model.predict(X_batch)
                y_pred = np.array(y_pred).flatten()

                f1 = f1_score(y_batch, y_pred)
                acc = accuracy_score(y_batch, y_pred)
                prec = precision_score(y_batch, y_pred, zero_division=0)
                rec = recall_score(y_batch, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_batch, y_pred)

                drift_result = ks_drift_test(ref_data, X_batch)
                drift_pct = drift_result["drift_percentage"]
                has_drift = drift_result["dataset_drift"]
                n_drifted = drift_result["n_features_drifted"]
                total_feats = drift_result["total_features"]

                drift_flag = "[!] DRIFT DETECTED" if has_drift else "[OK] No drift"

                # Log all metrics
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("mcc", mcc)
                mlflow.log_metric("drift_percentage", drift_pct)
                mlflow.log_metric("n_features_drifted", n_drifted)
                mlflow.log_metric("total_features", total_feats)
                mlflow.log_metric("dataset_drift", int(has_drift))
                mlflow.log_metric("batch_number", i + 1)
                mlflow.log_param("injected_drift", str(is_drift_batch))
                mlflow.set_tag("stage", "monitoring_batch")
                mlflow.set_tag("drift_status", "DRIFTED" if has_drift else "STABLE")

                # Save per-feature drift details as artifact
                drift_detail_path = f"artifacts/monitoring/batch_{i+1}_drift_details.json"
                with open(drift_detail_path, "w") as f:
                    json.dump({
                        "batch": i + 1,
                        "drift_percentage": drift_pct,
                        "dataset_drift": has_drift,
                        "n_features_drifted": n_drifted,
                        "total_features": total_feats,
                        "injected_drift": is_drift_batch,
                        "feature_results": drift_result["feature_results"]
                    }, f, indent=2)
                mlflow.log_artifact(drift_detail_path)

                batch_metrics.append({
                    "batch": i + 1, "f1": f1, "accuracy": acc,
                    "precision": prec, "recall": rec, "mcc": mcc,
                    "drift_pct": drift_pct, "drift": has_drift,
                    "n_drifted": n_drifted, "injected": is_drift_batch
                })

                print(f"  Batch {i+1}/{n_batches}: F1={f1:.4f}, Acc={acc:.4f}, "
                      f"Drift={drift_pct*100:.1f}% ({n_drifted}/{total_feats} feats), "
                      f"{drift_flag}")

        # ── Generate Comprehensive Monitoring Dashboard ──
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        batches = [m["batch"] for m in batch_metrics]
        f1s = [m["f1"] for m in batch_metrics]
        accs = [m["accuracy"] for m in batch_metrics]
        drifts = [m["drift_pct"] * 100 for m in batch_metrics]
        mccs = [m["mcc"] for m in batch_metrics]

        colors = ["#F44336" if m["drift"] else "#4CAF50" for m in batch_metrics]

        # F1 per batch
        axes[0, 0].bar(batches, f1s, color=colors, alpha=0.8, edgecolor="white")
        axes[0, 0].set_xlabel("Batch", fontsize=11)
        axes[0, 0].set_ylabel("F1 Score", fontsize=11)
        axes[0, 0].set_title("F1 Score per Batch", fontsize=13, fontweight="bold")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, axis="y", alpha=0.3)
        for b, v in zip(batches, f1s):
            axes[0, 0].text(b, v + 0.02, f"{v:.3f}", ha="center", fontsize=10,
                           fontweight="bold")

        # Drift percentage
        axes[0, 1].bar(batches, drifts, color=colors, alpha=0.8, edgecolor="white")
        axes[0, 1].axhline(y=25, color="red", linestyle="--", linewidth=2,
                           label="Drift Threshold (25%)")
        axes[0, 1].set_xlabel("Batch", fontsize=11)
        axes[0, 1].set_ylabel("Features Drifted (%)", fontsize=11)
        axes[0, 1].set_title("Data Drift Detection", fontsize=13, fontweight="bold")
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, axis="y", alpha=0.3)
        for b, v in zip(batches, drifts):
            axes[0, 1].text(b, v + 1, f"{v:.1f}%", ha="center", fontsize=10,
                           fontweight="bold")

        # Accuracy per batch
        axes[1, 0].bar(batches, accs, color=colors, alpha=0.8, edgecolor="white")
        axes[1, 0].set_xlabel("Batch", fontsize=11)
        axes[1, 0].set_ylabel("Accuracy", fontsize=11)
        axes[1, 0].set_title("Accuracy per Batch", fontsize=13, fontweight="bold")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, axis="y", alpha=0.3)
        for b, v in zip(batches, accs):
            axes[1, 0].text(b, v + 0.02, f"{v:.3f}", ha="center", fontsize=10,
                           fontweight="bold")

        # MCC per batch
        axes[1, 1].bar(batches, mccs, color=colors, alpha=0.8, edgecolor="white")
        axes[1, 1].set_xlabel("Batch", fontsize=11)
        axes[1, 1].set_ylabel("MCC", fontsize=11)
        axes[1, 1].set_title("Matthews Correlation Coefficient per Batch",
                             fontsize=13, fontweight="bold")
        axes[1, 1].grid(True, axis="y", alpha=0.3)
        for b, v in zip(batches, mccs):
            axes[1, 1].text(b, v + 0.01, f"{v:.3f}", ha="center", fontsize=10,
                           fontweight="bold")

        plt.suptitle("Model Monitoring Dashboard — Drift & Performance",
                     fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plot_path = "artifacts/monitoring/drift_report.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(plot_path)

        # ── Monitoring Summary Report ──
        summary_path = "artifacts/monitoring/monitoring_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{'='*70}\n")
            f.write(f"MONITORING SUMMARY REPORT\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Batches: {n_batches}\n")
            f.write(f"Batch Size: {batch_size:,}\n")
            f.write(f"Drift Threshold: {config['monitoring']['drift_threshold']*100:.0f}%\n\n")
            f.write(f"{'Batch':>6} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
                    f"{'MCC':>8} {'Drift%':>8} {'Status':>12}\n")
            f.write("-" * 72 + "\n")
            for m in batch_metrics:
                status = "!! DRIFT" if m["drift"] else "OK"
                f.write(f"{m['batch']:>6} {m['f1']:>8.4f} {m['accuracy']:>8.4f} "
                        f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                        f"{m['mcc']:>8.4f} {m['drift_pct']*100:>7.1f}% "
                        f"{status:>12}\n")
            drift_batches = sum(1 for m in batch_metrics if m["drift"])
            f.write(f"\nDrift detected in {drift_batches}/{n_batches} batches\n")
        mlflow.log_artifact(summary_path)

        # Save batch metrics CSV
        metrics_df = pd.DataFrame(batch_metrics)
        csv_path = "artifacts/monitoring/batch_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # Log summary metrics at the parent level
        avg_f1 = np.mean([m["f1"] for m in batch_metrics if not m["drift"]])
        avg_acc = np.mean([m["accuracy"] for m in batch_metrics if not m["drift"]])
        mlflow.log_metric("avg_f1_stable", avg_f1)
        mlflow.log_metric("avg_accuracy_stable", avg_acc)
        mlflow.log_metric("n_drift_batches",
                         sum(1 for m in batch_metrics if m["drift"]))

        print(f"\n[OK] Monitoring complete! Report saved to {plot_path}")


if __name__ == "__main__":
    main()
