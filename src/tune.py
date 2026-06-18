import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import warnings
import json

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    log_loss, matthews_corrcoef, roc_curve
)
import yaml

warnings.filterwarnings("ignore")

CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
print(f"[GPU] Tuning on: {GPU_NAME}")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    proc_dir = config["data"]["processed_dir"]
    prefix = "brfss"

    X_train = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_train.csv"))
    X_val = pd.read_csv(os.path.join(proc_dir, f"{prefix}_X_val.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, f"{prefix}_y_train.csv")).squeeze()
    y_val = pd.read_csv(os.path.join(proc_dir, f"{prefix}_y_val.csv")).squeeze()

    from src.feature_engineering import engineer_brfss_features
    X_train = engineer_brfss_features(X_train)
    X_val = engineer_brfss_features(X_val)

    n_trials = config["tuning"]["n_trials"]
    rs = config["data"]["random_state"]

    sample_size = min(100000, len(X_train))
    idx = np.random.RandomState(rs).choice(len(X_train), sample_size, replace=False)
    X_tune = X_train.iloc[idx]
    y_tune = y_train.iloc[idx]

    print(f"\n{'='*60}")
    print(f"Hyperopt Tuning: {n_trials} trials on {sample_size:,} samples (GPU)")
    print(f"{'='*60}")

    search_space = {
        "n_estimators": hp.choice("n_estimators", [100, 150, 200, 300, 400, 500]),
        "max_depth": hp.quniform("max_depth", 3, 12, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": hp.quniform("min_child_weight", 1, 20, 1),
        "gamma": hp.uniform("gamma", 0, 5),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10)),
    }

    trial_results = []

    with mlflow.start_run(run_name="Hyperopt_XGBoost_GPU_Tuning") as parent:
        mlflow.log_param("gpu", GPU_NAME)
        mlflow.log_param("tuning_samples", sample_size)
        mlflow.log_param("total_trials", n_trials)
        mlflow.log_param("full_train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("optimizer", "TPE (Tree-structured Parzen Estimator)")
        mlflow.log_param("objective", "maximize AUC")
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("model_family", "XGBoost")
        mlflow.set_tag("compute_device", "GPU")

        # Log the search space as an artifact
        os.makedirs("artifacts/tuning", exist_ok=True)
        search_space_info = {
            "n_estimators": [100, 150, 200, 300, 400, 500],
            "max_depth": "quniform(3, 12, 1)",
            "learning_rate": "loguniform(0.005, 0.3)",
            "subsample": "uniform(0.5, 1.0)",
            "colsample_bytree": "uniform(0.5, 1.0)",
            "min_child_weight": "quniform(1, 20, 1)",
            "gamma": "uniform(0, 5)",
            "reg_alpha": "loguniform(1e-8, 10)",
            "reg_lambda": "loguniform(1e-8, 10)",
        }
        with open("artifacts/tuning/search_space.json", "w") as f:
            json.dump(search_space_info, f, indent=2)
        mlflow.log_artifact("artifacts/tuning/search_space.json")

        def objective(params):
            params["max_depth"] = int(params["max_depth"])
            params["min_child_weight"] = int(params["min_child_weight"])

            with mlflow.start_run(
                run_name=f"trial_{len(trial_results)+1}",
                nested=True
            ):
                model = XGBClassifier(
                    **params,
                    device="cuda",
                    tree_method="hist",
                    random_state=rs,
                    eval_metric="logloss",
                    verbosity=0,
                )

                start = time.time()
                model.fit(X_tune, y_tune)
                elapsed = time.time() - start

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_proba)
                f1 = f1_score(y_val, y_pred)
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred)
                rec = recall_score(y_val, y_pred)
                mcc = matthews_corrcoef(y_val, y_pred)
                ll = log_loss(y_val, y_proba)

                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("device", "cuda")
                mlflow.set_tag("stage", "tuning_trial")
                mlflow.set_tag("trial_number", str(len(trial_results) + 1))
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("mcc", mcc)
                mlflow.log_metric("log_loss", ll)
                mlflow.log_metric("training_time", elapsed)

                trial_results.append({
                    "trial": len(trial_results) + 1,
                    "auc": auc, "f1": f1, "accuracy": acc,
                    "precision": prec, "recall": rec,
                    "mcc": mcc, "log_loss": ll,
                    "time": elapsed,
                    **params
                })

                print(f"  Trial {len(trial_results):>3}/{n_trials}: "
                      f"AUC={auc:.4f}, F1={f1:.4f}, MCC={mcc:.4f} ({elapsed:.1f}s)")

            return {"loss": 1 - auc, "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            show_progressbar=False,
        )

        n_est_choices = [100, 150, 200, 300, 400, 500]
        best_params = {
            "n_estimators": n_est_choices[best["n_estimators"]],
            "max_depth": int(best["max_depth"]),
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": int(best["min_child_weight"]),
            "gamma": best["gamma"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
        }

        print(f"\n{'='*60}")
        print("BEST HYPERPARAMETERS:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
            mlflow.log_param(f"best_{k}", v)

        # --- Train Final Model on Full Data ---
        print(f"\nTraining final model on {len(X_train):,} rows (GPU)...")
        final_model = XGBClassifier(
            **best_params,
            device="cuda",
            tree_method="hist",
            random_state=rs,
            eval_metric="logloss",
            verbosity=0,
        )
        start_final = time.time()
        final_model.fit(X_train, y_train)
        final_time = time.time() - start_final

        y_pred = final_model.predict(X_val)
        y_proba = final_model.predict_proba(X_val)[:, 1]
        final_auc = roc_auc_score(y_val, y_proba)
        final_f1 = f1_score(y_val, y_pred)
        final_acc = accuracy_score(y_val, y_pred)
        final_prec = precision_score(y_val, y_pred)
        final_rec = recall_score(y_val, y_pred)
        final_mcc = matthews_corrcoef(y_val, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        mlflow.log_metric("best_auc", final_auc)
        mlflow.log_metric("best_f1", final_f1)
        mlflow.log_metric("best_accuracy", final_acc)
        mlflow.log_metric("best_precision", final_prec)
        mlflow.log_metric("best_recall", final_rec)
        mlflow.log_metric("best_mcc", final_mcc)
        mlflow.log_metric("best_training_time", final_time)
        mlflow.log_metric("best_true_positives", int(tp))
        mlflow.log_metric("best_true_negatives", int(tn))
        mlflow.log_metric("best_false_positives", int(fp))
        mlflow.log_metric("best_false_negatives", int(fn))
        mlflow.xgboost.log_model(final_model, "best_model")

        print(f"Final Model: AUC={final_auc:.4f}, F1={final_f1:.4f}, "
              f"MCC={final_mcc:.4f}")

        # --- Classification Report ---
        report_text = classification_report(
            y_val, y_pred, target_names=["No Disease", "Disease"]
        )
        report_path = "artifacts/tuning/best_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Best Tuned Model — Classification Report\n")
            f.write(f"{'='*50}\n")
            f.write(report_text)
            f.write(f"\nAUC: {final_auc:.4f}\n")
            f.write(f"MCC: {final_mcc:.4f}\n")
            f.write(f"Training Time: {final_time:.2f}s\n")
        mlflow.log_artifact(report_path)

        # --- Convergence Plot ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        aucs = [r["auc"] for r in trial_results]
        best_so_far = [max(aucs[:i+1]) for i in range(len(aucs))]
        axes[0, 0].plot(range(1, len(aucs)+1), aucs, "o-", alpha=0.5,
                        label="Trial AUC", color="#2196F3", markersize=4)
        axes[0, 0].plot(range(1, len(aucs)+1), best_so_far, "r-", lw=2,
                        label="Best so far")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("AUC")
        axes[0, 0].set_title(f"Hyperopt Convergence ({GPU_NAME})",
                             fontsize=12, fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Training time per trial
        times = [r["time"] for r in trial_results]
        axes[0, 1].bar(range(1, len(times)+1), times, alpha=0.7, color="#FF9800")
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Training Time (s)")
        axes[0, 1].set_title("GPU Training Time per Trial",
                             fontsize=12, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # AUC vs F1 scatter
        f1s = [r["f1"] for r in trial_results]
        scatter = axes[1, 0].scatter(aucs, f1s,
                                      c=range(len(aucs)), cmap="viridis",
                                      alpha=0.7, s=50, edgecolors="white")
        axes[1, 0].set_xlabel("AUC")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("AUC vs F1 Score (colored by trial #)",
                             fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label="Trial #")

        # Learning rate vs AUC
        lrs = [r["learning_rate"] for r in trial_results]
        axes[1, 1].scatter(lrs, aucs, alpha=0.7, color="#4CAF50", s=50,
                          edgecolors="white")
        axes[1, 1].set_xlabel("Learning Rate")
        axes[1, 1].set_ylabel("AUC")
        axes[1, 1].set_title("Learning Rate vs AUC",
                             fontsize=12, fontweight="bold")
        axes[1, 1].set_xscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        conv_path = "artifacts/tuning/convergence.png"
        plt.savefig(conv_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(conv_path)

        # --- ROC Curve for Best Model ---
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, color="#2196F3",
                label=f"Tuned XGBoost (AUC={final_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title("ROC Curve — Best Tuned Model", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = "artifacts/tuning/best_roc_curve.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(roc_path)

        # --- Confusion Matrix for Best Model ---
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_title("Confusion Matrix — Best Tuned Model",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        cm_path = "artifacts/tuning/best_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        # --- Feature Importance ---
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = final_model.feature_importances_
        feat_names = X_train.columns.tolist()
        top_n = min(20, len(importances))
        indices = np.argsort(importances)[-top_n:]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), importances[indices], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in indices], fontsize=9)
        ax.set_title("Best Tuned Model — Top 20 Features",
                     fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fi_path = "artifacts/tuning/feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(fi_path)

        # --- Save All Trial Results ---
        results_df = pd.DataFrame(trial_results)
        results_path = "artifacts/tuning/trial_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # --- Best Params JSON ---
        best_params_path = "artifacts/tuning/best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(best_params_path)

        print(f"\n[OK] Hyperopt GPU tuning complete! Best AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
