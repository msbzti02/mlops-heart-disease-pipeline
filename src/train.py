import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import warnings
import json

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
    precision_recall_curve, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
import yaml

warnings.filterwarnings("ignore")

CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
print(f"[GPU] Device: {GPU_NAME} | CUDA: {CUDA_AVAILABLE}")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "log_loss": log_loss(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "avg_precision": average_precision_score(y_true, y_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    axes[0].set_title(f"Confusion Matrix (Counts) - {model_name}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # Percentages
    labels = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Oranges", ax=axes[1],
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    axes[1].set_title(f"Confusion Matrix (%) - {model_name}", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_threshold = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", linewidth=2, color="#2196F3")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color="red", s=100, zorder=5,
               label=f"Optimal Threshold = {opt_threshold:.3f}")
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, model_name, save_path):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, linewidth=2, color="#FF5722",
            label=f"AP = {ap:.4f}")
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances, feature_names, model_name, save_path):
    if importances is None:
        return
    top_n = min(20, len(importances))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), importances[indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_roc(all_results, y_val, save_path):
    """Generate a combined ROC curve comparison plot for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (name, data) in enumerate(all_results.items()):
        fpr, tpr, _ = roc_curve(y_val, data["y_proba"])
        auc_val = data["metrics"]["auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})",
                linewidth=2, color=colors[i % len(colors)])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(all_results, save_path):
    """Generate a grouped bar chart comparing metrics across all models."""
    metrics_to_plot = ["accuracy", "f1_score", "precision", "recall", "auc", "mcc"]
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_metrics)
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, name in enumerate(model_names):
        values = [all_results[name]["metrics"].get(m, 0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison — Key Metrics", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchClassifierWrapper:
    def __init__(self, input_dim, epochs=30, batch_size=4096, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = HeartDiseaseNet(input_dim).to(DEVICE)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_t = torch.FloatTensor(np.array(X)).to(DEVICE)
        y_t = torch.FloatTensor(np.array(y)).to(DEVICE).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs} loss={avg_loss:.4f}")
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X)).to(DEVICE)
            probas = []
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i:i+self.batch_size]
                out = self.model(batch).cpu().numpy()
                probas.append(out)
            pos_proba = np.concatenate(probas).flatten()
        return np.column_stack([1 - pos_proba, pos_proba])

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "device": str(DEVICE),
            "gpu": GPU_NAME,
        }


def train_and_log_model(model, model_name, X_train, y_train,
                        X_val, y_val, feature_names, config,
                        is_xgboost=False, is_pytorch=False):
    print(f"\n  Training {model_name}...")
    start = time.time()

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics["training_time_seconds"] = elapsed

        # --- Log Parameters ---
        params = model.get_params()
        for k, v in params.items():
            try:
                mlflow.log_param(k, str(v)[:250])
            except Exception:
                pass
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_val_samples", X_val.shape[0])
        mlflow.log_param("gpu_used", str(CUDA_AVAILABLE))
        mlflow.log_param("gpu_name", GPU_NAME)

        # --- Log Tags ---
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("stage", "baseline_training")
        mlflow.set_tag("dataset", "brfss_combined")
        accel = "GPU" if (is_xgboost or is_pytorch) else "CPU"
        mlflow.set_tag("compute_device", accel)

        # --- Log ALL Metrics ---
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # --- Artifacts Directory ---
        artifacts_dir = f"artifacts/{model_name}"
        os.makedirs(artifacts_dir, exist_ok=True)

        # --- Confusion Matrix (counts + percentages) ---
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_val, y_pred, model_name, cm_path)
        mlflow.log_artifact(cm_path)

        # --- ROC Curve ---
        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        plot_roc_curve(y_val, y_proba, model_name, roc_path)
        mlflow.log_artifact(roc_path)

        # --- Precision-Recall Curve ---
        pr_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(y_val, y_proba, model_name, pr_path)
        mlflow.log_artifact(pr_path)

        # --- Feature Importance ---
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        fi_path = os.path.join(artifacts_dir, "feature_importance.png")
        plot_feature_importance(importances, feature_names, model_name, fi_path)
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path)

        # --- Classification Report (text artifact) ---
        report_text = classification_report(
            y_val, y_pred, target_names=["No Disease", "Disease"]
        )
        report_path = os.path.join(artifacts_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report — {model_name}\n")
            f.write(f"{'='*50}\n")
            f.write(report_text)
            f.write(f"\n{'='*50}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
            f.write(f"MCC: {metrics['mcc']:.4f}\n")
            f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
            f.write(f"Log Loss: {metrics['log_loss']:.4f}\n")
            f.write(f"Avg Precision: {metrics['avg_precision']:.4f}\n")
            f.write(f"Training Time: {elapsed:.2f}s\n")
            f.write(f"Compute: {accel} ({GPU_NAME})\n")
        mlflow.log_artifact(report_path)

        # --- Metrics Summary JSON (for programmatic access) ---
        summary_path = os.path.join(artifacts_dir, "metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(summary_path)

        # --- Log Model ---
        if is_xgboost:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"  [OK] {model_name} ({accel}): F1={metrics['f1_score']:.4f}, "
              f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, "
              f"MCC={metrics['mcc']:.4f} ({elapsed:.1f}s)")

    return model, metrics, y_proba


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

    feature_names = X_train.columns.tolist()
    n_features = len(feature_names)
    rs = config["data"]["random_state"]

    print(f"\n{'='*60}")
    print(f"GPU-ACCELERATED TRAINING on {GPU_NAME}")
    print(f"Training: {len(X_train):,} samples, {n_features} features")
    print(f"Validation: {len(X_val):,} samples")
    print(f"{'='*60}")

    svm_sample_size = min(30000, len(X_train))
    svm_idx = np.random.RandomState(rs).choice(len(X_train), svm_sample_size, replace=False)
    X_train_svm = X_train.iloc[svm_idx]
    y_train_svm = y_train.iloc[svm_idx]
    print(f"  SVM subsample: {svm_sample_size:,} rows (SVM is O(n²))")

    models = [
        ("GradientBoosting_XGB_GPU", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            device="cuda", tree_method="hist", random_state=rs,
            eval_metric="logloss", verbosity=0,
        ), X_train, y_train, True, False),

        ("SVM", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, random_state=rs,
        ), X_train_svm, y_train_svm, False, False),

        ("KNN", KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="minkowski",
        ), X_train, y_train, False, False),

        ("DecisionTree", DecisionTreeClassifier(
            max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=rs,
        ), X_train, y_train, False, False),

        ("NeuralNetwork_PyTorch_GPU", PyTorchClassifierWrapper(
            input_dim=n_features, epochs=30,
            batch_size=4096, lr=0.001,
        ), X_train, y_train, False, True),
    ]

    all_results = {}
    for name, model, X_tr, y_tr, is_xgb, is_pt in models:
        _, metrics, y_proba = train_and_log_model(
            model, name, X_tr, y_tr,
            X_val, y_val, feature_names, config,
            is_xgboost=is_xgb, is_pytorch=is_pt,
        )
        all_results[name] = {"metrics": metrics, "y_proba": y_proba}

    # ── Combined Comparison Plots (logged under a summary run) ──
    print(f"\n  Generating comparison plots...")
    os.makedirs("artifacts", exist_ok=True)

    roc_comp_path = "artifacts/roc_curve_comparison_all_models.png"
    plot_combined_roc(all_results, y_val, roc_comp_path)

    metrics_comp_path = "artifacts/metrics_comparison_all_models.png"
    plot_metrics_comparison(all_results, metrics_comp_path)

    # Log the comparison plots under a summary parent run
    with mlflow.start_run(run_name="Training_Summary_Comparison"):
        mlflow.set_tag("stage", "summary")
        mlflow.set_tag("description", "Combined comparison of all 5 models")

        mlflow.log_artifact(roc_comp_path)
        mlflow.log_artifact(metrics_comp_path)

        # Log the best model's metrics at the summary level
        best_model = max(all_results.items(), key=lambda x: x[1]["metrics"]["auc"])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_auc", best_model[1]["metrics"]["auc"])
        mlflow.log_metric("best_f1", best_model[1]["metrics"]["f1_score"])
        mlflow.log_metric("best_accuracy", best_model[1]["metrics"]["accuracy"])

        # Save full comparison table as text artifact
        summary_path = "artifacts/training_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{'='*80}\n")
            f.write(f"TRAINING SUMMARY — All Models (GPU: {GPU_NAME})\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Model':<32} {'F1':>8} {'AUC':>8} {'Acc':>8} "
                    f"{'MCC':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}\n")
            f.write("-" * 100 + "\n")
            for name, data in all_results.items():
                m = data["metrics"]
                f.write(f"{name:<32} {m['f1_score']:>8.4f} {m['auc']:>8.4f} "
                        f"{m['accuracy']:>8.4f} {m['mcc']:>8.4f} "
                        f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                        f"{m['training_time_seconds']:>7.1f}s\n")
            f.write(f"\nBest Model: {best_model[0]} (AUC={best_model[1]['metrics']['auc']:.4f})\n")
        mlflow.log_artifact(summary_path)

    # ── Console Summary ──
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY (GPU: {GPU_NAME})")
    print(f"{'='*60}")
    print(f"{'Model':<32} {'F1':>8} {'AUC':>8} {'Acc':>8} {'MCC':>8} {'Time':>8}")
    print("-" * 80)
    for name, data in all_results.items():
        m = data["metrics"]
        print(f"{name:<32} {m['f1_score']:>8.4f} {m['auc']:>8.4f} "
              f"{m['accuracy']:>8.4f} {m['mcc']:>8.4f} "
              f"{m['training_time_seconds']:>7.1f}s")

    if CUDA_AVAILABLE:
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU Memory: {mem_alloc:.2f} GB allocated, {mem_max:.2f} GB peak")


if __name__ == "__main__":
    main()
