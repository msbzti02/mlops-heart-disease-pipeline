import os
import sys
import time
import mlflow
from mlflow.tracking import MlflowClient
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_best_run(client, experiment_name, metric="auc"):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"  [ERROR] Experiment '{experiment_name}' not found")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.`mlflow.parentRunId` IS NULL",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        print("  [ERROR] No runs found")
        return None

    best = runs[0]
    print(f"  Best run: {best.info.run_id}")
    print(f"  Model: {best.data.params.get('model_type', 'unknown')}")
    print(f"  AUC: {best.data.metrics.get('auc', 0):.4f}")
    print(f"  F1:  {best.data.metrics.get('f1_score', 0):.4f}")
    return best


def register_model(client, run, model_name):
    model_uri = f"runs:/{run.info.run_id}/model"

    try:
        tuned_uri = f"runs:/{run.info.run_id}/best_model"
        mlflow.pyfunc.load_model(tuned_uri)
        model_uri = tuned_uri
        print("  Using tuned best_model artifact")
    except Exception:
        pass

    result = mlflow.register_model(model_uri, model_name)
    print(f"  [OK] Registered '{model_name}' version {result.version}")
    return result


def transition_stages(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("  [ERROR] No model versions found")
        return

    latest = max(versions, key=lambda v: int(v.version))
    version = latest.version

    client.update_model_version(
        name=model_name,
        version=version,
        description="Heart Disease Prediction model trained on BRFSS + Clinical data"
    )
    client.set_model_version_tag(model_name, version, "dataset", "brfss_combined")
    client.set_model_version_tag(model_name, version, "task", "binary_classification")

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging"
    )
    print(f"  [OK] v{version} -> Staging")
    time.sleep(1)

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True
    )
    print(f"  [OK] v{version} -> Production")

    print(f"\n  Registry Summary for '{model_name}':")
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"    v{v.version}: stage={v.current_stage}, status={v.status}")


def main():
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    exp_name = config["mlflow"]["experiment_name"]
    model_name = config["mlflow"]["registry_model_name"]

    print(f"\n{'='*60}")
    print("MODEL REGISTRY OPERATIONS")
    print(f"{'='*60}")

    print("\n[1] Finding best run...")
    best_run = get_best_run(client, exp_name, metric="auc")
    if best_run is None:
        return

    print("\n[2] Registering model...")
    register_model(client, best_run, model_name)

    print("\n[3] Transitioning stages...")
    transition_stages(client, model_name)

    print(f"\n[OK] Model Registry operations complete!")


if __name__ == "__main__":
    main()
