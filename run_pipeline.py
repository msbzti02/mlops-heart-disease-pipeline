import argparse
import subprocess
import sys
import time

STAGES = {
    "preprocess": {
        "script": "src/data_preprocessing.py",
        "description": "Data Preprocessing (merge 6 datasets, clean, scale, split, SMOTE)",
    },
    "train": {
        "script": "src/train.py",
        "description": "Model Training (5 models with MLflow tracking)",
    },
    "tune": {
        "script": "src/tune.py",
        "description": "Hyperparameter Tuning (Hyperopt + nested MLflow runs)",
    },
    "register": {
        "script": "src/register_model.py",
        "description": "Model Registry (None -> Staging -> Production)",
    },
    "monitor": {
        "script": "src/monitor.py",
        "description": "Performance Monitoring (drift detection on 5 batches)",
    },
    "explain": {
        "script": "src/explainability.py",
        "description": "Explainability (SHAP + LIME analysis)",
    },
}


def run_stage(name, info):
    print(f"\n{'='*60}")
    print(f"STAGE: {info['description']}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, info["script"]],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [FAIL] {name} failed (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument(
        "--stage", type=str, default=None,
        choices=list(STAGES.keys()),
        help="Run a specific stage (default: run all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MLOps Pipeline - Heart Disease Prediction")
    print("=" * 60)

    if args.stage:
        stages_to_run = {args.stage: STAGES[args.stage]}
    else:
        stages_to_run = STAGES

    results = {}
    total_start = time.time()

    for name, info in stages_to_run.items():
        success = run_stage(name, info)
        results[name] = success
        if not success and args.stage is None:
            print(f"\n[ABORT] Pipeline stopped at '{name}'")
            break

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    passed = sum(results.values())
    total = len(results)
    print(f"  Result: {passed}/{total} stages passed")


if __name__ == "__main__":
    main()
