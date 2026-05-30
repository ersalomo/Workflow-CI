"""
modelling.py - MLProject Entry Point for CI/CD Training Pipeline
Digunakan oleh: mlflow run Workflow-CI/MLProject -P n_estimators=120 -P max_depth=6
Kriteria 3: Workflow CI dengan MLflow Projects
- Menerima hyperparameters via argparse (dari MLProject entry_points)
- Manual MLflow logging (bukan autolog)
- Log parameters, metrics, artifacts ke DagsHub Tracking Server
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
import os


def train_model(n_estimators: int, max_depth: int):
    """Train RandomForest dengan hyperparameters dari CLI dan log ke MLflow."""
    print(f"[MLProject] Training with n_estimators={n_estimators}, max_depth={max_depth}")

    # Load preprocessed data (dari direktori yang sama dengan MLProject)
    data_path = os.path.join(os.path.dirname(__file__), "wine_preprocessed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    X  = df.drop(columns=['quality'])
    y  = df['quality']

    # Determine averaging method based on number of classes
    n_classes  = len(y.unique())
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    print(f"Detected {n_classes} classes → average='{avg_method}'")

    # Split train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow run (tracking URI di-set oleh environment variable MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=f"CI_RF_n{n_estimators}_d{max_depth}"):

        # --- Log Parameters ---
        mlflow.log_param("n_estimators",   n_estimators)
        mlflow.log_param("max_depth",      max_depth)
        mlflow.log_param("test_size",      0.2)
        mlflow.log_param("random_state",   42)
        mlflow.log_param("avg_method",     avg_method)
        mlflow.log_param("train_samples",  X_train.shape[0])
        mlflow.log_param("test_samples",   X_test.shape[0])
        mlflow.log_param("n_features",     X_train.shape[1])

        # --- Train Model ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        # --- Log Metrics ---
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        print(f"  accuracy : {accuracy:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall   : {recall:.4f}")
        print(f"  f1_score : {f1:.4f}")

        # --- Artifact: Confusion Matrix ---
        os.makedirs("artifacts", exist_ok=True)
        cm   = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix (n={n_estimators}, depth={max_depth})\nAcc={accuracy:.3f}, F1={f1:.3f}")
        plt.tight_layout()
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # --- Artifact: Feature Importance ---
        importances   = model.feature_importances_
        indices       = np.argsort(importances)[::-1]
        feature_names = X.columns.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(feature_names)), importances[indices], color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title(f"Feature Importance (n_estimators={n_estimators})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # --- Log Model with Signature ---
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model, "model",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLProject] Training complete. Run ID: {run_id}")
        print("[MLProject] Model and artifacts logged to MLflow tracking server.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLProject Wine Quality Training")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the RandomForest")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Maximum depth of each tree")
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)
