import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from mlflow.models import infer_signature

import config
from train import train

# Set Tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

def run_mlflow():
    mlflow.set_experiment("Credit_Risk_Reproducibility")

    with mlflow.start_run() as run:
        # ---------- 1. Train and Prediction ----------
        # train() returns pipeline, X_test, y_test
        pipeline, X_test, y_test = train()
        y_pred = pipeline.predict(X_test)

        # ---------- 2. Log Standard Metrics ----------
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # ---------- 3. Fairness Audit ----------
        try:
            raw_df = config.load_data()
            # Align indices to get gender for the test set only
            genders = raw_df.loc[X_test.index, "Sex"]
            
            for gender in genders.unique():
                mask = (genders == gender)
                if mask.any():
                    group_acc = accuracy_score(y_test[mask], y_pred[mask])
                    # Selection Rate: How often model predicts 'Good' (1)
                    selection_rate = np.mean(y_pred[mask])
                    
                    mlflow.log_metric(f"accuracy_{gender.lower()}", group_acc)
                    mlflow.log_metric(f"selection_rate_{gender.lower()}", selection_rate)
            
            mlflow.set_tag("fairness_audit", "success")
        except Exception as e:
            print(f"Fairness audit skipped: {e}")
            mlflow.set_tag("fairness_audit", "failed")

        # ---------- 4. Enhanced Parameters ----------
        model = pipeline.named_steps["model"]
        params = model.get_params()
        params["with_gender"] = config.WITH_GENDER
        params["data_source"] = config.DATA_PATH
        mlflow.log_params(params)

        # ---------- 5. Log Model with Signature ----------
        # Signatures define the schema for API reproducibility
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        # ---------- 6 Log Artifacts ----------
        os.makedirs(config.VIZ_DIR, exist_ok=True)
        mlflow.log_artifact(config.MODEL_PATH)
        mlflow.log_artifact("config.py")

        print(f"Successfully logged run: {run.info.run_id}")

if __name__ == "__main__":
    run_mlflow()