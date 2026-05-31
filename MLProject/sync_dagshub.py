import mlflow
import os

try:
    local_client = mlflow.tracking.MlflowClient(tracking_uri="file:./mlruns")
    experiments  = local_client.search_experiments()
    if experiments:
        runs = local_client.search_runs([e.experiment_id for e in experiments], order_by=["start_time DESC"])
        latest_run = None
        for r in runs:
            if "f1_score" in r.data.metrics or "accuracy" in r.data.metrics:
                latest_run = r
                break
        if latest_run is None and runs:
            latest_run = runs[0]
        
        if latest_run:
            params  = latest_run.data.params
            metrics = latest_run.data.metrics
            run_name = latest_run.info.run_name
            
            mlflow.set_experiment("Wine_Quality_CI_Experiment")
            with mlflow.start_run(run_name=f"CI_{run_name}"):
                for k, v in params.items():
                    mlflow.log_param(k, v)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_param("ci_run_number", os.environ.get("GITHUB_RUN_NUMBER", "0"))
                mlflow.log_param("ci_triggered_by", "github-actions")
            print("Run synced to DagsHub successfully!")
except Exception as e:
    print(f"DagsHub sync error: {e}")
