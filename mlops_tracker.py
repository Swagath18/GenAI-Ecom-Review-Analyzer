# import mlflow
# import mlflow.pytorch

# class MLOpsTracker:
#     def __init__(self, experiment_name="walmart_review_analysis"):
#         mlflow.set_experiment(experiment_name)

#     def log_model_performance(self, metrics):
#         with mlflow.start_run():
#             for key, value in metrics.items():
#                 mlflow.log_metric(key, value)

import mlflow
import mlflow.pytorch
import logging

class MLOpsTracker:
    def __init__(self, experiment_name="walmart_review_analysis"):
        mlflow.set_experiment(experiment_name)
        # Set up a logger for error tracking
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)  # Can be adjusted to other levels (e.g., INFO, DEBUG)

    def log_model_performance(self, metrics):
        """
        Logs the model performance metrics to MLflow.
        """
        with mlflow.start_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

    def log_error(self, error_message):
        """
        Logs an error message to the logger and optionally to MLflow.
        """
        self.logger.error(f"Error logged: {error_message}")
        # Optionally log this error message in MLflow as a parameter
        with mlflow.start_run():
            mlflow.log_param("error_message", error_message)
