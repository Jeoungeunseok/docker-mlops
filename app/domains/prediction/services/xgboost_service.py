from typing import Any

import mlflow.xgboost

from app.domains.mlops.schemas import EvaluationMetrics, TrainingContext


class XGBoostTrainer:
    def train_model(self, context: TrainingContext) -> Any:
        raise NotImplementedError("Connect the project-specific XGBoost training implementation here.")

    def evaluate_model(self, model: Any, context: TrainingContext) -> EvaluationMetrics:
        raise NotImplementedError("Connect the project-specific XGBoost evaluation implementation here.")

    def log_model(self, model: Any, artifact_path: str) -> str:
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
        return f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
