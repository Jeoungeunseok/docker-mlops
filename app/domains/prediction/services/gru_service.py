from typing import Any

import mlflow

from app.domains.mlops.schemas import EvaluationMetrics, TrainingContext


class GruTrainer:
    def train_model(self, context: TrainingContext) -> Any:
        raise NotImplementedError("Connect the project-specific GRU training implementation here.")

    def evaluate_model(self, model: Any, context: TrainingContext) -> EvaluationMetrics:
        raise NotImplementedError("Connect the project-specific GRU evaluation implementation here.")

    def log_model(self, model: Any, artifact_path: str) -> str:
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model)
        return f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
