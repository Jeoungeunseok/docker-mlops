from app.domains.mlops.registry import trainer_registry
from app.domains.mlops.schemas import ModelTrainer, TrainingContext, TrainingResult
from app.domains.mlops.training_pipeline import run_training_pipeline


def train_model_job(context: TrainingContext, trainer: ModelTrainer | None = None) -> TrainingResult:
    selected_trainer = trainer or trainer_registry.get(context.model_type)
    return run_training_pipeline(context=context, trainer=selected_trainer)
