from app.domains.mlops.schemas import ModelTrainer, TrainingContext, TrainingResult
from app.domains.mlops.training_pipeline import run_training_pipeline


def train_model_job(context: TrainingContext, trainer: ModelTrainer) -> TrainingResult:
    return run_training_pipeline(context=context, trainer=trainer)
