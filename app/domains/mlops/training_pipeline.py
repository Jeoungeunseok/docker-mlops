from typing import Any

from app.core.logging import app_logger
from app.domains.mlops import mlflow_client, model_registry
from app.domains.mlops.schemas import ModelTrainer, TrainingContext, TrainingResult


def run_training_pipeline(
    context: TrainingContext,
    trainer: ModelTrainer,
    run_name: str | None = None,
) -> TrainingResult:
    model_name = model_registry.build_model_name(
        model_type=context.model_type,
        target_type=context.target_type,
        target_id=context.target_id,
        qualifiers=context.qualifiers,
    )
    app_logger.info(
        "Starting training pipeline",
        extra={"model_name": model_name, "model_type": context.model_type},
    )

    with mlflow_client.start_run(run_name=run_name or model_name) as run:
        model = trainer.train_model(context)
        metrics = trainer.evaluate_model(model, context)
        app_logger.info(
            "Training evaluation completed",
            extra={
                "model_name": model_name,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "validation_samples": metrics.validation_samples,
            },
        )

        mlflow_client.log_params(_context_params(context))
        mlflow_client.log_metrics(metrics.as_mlflow_metrics())
        mlflow_client.log_tags(
            {
                "model_type": context.model_type,
                "model_name": model_name,
            }
        )

        model_uri = trainer.log_model(model, artifact_path="model")
        registered_model = model_registry.register_model(model_uri=model_uri, model_name=model_name)
        app_logger.info(
            "Training model artifact registered",
            extra={
                "model_name": model_name,
                "model_uri": model_uri,
                "model_version": registered_model.version,
                "run_id": run.info.run_id,
            },
        )
        model_registry.set_candidate(model_name, registered_model.version)

        promoted = model_registry.should_promote(model_name, metrics)
        if promoted:
            model_registry.promote_to_champion(model_name, registered_model.version)
        app_logger.info(
            "Training pipeline completed",
            extra={"model_name": model_name, "run_id": run.info.run_id, "promoted": promoted},
        )

        return TrainingResult(
            model_name=model_name,
            run_id=run.info.run_id,
            model_uri=model_uri,
            metrics=metrics,
            registered_model=registered_model,
            promoted=promoted,
        )


def _context_params(context: TrainingContext) -> dict[str, Any]:
    data = context.model_dump()
    extra_params = data.pop("extra_params", {})
    return {**data, **extra_params}
