from datetime import datetime

from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.model_loader import model_loader
from app.domains.mlops.registry import prediction_input_validator_registry
from app.domains.mlops.schemas import PredictionLogPayload
from app.domains.prediction.log_store import prediction_log_store
from app.domains.prediction.schemas import PredictionActualUpdateRequest, PredictionRequest, PredictionResponse


def predict(request: PredictionRequest) -> PredictionResponse:
    app_logger.info("Running prediction", extra={"model_name": request.model_name})
    _validate_prediction_inputs(request)
    load_info = model_loader.load(request.model_name)
    predictions = model_loader.predict(request.model_name, request.inputs)
    predicted_at = now_in_app_timezone()
    _save_prediction_log(request, load_info.version, load_info.run_id, predictions, predicted_at)
    app_logger.info(
        "Prediction completed",
        extra={
            "model_name": request.model_name,
            "model_version": load_info.version,
            "run_id": load_info.run_id,
        },
    )
    return PredictionResponse(
        model_name=request.model_name,
        model_version=load_info.version,
        run_id=load_info.run_id,
        request_id=request.request_id,
        predicted_at=predicted_at,
        predictions=predictions,
    )


def update_prediction_actual(request_id: str, request: PredictionActualUpdateRequest) -> None:
    prediction_log_store.update_actual(
        request_id=request_id,
        actual_value=request.actual_value,
        error_value=request.error_value,
        error_metrics=request.error_metrics,
    )


def list_prediction_logs(model_name: str) -> list[PredictionLogPayload]:
    return prediction_log_store.list_by_model(model_name)


def _validate_prediction_inputs(request: PredictionRequest) -> None:
    model_type = request.model_name.split("_", 1)[0]
    try:
        validator = prediction_input_validator_registry.get(model_type)
    except KeyError:
        app_logger.info("Prediction input validator was not registered", extra={"model_type": model_type})
        return
    validator.validate(request.inputs)


def _save_prediction_log(
    request: PredictionRequest,
    model_version: str | None,
    run_id: str | None,
    predictions: object,
    predicted_at: datetime,
) -> None:
    try:
        prediction_log_store.save(
            PredictionLogPayload(
                model_name=request.model_name,
                model_version=model_version,
                run_id=run_id,
                request_id=request.request_id,
                target_type=request.target_type,
                target_id=request.target_id,
                qualifiers=request.qualifiers,
                predicted_at=predicted_at,
                target_timestamp=request.target_timestamp or predicted_at,
                predicted_value=predictions,
                input_features={"inputs": request.inputs},
            )
        )
    except Exception:
        app_logger.exception(
            "Failed to save prediction log",
            extra={"model_name": request.model_name, "request_id": request.request_id},
        )
