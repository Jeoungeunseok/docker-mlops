from datetime import datetime
from typing import Any

from app.domains.mlops.schemas import ModelLoadResult, PredictionLogPayload
from app.domains.prediction.schemas import PredictionRequest
from app.domains.prediction.services import prediction_service


class DummyModelLoader:
    def load(self, model_name: str) -> ModelLoadResult:
        return ModelLoadResult(
            model_name=model_name,
            model_uri=f"models:/{model_name}@champion",
            version="3",
            run_id="run-1",
            loaded_at=datetime(2026, 1, 1, 12, 0, 0),
        )

    def predict(self, model_name: str, payload: Any) -> list[float]:
        return [1.25]


class CapturingPredictionLogStore:
    def __init__(self) -> None:
        self.payloads: list[PredictionLogPayload] = []

    def save(self, payload: PredictionLogPayload) -> None:
        self.payloads.append(payload)

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        pass

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        return [payload for payload in self.payloads if payload.model_name == model_name]


def test_predict_returns_model_metadata_and_saves_prediction_log(monkeypatch: Any) -> None:
    log_store = CapturingPredictionLogStore()
    monkeypatch.setattr(prediction_service, "model_loader", DummyModelLoader())
    monkeypatch.setattr(prediction_service, "prediction_log_store", log_store)
    monkeypatch.setattr(prediction_service, "now_in_app_timezone", lambda: datetime(2026, 1, 1, 12, 30, 0))

    response = prediction_service.predict(
        PredictionRequest(
            model_name="forecast_global",
            request_id="req-1",
            inputs=[{"x": 1}],
        )
    )

    assert response.model_name == "forecast_global"
    assert response.model_version == "3"
    assert response.run_id == "run-1"
    assert response.predictions == [1.25]
    assert len(log_store.payloads) == 1
    assert log_store.payloads[0].request_id == "req-1"
    assert log_store.payloads[0].input_features == {"inputs": [{"x": 1}]}
