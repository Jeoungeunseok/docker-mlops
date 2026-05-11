from threading import RLock
from typing import Protocol

from app.domains.mlops.schemas import PredictionLogPayload


class PredictionLogStore(Protocol):
    def save(self, payload: PredictionLogPayload) -> None:
        ...

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        ...

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        ...


class InMemoryPredictionLogStore:
    def __init__(self) -> None:
        self._payloads: list[PredictionLogPayload] = []
        self._lock = RLock()

    def save(self, payload: PredictionLogPayload) -> None:
        with self._lock:
            self._payloads.append(payload)

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            for index, payload in enumerate(self._payloads):
                if payload.request_id != request_id:
                    continue
                self._payloads[index] = payload.model_copy(
                    update={
                        "actual_value": actual_value,
                        "error_value": error_value,
                        "error_metrics": error_metrics or {},
                    }
                )
                return

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        with self._lock:
            return [payload for payload in self._payloads if payload.model_name == model_name]


prediction_log_store: PredictionLogStore = InMemoryPredictionLogStore()
