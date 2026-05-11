from threading import RLock
from typing import Any

import mlflow.pyfunc

from app.core.exceptions import ModelNotLoadedError
from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.config import mlops_settings
from app.domains.mlops.model_registry import get_model_by_alias
from app.domains.mlops.schemas import ModelLoadResult


class CachedModel:
    def __init__(self, model: Any, info: ModelLoadResult) -> None:
        self.model = model
        self.info = info


class ModelLoader:
    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}
        self._lock = RLock()

    def load(self, model_name: str, force_reload: bool = False) -> ModelLoadResult:
        with self._lock:
            cached = self._cache.get(model_name)
            if cached is not None and not force_reload:
                app_logger.info(
                    "Model cache hit",
                    extra={
                        "model_name": model_name,
                        "model_version": cached.info.version,
                        "run_id": cached.info.run_id,
                    },
                )
                return cached.info

            model_uri = f"models:/{model_name}@{mlops_settings.model_champion_alias}"
            app_logger.info(
                "Loading model from MLflow",
                extra={"model_name": model_name, "model_uri": model_uri, "force_reload": force_reload},
            )
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                registry_info = get_model_by_alias(model_name, mlops_settings.model_champion_alias)
                load_info = ModelLoadResult(
                    model_name=model_name,
                    model_uri=model_uri,
                    version=registry_info.version if registry_info else None,
                    run_id=registry_info.run_id if registry_info else None,
                    loaded_at=now_in_app_timezone(),
                )
                self._cache[model_name] = CachedModel(model=model, info=load_info)
                app_logger.info(
                    "Model loaded from MLflow",
                    extra={
                        "model_name": model_name,
                        "model_uri": model_uri,
                        "model_version": load_info.version,
                        "run_id": load_info.run_id,
                    },
                )
                return load_info
            except Exception:
                app_logger.exception(
                    "Failed to load model from MLflow",
                    extra={"model_name": model_name, "model_uri": model_uri},
                )
                raise

    def get(self, model_name: str) -> CachedModel:
        with self._lock:
            cached = self._cache.get(model_name)
            if cached is None:
                app_logger.warning("Model cache miss", extra={"model_name": model_name})
                raise ModelNotLoadedError(f"Model is not loaded: {model_name}")
            return cached

    def predict(self, model_name: str, payload: Any) -> Any:
        cached = self.get(model_name)
        return cached.model.predict(payload)

    def status(self, model_name: str) -> ModelLoadResult | None:
        with self._lock:
            cached = self._cache.get(model_name)
            return cached.info if cached else None


model_loader = ModelLoader()
