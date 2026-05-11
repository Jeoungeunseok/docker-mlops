from app.domains.mlops.model_loader import model_loader
from app.domains.mlops.schemas import ModelLoadResult


def reload_model_job(model_name: str) -> ModelLoadResult:
    return model_loader.load(model_name, force_reload=True)
