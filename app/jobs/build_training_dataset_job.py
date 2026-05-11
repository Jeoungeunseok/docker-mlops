from app.domains.mlops.data_processing import build_training_dataset
from app.domains.mlops.registry import data_processor_registry
from app.domains.mlops.schemas import TrainingContext, TrainingDataProcessor, TrainingDataset


def build_training_dataset_job(
    context: TrainingContext,
    processor: TrainingDataProcessor | None = None,
) -> TrainingDataset:
    selected_processor = processor or data_processor_registry.get(context.model_type)
    return build_training_dataset(context=context, processor=selected_processor)
