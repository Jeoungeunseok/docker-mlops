from typing import Any

from app.domains.mlops.schemas import TrainingContext, TrainingDataProcessor, TrainingDataset


def build_training_dataset(
    context: TrainingContext,
    processor: TrainingDataProcessor,
) -> TrainingDataset:
    raw_data = processor.load_training_data(context)
    processed_data = processor.preprocess(raw_data, context)
    features = processor.build_features(processed_data, context)
    return processor.split_validation(features, context)


class PassthroughTrainingDataProcessor:
    def load_training_data(self, context: TrainingContext) -> Any:
        raise NotImplementedError("Connect the project-specific training data loader here.")

    def preprocess(self, raw_data: Any, context: TrainingContext) -> Any:
        return raw_data

    def build_features(self, processed_data: Any, context: TrainingContext) -> Any:
        return processed_data

    def split_validation(self, features: Any, context: TrainingContext) -> TrainingDataset:
        raise NotImplementedError("Connect the project-specific validation split implementation here.")
