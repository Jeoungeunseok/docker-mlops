from typing import Protocol


class PredictionInputValidator(Protocol):
    def validate(self, inputs: list[dict[str, object]]) -> None:
        ...


class RequiredFieldsValidator:
    def __init__(self, required_fields: set[str]) -> None:
        self._required_fields = required_fields

    def validate(self, inputs: list[dict[str, object]]) -> None:
        for index, row in enumerate(inputs):
            missing_fields = self._required_fields - set(row)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Prediction input at index {index} is missing required fields: {missing}")
