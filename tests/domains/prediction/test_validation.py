import pytest

from app.domains.prediction.validation import RequiredFieldsValidator


def test_required_fields_validator_allows_inputs_with_required_fields() -> None:
    validator = RequiredFieldsValidator({"feature_a", "feature_b"})

    validator.validate([{"feature_a": 1, "feature_b": 2}])


def test_required_fields_validator_rejects_missing_fields() -> None:
    validator = RequiredFieldsValidator({"feature_a", "feature_b"})

    with pytest.raises(ValueError) as exc:
        validator.validate([{"feature_a": 1}])

    assert "feature_b" in str(exc.value)
