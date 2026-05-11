from app.domains.mlops.registry import ModelTypeRegistry


class DummyComponent:
    pass


def test_registry_returns_factory_result_by_normalized_model_type() -> None:
    registry = ModelTypeRegistry[DummyComponent]()
    registry.register(" XGBoost ", DummyComponent)

    component = registry.get("xgboost")

    assert isinstance(component, DummyComponent)


def test_registry_lists_registered_model_types() -> None:
    registry = ModelTypeRegistry[DummyComponent]()
    registry.register("gru", DummyComponent)
    registry.register("xgboost", DummyComponent)

    assert registry.registered_model_types() == ["gru", "xgboost"]
