from typing import Any

from app.domains.mlops import model_registry
from app.domains.mlops.model_registry import build_model_name


def test_build_target_model_name_with_qualifier() -> None:
    assert (
        build_model_name("classifier", target_type="customer", target_id="42", qualifiers={"region": "apac"})
        == "classifier_customer_42_region_apac"
    )


def test_build_target_model_name_with_multiple_qualifiers() -> None:
    assert (
        build_model_name("forecast", target_type="store", target_id="12", qualifiers={"horizon": 24, "region": "kr"})
        == "forecast_store_12_horizon_24_region_kr"
    )


def test_build_global_model_name() -> None:
    assert build_model_name("transformer") == "transformer_global"


def test_promote_to_champion_sends_notification(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(model_registry, "set_alias", lambda model_name, version, alias: None)
    monkeypatch.setattr(
        model_registry.notification_dispatcher,
        "notify",
        lambda event: captured.setdefault("event", event),
    )

    model_registry.promote_to_champion("xgboost_global", "3")

    assert captured["event"].event_type == "champion_promoted"
    assert captured["event"].payload == {"model_name": "xgboost_global", "model_version": "3"}
