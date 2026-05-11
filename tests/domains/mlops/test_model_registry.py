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
