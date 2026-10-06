from trust_safety_agent.price_search import OfflineFixturePriceProvider, PriceSearchStatus, UnavailableLivePriceProvider


def test_offline_price_fixture_is_explicit_and_has_source_range() -> None:
    result = OfflineFixturePriceProvider().search_reference_price("Nike", "Air Force 1")
    assert result.status == PriceSearchStatus.SUCCESS
    assert result.provider == "demo_fixture"
    assert result.reference_price_min == 115
    assert len(result.sources) == 2
    assert all(item.source_type == "demo_fixture" for item in result.sources)


def test_unresolved_and_unavailable_prices_are_not_success() -> None:
    fixture = OfflineFixturePriceProvider()
    assert fixture.search_reference_price("Nike", "").status == PriceSearchStatus.PRODUCT_UNRESOLVED
    assert fixture.search_reference_price("Nike", "Unknown").status == PriceSearchStatus.INSUFFICIENT_SOURCES
    assert UnavailableLivePriceProvider().search_reference_price("Nike", "Air Force 1").status == PriceSearchStatus.UNAVAILABLE
