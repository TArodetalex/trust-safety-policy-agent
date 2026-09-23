from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_LIBRARY_PATH


def test_controlled_brand_library_loads_expected_records() -> None:
    library = ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)

    assert len(library.brands) == 10
    assert library.get("古驰").brand_name == "Gucci"
    assert library.get("iphone").brand_name == "Apple"


def test_short_alias_uses_word_boundaries() -> None:
    library = ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)

    assert library.find_mentions("a large replacement part") == []
    mentions = library.find_mentions("GE replacement part")
    assert [mention.brand_name for mention in mentions] == ["GE"]

