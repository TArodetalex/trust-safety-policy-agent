"""Controlled reference-price providers used by the counterfeit rule."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol

from pydantic import Field

from trust_safety_agent.schema import StrictModel, StringEnum


class PriceSearchStatus(StringEnum):
    SUCCESS = "success"
    INSUFFICIENT_SOURCES = "insufficient_sources"
    PRODUCT_UNRESOLVED = "product_unresolved"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class PriceSource(StrictModel):
    name: str
    url: str
    source_type: str
    price: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)


class ReferencePriceResult(StrictModel):
    detected_brand: str
    detected_product: str
    currency: str = "USD"
    reference_price_min: Optional[float] = Field(default=None, gt=0)
    reference_price_max: Optional[float] = Field(default=None, gt=0)
    reference_price_selected: Optional[float] = Field(default=None, gt=0)
    sources: List[PriceSource] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(default=0, ge=0, le=1)
    status: PriceSearchStatus
    provider: str


class PriceSearchProvider(Protocol):
    def search_reference_price(self, detected_brand: str, detected_product: str, market: str = "US") -> ReferencePriceResult: ...


class OfflineFixturePriceProvider:
    """Deterministic demo data. Results are always labelled demo_fixture."""

    provider_name = "demo_fixture"

    def __init__(self, fixtures: Optional[Dict[str, List[float]]] = None) -> None:
        self.fixtures = fixtures or {
            "velora classic bag": [1700.0, 1800.0],
            "northline running shoe": [850.0, 900.0],
            "nike air force 1": [115.0, 125.0],
            "apple iphone": [799.0, 899.0],
        }

    def search_reference_price(self, detected_brand: str, detected_product: str, market: str = "US") -> ReferencePriceResult:
        key = f"{detected_brand} {detected_product}".casefold().strip()
        prices = self.fixtures.get(key)
        if not detected_product.strip():
            return ReferencePriceResult(detected_brand=detected_brand, detected_product=detected_product, status=PriceSearchStatus.PRODUCT_UNRESOLVED, provider=self.provider_name)
        if not prices:
            return ReferencePriceResult(detected_brand=detected_brand, detected_product=detected_product, status=PriceSearchStatus.INSUFFICIENT_SOURCES, provider=self.provider_name)
        sources = [PriceSource(name=f"Offline fixture {index}", url=f"demo://price/{key.replace(' ', '-')}/{index}", source_type="demo_fixture", price=price, currency="USD") for index, price in enumerate(prices, 1)]
        return ReferencePriceResult(
            detected_brand=detected_brand,
            detected_product=detected_product,
            reference_price_min=min(prices),
            reference_price_max=max(prices),
            reference_price_selected=sum(prices) / len(prices),
            sources=sources,
            confidence=0.9,
            status=PriceSearchStatus.SUCCESS,
            provider=self.provider_name,
        )


class UnavailableLivePriceProvider:
    provider_name = "live_unconfigured"

    def search_reference_price(self, detected_brand: str, detected_product: str, market: str = "US") -> ReferencePriceResult:
        return ReferencePriceResult(detected_brand=detected_brand, detected_product=detected_product, status=PriceSearchStatus.UNAVAILABLE, provider=self.provider_name)
