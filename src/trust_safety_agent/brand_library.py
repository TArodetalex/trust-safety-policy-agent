"""Structured controlled-brand library and deterministic mention matching."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

from pydantic import Field, model_validator

from trust_safety_agent.schema import PolicyLabel, StrictModel, StringEnum


class BrandStatus(StringEnum):
    ACTIVE = "active"
    WATCHLIST = "watchlist"
    INACTIVE = "inactive"


class ControlledBrand(StrictModel):
    brand_name: str = Field(min_length=1, max_length=100)
    aliases: List[str] = Field(default_factory=list, max_length=30)
    region: str = Field(min_length=1, max_length=100)
    policy_category: PolicyLabel
    status: BrandStatus = BrandStatus.ACTIVE
    is_common_word: bool = False
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_aliases(self) -> "ControlledBrand":
        names = [self.brand_name, *self.aliases]
        normalized = [normalize_brand_text(name) for name in names]
        if any(not name for name in normalized):
            raise ValueError("brand names and aliases cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("brand aliases must be unique within one record")
        return self


class BrandMention(StrictModel):
    brand_name: str
    matched_alias: str
    evidence: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    is_common_word: bool = False


def normalize_brand_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    left = r"(?<![\w])" if alias[0].isalnum() else ""
    right = r"(?![\w])" if alias[-1].isalnum() else ""
    return re.compile(f"{left}{escaped}{right}", re.IGNORECASE)


class ControlledBrandLibrary:
    def __init__(self, brands: Iterable[ControlledBrand]) -> None:
        self.brands = list(brands)
        if not self.brands:
            raise ValueError("controlled brand library cannot be empty")
        self._aliases: Dict[str, ControlledBrand] = {}
        for brand in self.brands:
            for alias in (brand.brand_name, *brand.aliases):
                normalized = normalize_brand_text(alias)
                existing = self._aliases.get(normalized)
                if existing and existing.brand_name != brand.brand_name:
                    raise ValueError(
                        f"alias {alias!r} belongs to multiple controlled brands"
                    )
                self._aliases[normalized] = brand

    @classmethod
    def from_csv(cls, path: Path) -> "ControlledBrandLibrary":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        brands = []
        for row_number, row in enumerate(rows, start=2):
            try:
                brands.append(
                    ControlledBrand(
                        brand_name=row.get("brand_name", ""),
                        aliases=[
                            alias.strip()
                            for alias in row.get("aliases", "").split("|")
                            if alias.strip()
                        ],
                        region=row.get("region", "global"),
                        policy_category=row.get(
                            "policy_category", "trademark_misuse"
                        ),
                        status=row.get("status", "active"),
                        is_common_word=(
                            row.get("is_common_word", "").casefold() == "true"
                        ),
                        notes=row.get("notes", ""),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid controlled brand row {row_number}: {exc}"
                ) from exc
        return cls(brands)

    def find_mentions(self, text: str) -> List[BrandMention]:
        mentions: List[BrandMention] = []
        seen = set()
        for alias, brand in self._aliases.items():
            if brand.status == BrandStatus.INACTIVE:
                continue
            for match in _alias_pattern(alias).finditer(text):
                key = (brand.brand_name, match.start(), match.end())
                if key in seen:
                    continue
                seen.add(key)
                start = max(0, match.start() - 35)
                end = min(len(text), match.end() + 35)
                mentions.append(
                    BrandMention(
                        brand_name=brand.brand_name,
                        matched_alias=match.group(0),
                        evidence=text[start:end].strip(),
                        start=match.start(),
                        end=match.end(),
                        is_common_word=brand.is_common_word,
                    )
                )
        return sorted(mentions, key=lambda item: (item.start, item.brand_name))

    def get(self, brand_name: str) -> ControlledBrand | None:
        normalized = normalize_brand_text(brand_name)
        return self._aliases.get(normalized)
