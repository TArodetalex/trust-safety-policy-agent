"""Build Golden Set v4 and deterministic Day 7 multimodal fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v3.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"
ASSET_DIRECTORY = PROJECT_ROOT / "data/golden_set/assets/v4"
ASSET_MANIFEST = ASSET_DIRECTORY / "asset_manifest.csv"
SPLIT_DIRECTORY = PROJECT_ROOT / "data/golden_set/splits/v4"
TUNING_PATH = SPLIT_DIRECTORY / "golden_set_v4_tuning.csv"
BLIND_PATH = SPLIT_DIRECTORY / "blind_set_v2.csv"
SPLIT_MANIFEST = SPLIT_DIRECTORY / "split_manifest.json"
ANNOTATION_LEDGER = SPLIT_DIRECTORY / "annotation_ledger_v4.csv"
SCHEMA_VERSION = "1.2.0"
DATASET_VERSION = "v4.0.0"

FIELDNAMES = [
    "schema_version",
    "dataset_version",
    "case_id",
    "content_type",
    "input_text",
    "image_url",
    "image_path",
    "human_decision",
    "expected_policy",
    "expected_exemption",
    "risk_level",
    "is_boundary_case",
    "human_reason",
    "source",
    "split",
    "expected_route",
    "tags",
]


@dataclass(frozen=True)
class VisualSpec:
    content_type: str
    decision: str
    label: str
    visual_text: str
    condition: str
    split: str
    expected_route: str = "auto_decide"


POLICY_FIXTURES = [
    (
        "product",
        "counterfeit",
        "NIKE COUNTERFEIT SHOES - 1:1 REPLICA",
    ),
    (
        "video",
        "knockoff",
        "BIRKIN KNOCKOFF - SAME PROTECTED DESIGN",
    ),
    (
        "shop_avatar",
        "trademark_misuse",
        "PRINT NIKE LOGO ON GENERIC CAPS FOR RESALE",
    ),
    (
        "shop_name",
        "shop_impersonation",
        "APPLE OFFICIAL VERIFIED FLAGSHIP",
    ),
    (
        "query",
        "risky_query",
        "HOW TO HIDE COUNTERFEIT ROLEX LISTINGS FROM REVIEW",
    ),
]
EXEMPTION_FIXTURES = [
    (
        "product",
        "compatibility",
        "CASE COMPATIBLE WITH IPAD - NOT MADE BY APPLE",
    ),
    (
        "video",
        "second_hand",
        "PRE-OWNED AUTHENTIC NIKE SHOES - CONDITION SHOWN",
    ),
    (
        "query",
        "meaningful_word",
        "FRESH APPLE JUICE RECIPE",
    ),
    (
        "shop_name",
        "incidental_exposure",
        "FAMILY PORTRAIT - SMALL STORE SIGN IN BACKGROUND",
    ),
    (
        "shop_avatar",
        "co_brand",
        "LICENSED ADIDAS X LEGO COLLECTION - TAGS SHOWN",
    ),
]
CLEAN_TEXTS = [
    "PLAIN GLASS STORAGE JAR - NO LOGOS",
    "UNBRANDED COTTON TOTE - NO MARKS",
    "GENERIC STEEL WATER BOTTLE",
    "PLAIN CERAMIC CUP WITH GEOMETRIC PATTERN",
    "UNBRANDED DESK LAMP",
    "GENERIC WOODEN PHOTO FRAME",
    "PLAIN CANVAS POUCH - NO LOGOS",
    "UNBRANDED KITCHEN CONTAINER",
    "GENERIC NOTEBOOK WITH ABSTRACT PATTERN",
    "PLAIN PHONE STAND - NO BRAND MARKS",
]
POLICY_CONDITIONS = [
    "clear",
    "low_contrast",
    "rotated",
    "occluded",
    "clear",
    "tiny_text",
]
EXEMPTION_CONDITIONS = ["clear", "low_contrast", "rotated"]
CLEAN_CONDITIONS = [
    "clear",
    "low_contrast",
    "rotated",
    "tiny_text",
    "clear",
    "low_contrast",
    "rotated",
    "tiny_text",
    "clear",
    "low_contrast",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fixture_specs() -> List[VisualSpec]:
    specs: List[VisualSpec] = []
    case_number = 151
    for content_type, label, visual_text in POLICY_FIXTURES:
        for condition in POLICY_CONDITIONS:
            split = (
                "regression"
                if case_number <= 162
                else "development"
                if case_number <= 170
                else "blind"
            )
            specs.append(
                VisualSpec(
                    content_type=content_type,
                    decision="reject",
                    label=label,
                    visual_text=visual_text,
                    condition=condition,
                    split=split,
                    expected_route=(
                        "human_review"
                        if condition == "occluded"
                        else "auto_decide"
                    ),
                )
            )
            case_number += 1

    for content_type, label, visual_text in EXEMPTION_FIXTURES:
        for condition in EXEMPTION_CONDITIONS:
            specs.append(
                VisualSpec(
                    content_type=content_type,
                    decision="approve",
                    label=label,
                    visual_text=visual_text,
                    condition=condition,
                    split=(
                        "regression"
                        if case_number <= 192
                        else "development"
                    ),
                )
            )
            case_number += 1

    for visual_text, condition in zip(CLEAN_TEXTS, CLEAN_CONDITIONS):
        specs.append(
            VisualSpec(
                content_type="product",
                decision="approve",
                label="none",
                visual_text=visual_text,
                condition=condition,
                split="development" if case_number <= 200 else "blind",
                expected_route="human_review",
            )
        )
        case_number += 1

    for index in range(5):
        specs.append(
            VisualSpec(
                content_type="product",
                decision="approve",
                label="none",
                visual_text=(
                    f"GENERIC PRODUCT {index + 1} - BRAND AREA AMBIGUOUS"
                ),
                condition="occluded",
                split="blind",
                expected_route="human_review",
            )
        )
        case_number += 1
    return specs


def draw_subject(
    draw: ImageDraw.ImageDraw,
    content_type: str,
    accent: str,
) -> None:
    if content_type == "query":
        draw.ellipse((660, 255, 745, 340), outline=accent, width=9)
        draw.line((727, 327, 790, 390), fill=accent, width=11)
    elif content_type == "shop_name":
        draw.polygon(
            [(620, 300), (810, 300), (780, 235), (650, 235)],
            fill=accent,
        )
        draw.rectangle((645, 300, 785, 420), outline=accent, width=7)
    elif content_type == "shop_avatar":
        draw.ellipse((620, 220, 815, 415), outline=accent, width=8)
        draw.ellipse((680, 260, 755, 335), fill=accent)
        draw.arc((655, 315, 780, 420), 190, 350, fill=accent, width=24)
    elif content_type == "video":
        draw.rounded_rectangle(
            (620, 220, 815, 420),
            radius=18,
            fill="#20262b",
            outline=accent,
            width=5,
        )
        draw.polygon([(690, 270), (690, 370), (775, 320)], fill="#ffffff")
    else:
        draw.rounded_rectangle(
            (615, 225, 800, 430),
            radius=22,
            fill="#ffffff",
            outline=accent,
            width=7,
        )
        for y, width in ((275, 135), (320, 115), (365, 145)):
            draw.line((640, y, 640 + width, y), fill=accent, width=8)


def draw_fixture(case_id: str, spec: VisualSpec, path: Path) -> None:
    palettes = {
        "product": ("#e9f2f1", "#18736b"),
        "video": ("#f3ece5", "#9b4d27"),
        "query": ("#eaf0f8", "#315f9c"),
        "shop_name": ("#f5edf2", "#8b3f6b"),
        "shop_avatar": ("#eef0e8", "#687533"),
    }
    background, accent = palettes[spec.content_type]
    text_color = "#182026"
    if spec.condition == "low_contrast":
        accent = "#7f8c8d"
        text_color = "#92999e"

    image = Image.new("RGB", (960, 640), background)
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    body_font = load_font(22 if spec.condition == "tiny_text" else 40)
    small_font = load_font(22)
    draw.rectangle((0, 0, 960, 72), fill=accent)
    draw.text(
        (34, 20),
        "CONTROLLED SYNTHETIC EVIDENCE",
        fill="white",
        font=title_font,
    )
    draw.rounded_rectangle(
        (70, 120, 890, 540),
        radius=18,
        fill="white",
        outline=accent,
        width=4,
    )
    draw.text(
        (105, 150),
        spec.content_type.replace("_", " ").upper(),
        fill=accent,
        font=title_font,
    )
    line_width = 43 if spec.condition == "tiny_text" else 24
    y = 235
    for line in textwrap.wrap(spec.visual_text, width=line_width)[:5]:
        draw.text((105, y), line, fill=text_color, font=body_font)
        y += 34 if spec.condition == "tiny_text" else 58
    draw_subject(draw, spec.content_type, accent)
    draw.text(
        (105, 492),
        f"Fixture {case_id} | visual evidence only",
        fill="#56616a",
        font=small_font,
    )
    if spec.condition == "occluded":
        draw.rectangle((90, 215, 850, 440), fill="#d9dde1")
    if spec.condition == "rotated":
        image = image.rotate(4, resample=Image.Resampling.BICUBIC)
    image.save(path, format="PNG", optimize=True)


def reason_for(spec: VisualSpec) -> str:
    if spec.expected_route == "human_review":
        return (
            "The controlled image intentionally obscures material evidence, "
            "so a human must verify the known ground truth."
        )
    if spec.decision == "reject":
        return (
            f"The controlled image contains an explicit {spec.label} signal."
        )
    if spec.label != "none":
        return (
            f"The controlled image clearly satisfies the {spec.label} exemption."
        )
    if "AMBIGUOUS" in spec.visual_text:
        return (
            "The controlled image intentionally obscures material evidence, "
            "so a human must verify the known ground truth."
        )
    return (
        "The controlled image shows a generic product, but the absence of "
        "protected marks must remain in shadow evaluation until independently "
        "validated."
    )


def tags_for(spec: VisualSpec) -> List[str]:
    tags = {
        "day7",
        "image_dependent",
        "multimodal",
        "shadow",
        spec.condition,
    }
    if spec.decision == "reject":
        tags.update({"explicit_signal", spec.label})
    elif spec.label != "none":
        tags.update({"exemption", spec.label})
    elif "AMBIGUOUS" in spec.visual_text:
        tags.update({"negative_control", "brand_ambiguous"})
    else:
        tags.update({"negative_control", "clean_generic"})
    return sorted(tags)


def build_new_rows() -> tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    assets: List[dict] = []
    for case_number, spec in enumerate(fixture_specs(), start=151):
        case_id = f"C{case_number:03d}"
        relative_path = f"data/golden_set/assets/v4/IMG-{case_id}.png"
        asset_path = PROJECT_ROOT / relative_path
        draw_fixture(case_id, spec, asset_path)
        expected_policy = spec.label if spec.decision == "reject" else ""
        expected_exemption = (
            spec.label
            if spec.decision == "approve" and spec.label != "none"
            else "none"
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "case_id": case_id,
                "content_type": spec.content_type,
                "input_text": (
                    f"{spec.content_type.replace('_', ' ').title()} submitted "
                    "for Day 7 visual policy review."
                ),
                "image_url": "",
                "image_path": relative_path,
                "human_decision": spec.decision,
                "expected_policy": expected_policy,
                "expected_exemption": expected_exemption,
                "risk_level": (
                    "high"
                    if spec.decision == "reject"
                    else "medium"
                    if spec.label != "none"
                    or "AMBIGUOUS" in spec.visual_text
                    else "low"
                ),
                "is_boundary_case": "true",
                "human_reason": reason_for(spec),
                "source": "controlled_visual_fixture_day7",
                "split": spec.split,
                "expected_route": spec.expected_route,
                "tags": "|".join(tags_for(spec)),
            }
        )
        assets.append(
            {
                "asset_id": f"IMG-{case_id}",
                "relative_path": relative_path,
                "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                "media_type": "image/png",
                "source_type": "deterministic_synthetic",
                "license_or_terms": "project_generated",
                "created_by": "build_golden_set_v4.py",
                "notes": (
                    "Day 7 controlled visual fixture; "
                    f"condition={spec.condition}."
                ),
            }
        )
    return rows, assets


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    specs = fixture_specs()
    if len(specs) != 60:
        raise RuntimeError(f"expected 60 Day 7 fixtures, found {len(specs)}")
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SPLIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with SEED_PATH.open(newline="", encoding="utf-8") as source:
        seed_rows = list(csv.DictReader(source))
    for row in seed_rows:
        row["dataset_version"] = DATASET_VERSION
        row["schema_version"] = SCHEMA_VERSION

    new_rows, assets = build_new_rows()
    all_rows = seed_rows + new_rows
    tuning_rows = [row for row in all_rows if row["split"] != "blind"]
    blind_rows = [row for row in all_rows if row["split"] == "blind"]
    write_csv(OUTPUT_PATH, all_rows, FIELDNAMES)
    write_csv(TUNING_PATH, tuning_rows, FIELDNAMES)
    write_csv(BLIND_PATH, blind_rows, FIELDNAMES)

    asset_fields = [
        "asset_id",
        "relative_path",
        "sha256",
        "media_type",
        "source_type",
        "license_or_terms",
        "created_by",
        "notes",
    ]
    write_csv(ASSET_MANIFEST, assets, asset_fields)
    ledger_rows = [
        {
            "case_id": row["case_id"],
            "status": "synthetic_verified",
            "annotator": "day7_fixture_builder",
            "notes": (
                "Synthetic label; requires independent human signoff before "
                "production release."
            ),
        }
        for row in new_rows
    ]
    write_csv(
        ANNOTATION_LEDGER,
        ledger_rows,
        ["case_id", "status", "annotator", "notes"],
    )

    split_payload = {
        "manifest_version": "v1.0.0",
        "dataset_version": DATASET_VERSION,
        "full": {
            "path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            "count": len(all_rows),
            "sha256": file_sha256(OUTPUT_PATH),
        },
        "tuning": {
            "path": str(TUNING_PATH.relative_to(PROJECT_ROOT)),
            "count": len(tuning_rows),
            "sha256": file_sha256(TUNING_PATH),
        },
        "blind": {
            "path": str(BLIND_PATH.relative_to(PROJECT_ROOT)),
            "count": len(blind_rows),
            "sha256": file_sha256(BLIND_PATH),
        },
        "new_visual_cases": len(new_rows),
        "new_blind_visual_cases": sum(
            row["split"] == "blind" for row in new_rows
        ),
        "invariants": {
            "tuning_and_blind_are_disjoint": True,
            "tuning_union_blind_equals_full": True,
            "blind_content_requires_restricted_access": True,
        },
    }
    SPLIT_MANIFEST.write_text(
        json.dumps(split_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {OUTPUT_PATH} with {len(all_rows)} cases and "
        f"{len(assets)} new visual fixtures; "
        f"tuning={len(tuning_rows)}, blind={len(blind_rows)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
