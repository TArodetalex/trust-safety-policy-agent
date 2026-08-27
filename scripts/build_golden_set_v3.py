"""Build the frozen Golden Set v3 and deterministic multimodal fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v2.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v3.csv"
ASSET_DIRECTORY = PROJECT_ROOT / "data/golden_set/assets/v3"
ASSET_MANIFEST = ASSET_DIRECTORY / "asset_manifest.csv"
SPLIT_DIRECTORY = PROJECT_ROOT / "data/golden_set/splits/v3"
TUNING_PATH = SPLIT_DIRECTORY / "golden_set_v3_tuning.csv"
BLIND_PATH = SPLIT_DIRECTORY / "blind_set_v1.csv"
SPLIT_MANIFEST = SPLIT_DIRECTORY / "split_manifest.json"
ANNOTATION_LEDGER = SPLIT_DIRECTORY / "annotation_ledger_v3.csv"
SCHEMA_VERSION = "1.2.0"
DATASET_VERSION = "v3.0.0"

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

IMAGE_CASE_IDS = {
    "C072", "C074", "C076", "C078", "C080",
    "C082", "C084", "C086", "C088", "C090",
    "C092", "C094", "C096", "C098", "C100",
    "C112", "C114", "C116", "C118", "C120",
    "C121", "C123", "C125", "C127", "C129",
    "C131", "C133", "C135", "C137", "C139",
}

HUMAN_REVIEW_IDS = {
    "C071", "C075", "C077", "C081", "C083", "C087", "C091", "C093",
    "C134", "C136", "C138", "C140", "C141", "C142", "C143",
}

CONFLICT_IDS = {
    "C071", "C075", "C077", "C081", "C083",
    "C087", "C091", "C093", "C134", "C136",
}


@dataclass(frozen=True)
class CaseSpec:
    content_type: str
    evidence_text: str
    decision: str
    label: str


SPECS: List[CaseSpec] = [
    # Counterfeit: C071-C076
    CaseSpec("product", "Gucci handbag with serial card and premium box", "reject", "counterfeit"),
    CaseSpec("product", "GUCCI 1:1 MIRROR COPY - branded dust bag included", "reject", "counterfeit"),
    CaseSpec("video", "A close look at this fake Rolex watch and packaging", "reject", "counterfeit"),
    CaseSpec("query", "WHERE TO BUY FAKE NIKE SHOES WITH ORIGINAL-LOOK BOX", "reject", "counterfeit"),
    CaseSpec("shop_name", "Rolex Timepiece Market", "reject", "counterfeit"),
    CaseSpec("shop_avatar", "AAA CHANEL REPLICA BAGS", "reject", "counterfeit"),
    # Knockoff: C077-C086
    CaseSpec("product", "Hermes style leather sandals", "reject", "knockoff"),
    CaseSpec("product", "BIRKIN KNOCKOFF - SAME DESIGN, LOWER PRICE", "reject", "knockoff"),
    CaseSpec("product", "Chrome Hearts inspired copy bracelet with cross motif", "reject", "knockoff"),
    CaseSpec("video", "MAKING A DYSON AIRWRAP DUPE", "reject", "knockoff"),
    CaseSpec("video", "Tour of our Birkin-style leather workshop", "reject", "knockoff"),
    CaseSpec("query", "BEST ADIDAS SAMBA DUPE UNDER 30", "reject", "knockoff"),
    CaseSpec("query", "affordable alternative to Dyson Airwrap", "reject", "knockoff"),
    CaseSpec("shop_name", "LV Lookalike Design Outlet", "reject", "knockoff"),
    CaseSpec("shop_avatar", "SAME DESIGN AS NIKE DUNK", "reject", "knockoff"),
    CaseSpec("shop_avatar", "HERMES ORAN DUPE", "reject", "knockoff"),
    # Trademark misuse: C087-C094
    CaseSpec("product", "Nike graphic cap", "reject", "trademark_misuse"),
    CaseSpec("product", "GENERIC BLACK CAP PRINTED WITH LARGE NIKE SWOOSH", "reject", "trademark_misuse"),
    CaseSpec("product", "GENERIC PHONE CASE WITH LARGE APPLE LOGO", "reject", "trademark_misuse"),
    CaseSpec("video", "PRINT A STARBUCKS LOGO ON GENERIC MUGS FOR RESALE", "reject", "trademark_misuse"),
    CaseSpec("query", "custom Apple logo cases", "reject", "trademark_misuse"),
    CaseSpec("shop_name", "Disney Logo Gifts and Generic Mugs", "reject", "trademark_misuse"),
    CaseSpec("shop_avatar", "Starbucks themed cup icon", "reject", "trademark_misuse"),
    CaseSpec("shop_avatar", "FULL-FRAME CHANEL LOGO FOR GENERIC ACCESSORIES", "reject", "trademark_misuse"),
    # Shop impersonation: C095-C102
    CaseSpec("video", "Welcome to the official Samsung store operated by Mobile Deals", "reject", "shop_impersonation"),
    CaseSpec("shop_name", "NIKE VERIFIED FLAGSHIP", "reject", "shop_impersonation"),
    CaseSpec("shop_name", "Apple Official Store Express", "reject", "shop_impersonation"),
    CaseSpec("shop_name", "LEGO AUTHORIZED DEALER", "reject", "shop_impersonation"),
    CaseSpec("shop_name", "Chanel Official Outlet Online", "reject", "shop_impersonation"),
    CaseSpec("shop_name", "ADIDAS VERIFIED FLAGSHIP NOW", "reject", "shop_impersonation"),
    CaseSpec("shop_avatar", "LOUIS VUITTON VERIFIED FLAGSHIP", "reject", "shop_impersonation"),
    CaseSpec("shop_avatar", "APPLE AUTHORIZED STORE", "reject", "shop_impersonation"),
    # Risky query: C103-C111
    CaseSpec("query", "where to buy fake Gucci belts that pass authenticity checks", "reject", "risky_query"),
    CaseSpec("query", "best keywords to hide counterfeit Rolex listings", "reject", "risky_query"),
    CaseSpec("query", "Nike replica supplier with no logo photos", "reject", "risky_query"),
    CaseSpec("query", "how to evade checks when selling fake AirPods", "reject", "risky_query"),
    CaseSpec("query", "cheap Chanel knockoff bags customs will not detect", "reject", "risky_query"),
    CaseSpec("query", "where can I order AAA Adidas shoes that pass authenticity checks", "reject", "risky_query"),
    CaseSpec("query", "best Dyson dupe seller and how to hide brand names", "reject", "risky_query"),
    CaseSpec("query", "how to pass authenticity checks for replica LV wallets", "reject", "risky_query"),
    CaseSpec("query", "counterfeit Samsung charger seller guide to evade review", "reject", "risky_query"),
    # Compatibility: C112-C115
    CaseSpec("product", "Protective shell compatible with iPad; not made by Apple", "approve", "compatibility"),
    CaseSpec("product", "Replacement filter fits Dyson V12 and makes no affiliation claim", "approve", "compatibility"),
    CaseSpec("query", "WHICH GENERIC CABLE IS COMPATIBLE WITH SAMSUNG GALAXY?", "approve", "compatibility"),
    CaseSpec("query", "repair tip: third-party brush head for use with Philips Sonicare", "approve", "compatibility"),
    # Second hand: C116-C119
    CaseSpec("product", "PRE-OWNED AUTHENTIC COACH BAG - RECEIPT AND WEAR SHOWN", "approve", "second_hand"),
    CaseSpec("product", "Used genuine Patagonia jacket from my personal collection", "approve", "second_hand"),
    CaseSpec("video", "CONDITION TOUR OF MY SECOND-HAND NIKE SHOES", "approve", "second_hand"),
    CaseSpec("shop_name", "Mina's Pre-Owned Authentic Designer Closet", "approve", "second_hand"),
    # Meaningful word: C120-C123
    CaseSpec("product", "FRESH APPLE JUICE - SIX BOTTLES", "approve", "meaningful_word"),
    CaseSpec("video", "PLANNING A PRODUCTIVE GAP YEAR", "approve", "meaningful_word"),
    CaseSpec("shop_name", "Amazon Rainforest Field Notes", "approve", "meaningful_word"),
    CaseSpec("shop_avatar", "HANDMADE SHELL NECKLACE", "approve", "meaningful_word"),
    # Incidental exposure: C124-C127
    CaseSpec("video", "Street interview with a McDonald's sign incidental in the background", "approve", "incidental_exposure"),
    CaseSpec("video", "COOKING LESSON - COCA-COLA BOTTLE IN BACKGROUND", "approve", "incidental_exposure"),
    CaseSpec("shop_avatar", "Family portrait; small Nike logo is incidental", "approve", "incidental_exposure"),
    CaseSpec("shop_avatar", "SELLER PORTRAIT - APPLE STORE SIGN IN BACKGROUND", "approve", "incidental_exposure"),
    # Co-brand: C128-C133
    CaseSpec("product", "Licensed Adidas x LEGO sneaker with original tags", "approve", "co_brand"),
    CaseSpec("product", "NIKE X OFF-WHITE LICENSED RELEASE - PRODUCT CODE SHOWN", "approve", "co_brand"),
    CaseSpec("product", "Genuine Uniqlo x Keith Haring shirt with original tags", "approve", "co_brand"),
    CaseSpec("product", "ADIDAS X LEGO LICENSED SET - ORIGINAL TAGS", "approve", "co_brand"),
    CaseSpec("video", "Review of genuine Nike x Off-White release purchased at launch", "approve", "co_brand"),
    CaseSpec("shop_name", "Licensed LEGO x Adidas Collection Resale", "approve", "co_brand"),
    # No exemption: C134-C150
    CaseSpec("product", "Gucci leather wallet with storage pouch", "approve", "none"),
    CaseSpec("product", "PLAIN BLUE CANVAS TOTE - NO LOGOS", "approve", "none"),
    CaseSpec("product", "Nike running shoes, size 8, original box", "approve", "none"),
    CaseSpec("product", "UNBRANDED CERAMIC CUP WITH GEOMETRIC PATTERN", "approve", "none"),
    CaseSpec("product", "Rolex watch with service card", "approve", "none"),
    CaseSpec("product", "PLAIN STEEL WATER BOTTLE - NO BRAND MARKS", "approve", "none"),
    CaseSpec("product", "Chanel handbag with authenticity card", "approve", "none"),
    CaseSpec("product", "Apple laptop, sealed retail package", "approve", "none"),
    CaseSpec("video", "Unboxing Samsung phone purchased from a local seller", "approve", "none"),
    CaseSpec("video", "Adidas sneaker condition overview", "approve", "none"),
    CaseSpec("video", "Tutorial for sewing an unbranded cotton pouch", "approve", "none"),
    CaseSpec("video", "Review of a plain desk lamp with no logos", "approve", "none"),
    CaseSpec("video", "How to photograph generic kitchen containers", "approve", "none"),
    CaseSpec("shop_name", "River Street General Goods", "approve", "none"),
    CaseSpec("shop_name", "Northside Home Supplies", "approve", "none"),
    CaseSpec("shop_avatar", "Abstract green circle with no words or logos", "approve", "none"),
    CaseSpec("shop_avatar", "Portrait illustration with no commercial branding", "approve", "none"),
]


def split_for(case_number: int, decision: str) -> str:
    if decision == "reject":
        if case_number <= 81:
            return "regression"
        if case_number <= 91:
            return "development"
        return "blind"
    if case_number <= 120:
        return "regression"
    if case_number <= 130:
        return "development"
    return "blind"


def reason_for(spec: CaseSpec, case_id: str, has_image: bool) -> str:
    evidence = "The controlled image" if has_image else "The submitted content"
    if case_id in HUMAN_REVIEW_IDS:
        if spec.decision == "reject":
            return (
                "Human investigation confirmed the violation, but the submitted "
                "brand reference alone is insufficient for automatic enforcement."
            )
        return (
            "Human verification confirmed a legitimate item, but the submitted "
            "brand-only description is insufficient for automatic approval."
        )
    if spec.decision == "reject":
        policy_reasons = {
            "counterfeit": "contains an explicit counterfeit signal with a protected brand.",
            "knockoff": "promotes an imitation of a named protected design.",
            "trademark_misuse": "uses a protected mark to promote an unrelated generic item.",
            "shop_impersonation": "presents the seller as an official or authorized brand shop.",
            "risky_query": "explicitly seeks prohibited goods or methods to evade controls.",
        }
        return f"{evidence} {policy_reasons[spec.label]}"
    exemption_reasons = {
        "compatibility": "uses the brand only to describe accurate interoperability.",
        "second_hand": "clearly describes an authentic used item without counterfeit signals.",
        "meaningful_word": "uses the term in its ordinary non-trademark meaning.",
        "incidental_exposure": "shows the brand only as incidental background content.",
        "co_brand": "provides clear licensed collaboration evidence.",
        "none": "contains no protected brand violation or deceptive commercial signal.",
    }
    return f"{evidence} {exemption_reasons[spec.label]}"


def tags_for(spec: CaseSpec, case_id: str, has_image: bool) -> List[str]:
    tags = [spec.label if spec.label != "none" else "negative_control"]
    if spec.decision == "reject" and case_id not in HUMAN_REVIEW_IDS:
        tags.append("explicit_signal")
    if spec.decision == "approve" and spec.label != "none":
        tags.append("exemption")
    if case_id in HUMAN_REVIEW_IDS:
        tags.extend(["brand_only", "insufficient_context"])
    if case_id in CONFLICT_IDS:
        tags.append("conflicting_signals")
    if has_image:
        tags.extend(["multimodal", "image_dependent", "ocr"])
    if spec.label == "none" and case_id not in HUMAN_REVIEW_IDS:
        tags.append("unbranded")
    return sorted(set(tags))


def risk_for(spec: CaseSpec, case_id: str) -> str:
    if case_id in HUMAN_REVIEW_IDS:
        return "medium"
    if spec.decision == "reject":
        return "high"
    case_number = int(case_id[1:])
    if 112 <= case_number <= 124:
        return "medium"
    return "low"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_subject(
    draw: ImageDraw.ImageDraw,
    spec: CaseSpec,
    accent: str,
) -> None:
    x1, y1, x2, y2 = 570, 185, 835, 450
    if spec.content_type == "query":
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=16,
            fill="#f4f7fa",
            outline=accent,
            width=4,
        )
        draw.rectangle((x1, y1, x2, y1 + 48), fill=accent)
        draw.ellipse((610, 280, 700, 370), outline=accent, width=10)
        draw.line((682, 352, 755, 420), fill=accent, width=12)
        return
    if spec.content_type == "shop_name":
        draw.rectangle((x1 + 25, y1 + 90, x2 - 25, y2), fill="#f7f4ef")
        draw.polygon(
            [
                (x1, y1 + 100),
                (x2, y1 + 100),
                (x2 - 35, y1 + 25),
                (x1 + 35, y1 + 25),
            ],
            fill=accent,
        )
        draw.rectangle((650, 350, 755, 450), fill=accent)
        draw.rectangle((600, 305, 640, 350), outline=accent, width=5)
        draw.rectangle((770, 305, 810, 350), outline=accent, width=5)
        return
    if spec.content_type == "shop_avatar":
        draw.ellipse((x1, y1, x2, y2), fill="#f8faf9", outline=accent, width=8)
        draw.ellipse((650, 235, 755, 340), fill=accent)
        draw.arc((620, 315, 790, 455), 190, 350, fill=accent, width=30)
        return
    if spec.content_type == "video":
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=18,
            fill="#20262b",
            outline=accent,
            width=5,
        )
        draw.polygon(
            [(670, 255), (670, 385), (775, 320)],
            fill="#ffffff",
        )
        draw.rectangle((600, 420, 805, 430), fill="#ffffff")
        draw.rectangle((600, 420, 690, 430), fill=accent)
        return

    text = spec.evidence_text.casefold()
    if any(word in text for word in ("bag", "wallet", "tote", "pouch")):
        draw.rounded_rectangle(
            (600, 260, 805, 430),
            radius=18,
            fill="#ffffff",
            outline=accent,
            width=7,
        )
        draw.arc((650, 190, 755, 320), 180, 360, fill=accent, width=9)
    elif any(word in text for word in ("shoe", "sneaker", "sandal")):
        draw.polygon(
            [(590, 350), (690, 300), (735, 355), (825, 380), (805, 425), (610, 425)],
            fill="#ffffff",
            outline=accent,
        )
        draw.line((620, 390, 800, 390), fill=accent, width=7)
    elif any(word in text for word in ("bottle", "juice")):
        for x in (615, 690, 765):
            draw.rounded_rectangle(
                (x, 250, x + 55, 430),
                radius=12,
                fill="#ffffff",
                outline=accent,
                width=6,
            )
            draw.rectangle((x + 15, 220, x + 40, 255), fill=accent)
    else:
        draw.rounded_rectangle(
            (610, 225, 800, 430),
            radius=22,
            fill="#ffffff",
            outline=accent,
            width=7,
        )
        draw.line((640, 275, 770, 275), fill=accent, width=8)
        draw.line((640, 320, 750, 320), fill=accent, width=8)
        draw.line((640, 365, 780, 365), fill=accent, width=8)


def draw_fixture(case_id: str, spec: CaseSpec, path: Path) -> None:
    palettes = {
        "product": ("#e9f2f1", "#18736b"),
        "video": ("#f3ece5", "#9b4d27"),
        "query": ("#eaf0f8", "#315f9c"),
        "shop_name": ("#f5edf2", "#8b3f6b"),
        "shop_avatar": ("#eef0e8", "#687533"),
    }
    background, accent = palettes[spec.content_type]
    image = Image.new("RGB", (960, 640), background)
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    body_font = load_font(42)
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
    wrapped = textwrap.wrap(spec.evidence_text, width=21)
    y = 230
    for line in wrapped[:5]:
        draw.text((105, y), line, fill="#182026", font=body_font)
        y += 58
    draw_subject(draw, spec, accent)
    draw.text(
        (105, 492),
        f"Fixture {case_id} | visual evidence only",
        fill="#56616a",
        font=small_font,
    )
    image.save(path, format="PNG", optimize=True)


def build_new_rows() -> tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    assets: List[dict] = []
    for offset, spec in enumerate(SPECS, start=71):
        case_id = f"C{offset:03d}"
        has_image = case_id in IMAGE_CASE_IDS
        relative_image: Optional[str] = None
        if has_image:
            relative_image = (
                f"data/golden_set/assets/v3/IMG-{case_id}.png"
            )
            asset_path = PROJECT_ROOT / relative_image
            draw_fixture(case_id, spec, asset_path)
            assets.append(
                {
                    "asset_id": f"IMG-{case_id}",
                    "relative_path": relative_image,
                    "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                    "media_type": "image/png",
                    "source_type": "deterministic_synthetic",
                    "license_or_terms": "project_generated",
                    "created_by": "build_golden_set_v3.py",
                    "notes": "Controlled visual policy fixture; no production data.",
                }
            )

        expected_policy = spec.label if spec.decision == "reject" else ""
        expected_exemption = spec.label if spec.decision == "approve" else "none"
        if spec.decision == "approve" and spec.label == "none":
            expected_exemption = "none"
        input_text = (
            f"{spec.content_type.replace('_', ' ').title()} submitted for "
            "visual policy review."
            if has_image
            else spec.evidence_text
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "case_id": case_id,
                "content_type": spec.content_type,
                "input_text": input_text,
                "image_url": "",
                "image_path": relative_image or "",
                "human_decision": spec.decision,
                "expected_policy": expected_policy,
                "expected_exemption": expected_exemption,
                "risk_level": risk_for(spec, case_id),
                "is_boundary_case": "true",
                "human_reason": reason_for(spec, case_id, has_image),
                "source": (
                    "controlled_visual_fixture"
                    if has_image
                    else "synthetic_day6"
                ),
                "split": split_for(offset, spec.decision),
                "expected_route": (
                    "human_review"
                    if case_id in HUMAN_REVIEW_IDS
                    else "auto_decide"
                ),
                "tags": "|".join(tags_for(spec, case_id, has_image)),
            }
        )
    return rows, assets


def write_rows(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(SPECS) != 80:
        raise RuntimeError(f"expected 80 new cases, found {len(SPECS)}")
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SPLIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with SEED_PATH.open(newline="", encoding="utf-8") as source:
        seed_rows = list(csv.DictReader(source))
    for row in seed_rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_version"] = DATASET_VERSION
        row["image_path"] = ""

    new_rows, assets = build_new_rows()
    all_rows = seed_rows + new_rows
    tuning_rows = [row for row in all_rows if row["split"] != "blind"]
    blind_rows = [row for row in all_rows if row["split"] == "blind"]
    write_rows(OUTPUT_PATH, all_rows)
    write_rows(TUNING_PATH, tuning_rows)
    write_rows(BLIND_PATH, blind_rows)

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
    with ASSET_MANIFEST.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=asset_fields)
        writer.writeheader()
        writer.writerows(assets)

    ledger_fields = [
        "case_id",
        "annotator_id",
        "reviewer_id",
        "initial_decision",
        "review_decision",
        "initial_policy",
        "review_policy",
        "initial_route",
        "review_route",
        "adjudicator_id",
        "status",
        "notes",
    ]
    with ANNOTATION_LEDGER.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ledger_fields)
        writer.writeheader()
        for row in new_rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "annotator_id": "synthetic_author_v1",
                    "reviewer_id": "policy_consistency_check_v1",
                    "initial_decision": row["human_decision"],
                    "review_decision": row["human_decision"],
                    "initial_policy": row["expected_policy"],
                    "review_policy": row["expected_policy"],
                    "initial_route": row["expected_route"],
                    "review_route": row["expected_route"],
                    "adjudicator_id": "",
                    "status": "synthetic_verified",
                    "notes": (
                        "Requires independent human signoff before production use."
                    ),
                }
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
        f"Built {OUTPUT_PATH} with {len(seed_rows) + len(new_rows)} cases "
        f"and {len(assets)} multimodal assets; "
        f"tuning={len(tuning_rows)}, blind={len(blind_rows)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
