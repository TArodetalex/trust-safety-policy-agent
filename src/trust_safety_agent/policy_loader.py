"""Markdown policy parsing and structure-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trust_safety_agent.schema import ExemptionType, PolicyChunk, PolicyLabel


POLICY_HEADING = re.compile(
    r"^##\s+(?P<policy_id>POL-[A-Z]{2}-\d{3})\s+(?P<title>.+?)\s*$"
)
SUBHEADING = re.compile(r"^###\s+(?P<title>.+?)\s*$")
VERSION = re.compile(r"^Version:\s+`?(?P<version>v\d+\.\d+\.\d+)`?\s*$")

POLICY_LABELS: Dict[str, PolicyLabel] = {
    "POL-CF-001": PolicyLabel.COUNTERFEIT,
    "POL-KO-001": PolicyLabel.KNOCKOFF,
    "POL-TM-001": PolicyLabel.TRADEMARK_MISUSE,
    "POL-SI-001": PolicyLabel.SHOP_IMPERSONATION,
    "POL-RQ-001": PolicyLabel.RISKY_QUERY,
}

EXEMPTION_LABELS: Dict[str, ExemptionType] = {
    "compatibility": ExemptionType.COMPATIBILITY,
    "second-hand": ExemptionType.SECOND_HAND,
    "meaningful word": ExemptionType.MEANINGFUL_WORD,
    "incidental exposure": ExemptionType.INCIDENTAL_EXPOSURE,
    "co-brand": ExemptionType.CO_BRAND,
}


@dataclass(frozen=True)
class PolicySection:
    policy_id: str
    policy_title: str
    subheading: Optional[str]
    body: str

    @property
    def heading_path(self) -> List[str]:
        path = [self.policy_title]
        if self.subheading:
            path.append(self.subheading)
        return path


def _paragraphs(body: str) -> List[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", body)
        if paragraph.strip()
    ]


def _chunk_body(body: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")

    chunks: List[str] = []
    current = ""
    for paragraph in _paragraphs(body):
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def parse_policy_sections(markdown: str) -> Tuple[str, List[PolicySection]]:
    version = "v1.0.0"
    sections: List[PolicySection] = []
    policy_id: Optional[str] = None
    policy_title: Optional[str] = None
    subheading: Optional[str] = None
    body_lines: List[str] = []

    def flush() -> None:
        nonlocal body_lines
        body = "\n".join(body_lines).strip()
        if policy_id and policy_title and body:
            sections.append(
                PolicySection(
                    policy_id=policy_id,
                    policy_title=policy_title,
                    subheading=subheading,
                    body=body,
                )
            )
        body_lines = []

    for line in markdown.splitlines():
        version_match = VERSION.match(line)
        if version_match:
            version = version_match.group("version")
            continue

        policy_match = POLICY_HEADING.match(line)
        if policy_match:
            flush()
            policy_id = policy_match.group("policy_id")
            policy_title = policy_match.group("title")
            subheading = None
            continue

        subheading_match = SUBHEADING.match(line)
        if subheading_match and policy_id:
            flush()
            subheading = subheading_match.group("title")
            continue

        if line.startswith("## "):
            flush()
            policy_id = None
            policy_title = None
            subheading = None
            continue

        if policy_id:
            body_lines.append(line)

    flush()
    return version, sections


def load_policy_text(
    markdown: str,
    source: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> List[PolicyChunk]:
    version, sections = parse_policy_sections(markdown)
    chunks: List[PolicyChunk] = []

    for section in sections:
        prefix = " > ".join(section.heading_path)
        policy_label = POLICY_LABELS.get(section.policy_id)
        exemption_type = EXEMPTION_LABELS.get(
            (section.subheading or "").lower(), ExemptionType.NONE
        )
        for index, body_chunk in enumerate(
            _chunk_body(section.body, max_chars, overlap_chars)
        ):
            content = f"{prefix}\n\n{body_chunk}"
            digest_input = (
                f"{source}|{version}|{section.policy_id}|{prefix}|{index}|{content}"
            )
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                PolicyChunk(
                    chunk_id=f"PCH-{digest}",
                    policy_id=section.policy_id,
                    title=section.subheading or section.policy_title,
                    heading_path=section.heading_path,
                    source=source,
                    policy_version=version,
                    chunk_index=index,
                    content=content,
                    policy_label=policy_label,
                    exemption_type=exemption_type,
                )
            )
    return chunks


def load_policy_file(
    path: Path,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> List[PolicyChunk]:
    return load_policy_text(
        path.read_text(encoding="utf-8"),
        source=path.name,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
