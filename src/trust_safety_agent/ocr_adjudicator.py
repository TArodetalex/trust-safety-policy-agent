"""Conservative local OCR enrichment for the deterministic rules engine."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
)


MIN_OCR_CONFIDENCE = 0.65
MIN_OCR_CHARACTERS = 8


class OCRUnavailableError(RuntimeError):
    """Raised when local OCR cannot produce trustworthy text."""


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    line_count: int


class OCRExtractor(Protocol):
    def extract(self, image_path: Path) -> OCRResult:
        ...


class MacOSVisionOCR:
    """Compile and invoke the bundled macOS Vision OCR helper."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def _binary_path(self) -> Path:
        if platform.system() != "Darwin":
            raise OCRUnavailableError("macOS Vision OCR is only available on macOS")
        compiler = shutil.which("swiftc")
        if compiler is None:
            raise OCRUnavailableError("swiftc is required for macOS Vision OCR")
        if not self.source_path.is_file():
            raise OCRUnavailableError("macOS Vision OCR helper source is missing")

        digest = hashlib.sha256(self.source_path.read_bytes()).hexdigest()[:16]
        cache_dir = Path(tempfile.gettempdir()) / "trust-safety-agent-ocr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        binary = cache_dir / f"macos-vision-ocr-{digest}"
        if binary.is_file():
            return binary

        temporary = cache_dir / f".{binary.name}-{os.getpid()}"
        command = [
            compiler,
            str(self.source_path),
            "-o",
            str(temporary),
            "-framework",
            "Vision",
            "-framework",
            "AppKit",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            temporary.replace(binary)
        except (OSError, subprocess.SubprocessError) as exc:
            temporary.unlink(missing_ok=True)
            raise OCRUnavailableError(
                "unable to compile the macOS Vision OCR helper"
            ) from exc
        return binary

    def extract(self, image_path: Path) -> OCRResult:
        if not image_path.is_file():
            raise OCRUnavailableError(f"image does not exist: {image_path}")
        try:
            process = subprocess.run(
                [str(self._binary_path()), str(image_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            payload = json.loads(process.stdout)
            lines = payload["lines"]
            text_lines = [
                str(line["text"]).strip()
                for line in lines
                if str(line["text"]).strip()
            ]
            confidences = [
                float(line["confidence"])
                for line in lines
                if str(line["text"]).strip()
            ]
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            raise OCRUnavailableError("macOS Vision OCR failed") from exc
        if not text_lines:
            raise OCRUnavailableError("macOS Vision OCR returned no text")
        return OCRResult(
            text=" ".join(text_lines),
            confidence=sum(confidences) / len(confidences),
            line_count=len(text_lines),
        )


class OCRPolicyAdjudicator:
    """Use OCR evidence only for explicit violations or supported exemptions."""

    def __init__(
        self,
        rules: PolicyAdjudicator,
        extractor: OCRExtractor,
        minimum_confidence: float = MIN_OCR_CONFIDENCE,
    ) -> None:
        self.rules = rules
        self.extractor = extractor
        self.minimum_confidence = minimum_confidence

    @staticmethod
    def _review(case_id: str, reason: str) -> AgentDecision:
        return AgentDecision(
            case_id=case_id,
            decision=AgentDecisionLabel.NEED_REVIEW,
            reason=reason,
            recommended_action="Send the image case to a human reviewer.",
            confidence=0.0,
        )

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        image_path: Path,
    ) -> AgentDecision:
        try:
            result = self.extractor.extract(image_path)
        except OCRUnavailableError as exc:
            return self._review(case_id, f"Image OCR was unavailable: {exc}")
        if (
            result.confidence < self.minimum_confidence
            or len(result.text) < MIN_OCR_CHARACTERS
        ):
            return self._review(
                case_id,
                "Image OCR evidence was below the automation threshold.",
            )

        combined_text = f"{input_text}\nVisual OCR evidence: {result.text}"
        decision = self.rules.adjudicate(
            case_id=case_id,
            content_type=content_type,
            input_text=combined_text,
        )
        if (
            decision.decision == AgentDecisionLabel.APPROVE
            and not decision.matched_policy
        ):
            return self._review(
                case_id,
                "OCR found no explicit violation or supported exemption; "
                "absence of recognized text is not sufficient for approval.",
            )
        if decision.decision == AgentDecisionLabel.NEED_REVIEW:
            return decision
        return decision.model_copy(
            update={
                "reason": (
                    f"Local OCR ({result.line_count} lines, "
                    f"confidence {result.confidence:.2f}) supplied the visual "
                    f"evidence. {decision.reason}"
                )
            }
        )
