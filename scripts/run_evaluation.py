"""Run a versioned Golden Set evaluation and write reproducible artifacts."""

from __future__ import annotations

import argparse
import mimetypes
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.dataset import load_cases, validate_dataset
from trust_safety_agent.evaluation import (
    EvaluationGates,
    evaluate_cases,
    write_evaluation_artifacts,
)
from trust_safety_agent.llm_adjudicator import LLMPolicyAdjudicator
from trust_safety_agent.llm_client import (
    ImageInput,
    LLMSettings,
    OpenAICompatibleClient,
)
from trust_safety_agent.ocr_adjudicator import (
    MacOSVisionOCR,
    OCRPolicyAdjudicator,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.production_router import (
    ProductionRouter,
    ProductionRoutingConfig,
)
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    GoldenCase,
    ShadowEvaluationSummary,
)
from trust_safety_agent.shadow_evaluation import (
    ShadowEvaluationGates,
    make_shadow_record,
    write_shadow_artifacts,
)
from trust_safety_agent.vector_store import PolicyVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/eval_runs"
DEFAULT_GATES = PROJECT_ROOT / "config/evaluation_gates_v1.json"
DEFAULT_ROUTING = PROJECT_ROOT / "config/production_routing_v1.json"
DEFAULT_SHADOW_GATES = (
    PROJECT_ROOT / "config/shadow_evaluation_gates_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--engine",
        choices=["rules", "rules-ocr", "production", "llm"],
        default="rules",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument(
        "--shadow-gates",
        type=Path,
        default=DEFAULT_SHADOW_GATES,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--shadow-multimodal",
        action="store_true",
        help=(
            "Run the configured LLM on every image while preserving the "
            "production decision."
        ),
    )
    parser.add_argument("--min-auto-accuracy", type=float)
    parser.add_argument("--min-route-accuracy", type=float)
    parser.add_argument("--min-coverage", type=float)
    parser.add_argument("--max-false-approve-rate", type=float)
    parser.add_argument("--max-false-reject-rate", type=float)
    return parser.parse_args()


def build_decider(
    engine: str,
    store: PolicyVectorStore,
    *,
    shadow_multimodal: bool = False,
    eval_run_id: str = "EV-PENDING",
) -> tuple[Callable[[GoldenCase], AgentDecision], str]:
    if engine == "rules":
        agent = PolicyAdjudicator(store)

        def decide(case: GoldenCase) -> AgentDecision:
            if case.image_url or case.image_path:
                return AgentDecision(
                    case_id=case.case_id,
                    decision=AgentDecisionLabel.NEED_REVIEW,
                    reason=(
                        "The rules engine cannot inspect image evidence safely."
                    ),
                    recommended_action=(
                        "Use the multimodal engine or send the case to review."
                    ),
                    confidence=0.0,
                )
            return agent.adjudicate(
                case_id=case.case_id,
                content_type=case.content_type,
                input_text=case.input_text,
            )

        return decide, "rules-v1"

    if engine == "rules-ocr":
        rules = PolicyAdjudicator(store)
        ocr_agent = OCRPolicyAdjudicator(
            rules,
            MacOSVisionOCR(PROJECT_ROOT / "scripts/macos_vision_ocr.swift"),
        )

        def decide_with_ocr(case: GoldenCase) -> AgentDecision:
            if case.image_url:
                return AgentDecision(
                    case_id=case.case_id,
                    decision=AgentDecisionLabel.NEED_REVIEW,
                    reason="The local OCR engine does not fetch remote images.",
                    recommended_action=(
                        "Use the multimodal LLM or send the case to review."
                    ),
                    confidence=0.0,
                )
            if case.image_path:
                return ocr_agent.adjudicate(
                    case_id=case.case_id,
                    content_type=case.content_type,
                    input_text=case.input_text,
                    image_path=PROJECT_ROOT / case.image_path,
                )
            return rules.adjudicate(
                case_id=case.case_id,
                content_type=case.content_type,
                input_text=case.input_text,
            )

        return decide_with_ocr, "rules-ocr-v1"

    if engine == "production":
        rules = PolicyAdjudicator(store)
        ocr_agent = OCRPolicyAdjudicator(
            rules,
            MacOSVisionOCR(PROJECT_ROOT / "scripts/macos_vision_ocr.swift"),
        )
        settings = LLMSettings.from_env()
        llm_client = (
            OpenAICompatibleClient(settings) if settings.configured else None
        )
        llm_agent = (
            LLMPolicyAdjudicator(store, llm_client)
            if llm_client is not None
            else None
        )
        if shadow_multimodal and llm_agent is None:
            raise ValueError(
                "Set LLM_API_KEY and LLM_MODEL before using "
                "--shadow-multimodal."
            )
        routing = ProductionRoutingConfig.from_json(DEFAULT_ROUTING)
        router = ProductionRouter(
            config=routing,
            rules=rules,
            ocr=ocr_agent,
            llm=llm_agent,
        )
        route_results = []
        shadow_records = []

        def decide_with_production_router(
            case: GoldenCase,
        ) -> AgentDecision:
            result = router.adjudicate(
                case_id=case.case_id,
                content_type=case.content_type,
                input_text=case.input_text,
                image_path=(
                    PROJECT_ROOT / case.image_path
                    if case.image_path
                    else None
                ),
                image_url=str(case.image_url) if case.image_url else None,
            )
            route_results.append(result)
            if shadow_multimodal and (case.image_path or case.image_url):
                assert llm_agent is not None
                assert llm_client is not None
                shadow_started = time.perf_counter()
                shadow_decision = llm_agent.adjudicate(
                    case_id=case.case_id,
                    content_type=case.content_type,
                    input_text=case.input_text,
                    images=[image_input_for_case(case)],
                )
                shadow_latency_ms = max(
                    0,
                    round((time.perf_counter() - shadow_started) * 1000),
                )
                shadow_records.append(
                    make_shadow_record(
                        eval_run_id=eval_run_id,
                        case=case,
                        production_decision=result.decision,
                        shadow_decision=shadow_decision,
                        metadata=llm_client.last_metadata,
                        schema_valid=llm_agent.last_schema_valid,
                        latency_ms=shadow_latency_ms,
                    )
                )
            return result.decision

        setattr(
            decide_with_production_router,
            "route_results",
            route_results,
        )
        setattr(
            decide_with_production_router,
            "shadow_records",
            shadow_records,
        )
        setattr(
            decide_with_production_router,
            "shadow_model",
            settings.model if settings.configured else "",
        )
        provider = settings.model if settings.configured else "offline"
        return (
            decide_with_production_router,
            f"production-routing-{routing.routing_version}:{provider}",
        )

    settings = LLMSettings.from_env()
    if not settings.configured:
        raise ValueError(
            "Set LLM_API_KEY and LLM_MODEL before using --engine llm."
        )
    agent = LLMPolicyAdjudicator(
        store,
        OpenAICompatibleClient(settings),
    )

    def decide(case: GoldenCase) -> AgentDecision:
        images = (
            [image_input_for_case(case)]
            if case.image_url or case.image_path
            else []
        )
        return agent.adjudicate(
            case_id=case.case_id,
            content_type=case.content_type,
            input_text=case.input_text,
            images=images,
        )

    return decide, f"multimodal-v1:{settings.model}"


def image_input_for_case(case: GoldenCase) -> ImageInput:
    if case.image_url:
        return ImageInput(url=str(case.image_url))
    if case.image_path:
        image_path = PROJECT_ROOT / case.image_path
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        return ImageInput(
            data=image_path.read_bytes(),
            media_type=media_type,
        )
    raise ValueError(f"case {case.case_id} has no image input")


def main() -> int:
    args = parse_args()
    if args.shadow_multimodal and args.engine != "production":
        raise SystemExit("--shadow-multimodal requires --engine production.")
    cases = load_cases(args.dataset)
    validation_errors = validate_dataset(cases, project_root=PROJECT_ROOT)
    if validation_errors:
        for error in validation_errors:
            print(f"Dataset error: {error}")
        return 2
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        cases = cases[: args.limit]

    store = PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("EV-%Y%m%dT%H%M%SZ-")
        + uuid4().hex[:8]
    )
    try:
        decide, prompt_version = build_decider(
            args.engine,
            store,
            shadow_multimodal=args.shadow_multimodal,
            eval_run_id=run_id,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gate_values = EvaluationGates.from_json(args.gates).as_dict()
    overrides = {
        "min_auto_accuracy": args.min_auto_accuracy,
        "min_route_accuracy": args.min_route_accuracy,
        "min_coverage": args.min_coverage,
        "max_false_approve_rate": args.max_false_approve_rate,
        "max_false_reject_rate": args.max_false_reject_rate,
    }
    gate_values.update(
        {
            name: value
            for name, value in overrides.items()
            if value is not None
        }
    )
    gates = EvaluationGates(**gate_values)
    report, records = evaluate_cases(
        cases=cases,
        decide=decide,
        engine=args.engine,
        prompt_version=prompt_version,
        retrieval_version=store.collection_name,
        gates=gates,
        eval_run_id=run_id,
    )
    run_directory = args.output_dir / report.eval_run_id
    paths = write_evaluation_artifacts(report, records, run_directory)
    route_results = getattr(decide, "route_results", None)
    if route_results is not None:
        routing_path = run_directory / "routing_records.jsonl"
        routing_path.write_text(
            "\n".join(result.model_dump_json() for result in route_results)
            + "\n",
            encoding="utf-8",
        )
        route_counts = Counter(
            result.selected_engine for result in route_results
        )
        summary = ", ".join(
            f"{engine}={count}"
            for engine, count in sorted(route_counts.items())
        )
        print(f"Production routes: {summary}.")
        print(f"Routing records: {routing_path}")
    shadow_failed_gates = []
    shadow_records = getattr(decide, "shadow_records", None)
    if shadow_records:
        shadow_gates = ShadowEvaluationGates.from_json(args.shadow_gates)
        shadow_paths = write_shadow_artifacts(
            shadow_records,
            run_directory,
            report.eval_run_id,
            getattr(decide, "shadow_model"),
            shadow_gates,
        )
        shadow_summary = ShadowEvaluationSummary.model_validate_json(
            shadow_paths["shadow_report"].read_text(encoding="utf-8")
        )
        shadow_failed_gates = shadow_summary.failed_gates
        print(
            f"Shadow records: {shadow_paths['shadow_records']}\n"
            f"Shadow report: {shadow_paths['shadow_report']}"
        )
        if shadow_failed_gates:
            print("Failed shadow gates:")
            for failure in shadow_failed_gates:
                print(f"- {failure}")
        else:
            print("All shadow evaluation gates passed.")
    metrics = report.metrics
    print(
        f"Evaluation {report.eval_run_id}: "
        f"strict_accuracy={metrics.accuracy:.1%}, "
        f"auto_accuracy={metrics.auto_accuracy:.1%}, "
        f"coverage={metrics.coverage:.1%}, "
        f"route_accuracy={metrics.route_accuracy:.1%}, "
        f"review_rate={metrics.review_rate:.1%}."
    )
    print(
        f"False approve={metrics.false_approve_rate:.1%}; "
        f"false reject={metrics.false_reject_rate:.1%}; "
        f"policy accuracy={metrics.policy_accuracy:.1%}."
    )
    if report.failed_gates:
        print("Failed gates:")
        for failure in report.failed_gates:
            print(f"- {failure}")
    else:
        print("All evaluation gates passed.")
    print(f"Report: {paths['report']}")
    print(f"Records: {paths['records']}")
    print(f"Errors: {paths['errors']}")
    return (
        1
        if args.strict and (report.failed_gates or shadow_failed_gates)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
