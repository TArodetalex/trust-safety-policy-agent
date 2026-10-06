from pathlib import Path

import pytest

from trust_safety_agent.model_experiments import ExperimentMetrics, ModelExperimentResult, WorkflowConfiguration, WorkflowConfigurationStore
from trust_safety_agent.reviewer_prompts import PromptOwnerType, ReviewerPromptStore, ReviewerPromptVersion


def config(model_id: str) -> WorkflowConfiguration:
    return WorkflowConfiguration(workflow="product_ipr", model_id=model_id, dataset_version="candidate-v9", workflow_version="v9.1.0", prompt_version="p1", skill_version="s1", policy_index_version="pv2", brand_index_version="bv1")


def metrics() -> ExperimentMetrics:
    return ExperimentMetrics(accuracy=0.8, false_approve=1, false_reject=1, manual_review_rate=0.2, schema_valid_rate=1, exact_match_rate=0.7, conflict_rate=0, average_latency_ms=100, token_usage=1000, estimated_cost=0.1, bad_case_count=2)


def test_reviewer_prompt_versions_and_candidate_copy(tmp_path: Path) -> None:
    store = ReviewerPromptStore(tmp_path / "prompts.jsonl")
    first = store.save(ReviewerPromptVersion(name="我的商品审核", instructions="根据提供的商品证据分析品牌风险，并输出结构化结论。"))
    second = store.new_version(first.prompt_id, "根据商品证据和政策引用分析风险；证据不足时必须转人工。", use_brand_rag=False)
    candidate = store.copy_to_candidate(second.prompt_id, second.version)
    assert second.parent_version == 1
    assert second.use_brand_rag is False
    assert candidate.owner_type == PromptOwnerType.CANDIDATE
    assert candidate.prompt_id != first.prompt_id


def test_model_experiment_allows_only_model_to_change_and_publish_is_versioned(tmp_path: Path) -> None:
    baseline = config("qwen-vl-a")
    candidate = config("qwen-vl-b")
    experiment = ModelExperimentResult(baseline=baseline, candidates=[candidate], metrics_by_model={"qwen-vl-a": metrics(), "qwen-vl-b": metrics()})
    assert len(experiment.candidates) == 1
    changed = candidate.model_copy(update={"prompt_version": "p2"})
    with pytest.raises(ValueError, match="only model_id"):
        ModelExperimentResult(baseline=baseline, candidates=[changed], metrics_by_model={"qwen-vl-a": metrics(), "qwen-vl-b": metrics()})
    store = WorkflowConfigurationStore(tmp_path)
    assert store.publish(candidate).version == 1
    assert store.publish(candidate).version == 2
