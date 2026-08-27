import csv
from pathlib import Path

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.schema import AgentDecisionLabel, ContentType, PolicyLabel
from trust_safety_agent.vector_store import PolicyVectorStore


POLICY_PATH = Path(__file__).parents[1] / "data/policies/mock_policy_v1.md"
GOLDEN_SET = Path(__file__).parents[1] / "data/golden_set/golden_set_v1.csv"


def build_agent(tmp_path: Path) -> PolicyAdjudicator:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(POLICY_PATH))
    return PolicyAdjudicator(store)


def test_reject_has_direct_policy_evidence(tmp_path: Path) -> None:
    decision = build_agent(tmp_path).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci 1:1 mirror copy handbag",
    )

    assert decision.decision == AgentDecisionLabel.REJECT
    assert decision.policy_label == PolicyLabel.COUNTERFEIT
    assert decision.matched_policy[0].policy_id == "POL-CF-001"
    assert decision.matched_policy[0].quote
    assert decision.confidence >= 0.9


def test_supported_exemption_is_approved(tmp_path: Path) -> None:
    decision = build_agent(tmp_path).adjudicate(
        case_id="C026",
        content_type=ContentType.PRODUCT,
        input_text="Protective case compatible with iPhone 15; not made by Apple",
    )

    assert decision.decision == AgentDecisionLabel.APPROVE
    assert decision.policy_label is None
    assert decision.matched_policy[0].policy_id == "POL-EX-001"


def test_conflicting_signals_require_review(tmp_path: Path) -> None:
    decision = build_agent(tmp_path).adjudicate(
        case_id="C999",
        content_type=ContentType.PRODUCT,
        input_text="Pre-owned authentic Gucci replica handbag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert decision.policy_label is None
    assert decision.confidence < 0.5


def test_unresolved_brand_reference_requires_review(tmp_path: Path) -> None:
    decision = build_agent(tmp_path).adjudicate(
        case_id="C999",
        content_type=ContentType.PRODUCT,
        input_text="Apple accessory in excellent condition",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW


def test_plain_unbranded_content_is_approved(tmp_path: Path) -> None:
    decision = build_agent(tmp_path).adjudicate(
        case_id="C050",
        content_type=ContentType.PRODUCT,
        input_text="Plain blue cotton tote bag with no logos",
    )

    assert decision.decision == AgentDecisionLabel.APPROVE


def test_reject_without_policy_evidence_requires_review(tmp_path: Path) -> None:
    agent = PolicyAdjudicator(PolicyVectorStore(tmp_path / "empty-chroma"))

    decision = agent.adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica handbag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert decision.policy_label is None


def test_golden_set_regression_baseline(tmp_path: Path) -> None:
    agent = build_agent(tmp_path)
    with GOLDEN_SET.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    mismatches = []
    for row in rows:
        decision = agent.adjudicate(
            case_id=row["case_id"],
            content_type=ContentType(row["content_type"]),
            input_text=row["input_text"],
        )
        policy = decision.policy_label.value if decision.policy_label else ""
        if decision.decision.value != row["human_decision"]:
            mismatches.append(row["case_id"])
        elif row["human_decision"] == "reject" and policy != row["expected_policy"]:
            mismatches.append(row["case_id"])

    assert mismatches == []
