# Data Contract

`src/trust_safety_agent/schema.py` is the authoritative contract.

Schema `1.1.0` adds evaluation routing, tags, slices, gates, and dataset
fingerprints. Schema `1.2.0` adds reproducible project-local image assets and
blind splits while retaining support for older Golden Set rows.

## Golden Set

Golden Set rows contain human-authored truth only. Agent predictions, correctness
flags, and error classifications are written to evaluation run outputs instead of
being mixed into the source dataset.

Required columns:

- `schema_version`
- `dataset_version`
- `case_id`
- `content_type`
- `input_text`
- `image_url`
- `image_path`
- `human_decision`
- `expected_policy`
- `expected_exemption`
- `risk_level`
- `is_boundary_case`
- `human_reason`
- `source`
- `split`
- `expected_route`
- `tags`

Rules:

- A rejected case must have exactly one `expected_policy` and no exemption.
- An approved case cannot have an `expected_policy`.
- An approved exemption case must be marked as a boundary case.
- `expected_route` is `auto_decide` or `human_review`.
- `tags` is a pipe-delimited list of stable lowercase slice labels in CSV.
- Exactly one of `image_url` or `image_path` may be set. Both are empty for
  text-only cases.
- `image_path` must resolve under `data/golden_set/assets/` and may not contain
  parent-directory traversal.
- `split` is `development`, `regression`, or `blind`. Blind cases must not be
  exposed during rule, prompt, or retrieval tuning.

## Dataset Versions

- `v1.0.0` (schema `1.0.0`): 50 balanced approve/reject cases retained for
  regression.
- `v2.0.0` (schema `1.1.0`): 70 cases, including 10 cases where the presented evidence is
  intentionally insufficient and the expected route is human review.
- `v3.0.0` (schema `1.2.0`): 150 cases, including 40 blind cases and 30
  controlled multimodal cases. Its quota contract is stored in
  `data/golden_set/plans/golden_set_v3_plan.json`.
- `v4.0.0` (schema `1.2.0`): 210 cases, including 60 blind cases and 90
  controlled multimodal cases. The 60 new Day 7 cases cover explicit visual
  violations, five exemptions, clean generic controls, low contrast, rotation,
  small text, and occlusion. Its quota contract is stored in
  `data/golden_set/plans/golden_set_v4_plan.json`.

The human decision remains the final adjudicated truth. `expected_route`
describes whether the agent has enough submitted evidence to automate that
truth. This keeps abstention quality separate from binary classification.

Annotation identities, review disagreements, and image licensing metadata are
kept in separate audit ledgers. They are governance records, not model inputs
or Golden Set truth fields.

## Label Taxonomy

Violation policies:

- `counterfeit`
- `knockoff`
- `trademark_misuse`
- `shop_impersonation`
- `risky_query`

Exemptions:

- `compatibility`
- `second_hand`
- `meaningful_word`
- `incidental_exposure`
- `co_brand`

## Metric Denominators

- Strict accuracy: correct decision and policy / all cases. Expected reviews
  remain unresolved and therefore do not inflate this number.
- Auto accuracy: correct decision and policy / automatically decided cases.
- Coverage: automatically decided cases / all cases.
- Route accuracy: cases correctly routed to auto decision or human review / all
  cases.
- Policy accuracy: correct policy labels / auto-rejected human-reject cases.
- False reject rate: human-approved cases predicted as reject / all
  human-approved cases.
- False approve rate: human-rejected cases predicted as approve / all
  human-rejected cases.
- Review rate: `need_review` predictions / all cases.
- Reject recall uses all human-rejected cases as its denominator, so abstention
  reduces recall rather than being hidden.
- Accuracy, auto accuracy, coverage, route accuracy, precision, recall, and
  review rate must be reported together.

## Error Buckets

- `false_reject`
- `false_approve`
- `wrong_policy`
- `retrieval_miss`
- `invalid_output`
- `unexpected_review`
- `unexpected_auto_decision`
- `unsafe_auto_decision`
