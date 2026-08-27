# Adjudication Engines

## Scope

The application supports two explicit engines:

- `Multimodal LLM`: real OpenAI-compatible inference over text and images.
- `Offline rules`: the deterministic Day 3 text baseline.

## Decision Flow

1. Retrieve relevant Policy KB chunks.
2. Send case text, images, and policy context to the configured vision model.
3. Parse the model response into a strict local draft schema.
4. Resolve every cited chunk ID against the local Policy KB.
5. Require the cited chunk taxonomy to match the proposed violation label.
6. Route low-confidence, malformed, or unsupported results to `need_review`.
7. Serialize the result through the versioned `AgentDecision` contract.

## Safety Invariants

- A rejection always includes `policy_label` and directly quoted policy evidence.
- Policy quotes are copied locally; model-authored quotes are never trusted.
- An approval never declares a violation label.
- Weak, conflicting, or unsupported evidence is not forced into a binary result.
- Images are sent only to the API endpoint configured by the operator.
- The adjudicator does not use Golden Set labels at runtime.

## Baseline Limitations

Image understanding depends on the configured provider and model. The system
does not verify authenticity or authorization documents, and visual similarity
or a visible logo alone must not be treated as proof of a violation. The offline
engine does not inspect images.
