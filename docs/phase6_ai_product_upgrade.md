# Phase 6: AI Product Capability Upgrade

## Objective

This phase changes the project's center of gravity from a domain workflow demo
to an AI decision and evaluation lab. Trust & Safety remains the grounded use
case, while the new evidence is model orchestration, controlled tool use,
evaluation operations, protocol interoperability, and safe failure behavior.

## 1. Controlled Tool Calling Agent

Status: implemented prototype.

The model acts as a planner and chooses from typed tools. The host, rather than
the model, owns execution authority. Controls include:

- workflow-specific tool allowlist;
- Pydantic input and output validation;
- maximum four planning turns;
- duplicate-call detection;
- read/write permission separation;
- explicit confirmation for review-queue writes;
- policy-evidence requirement for reject;
- low-confidence and invalid-output fallback to human review;
- sanitized, persisted execution traces.
- versioned Skill selection for Prompt, tools, policy scope, step budget, and output contract.

The current implementation uses structured JSON planning over an
OpenAI-compatible chat endpoint. It is genuine model-selected tool use, but it
does not claim provider-native function-calling support. Narrow compatibility
normalization accepts common `action/arguments/args` response shapes only after
checking the action against the allowlist.

### Live verification

A real Bailian smoke case completed the path:

1. model loaded `SKL-KNOCKOFF-REVIEW@v1.0.0` and selected
   `classify_policy_signals`;
2. host validated the arguments;
3. deterministic tool returned a knockoff recommendation, 0.95 confidence,
   and `POL-KO-001` policy evidence;
4. model returned the final decision;
5. local evidence and confidence guardrails accepted it;
6. the run completed as `reject` rather than falling back to review.

The verified run used one tool call and 1,948 tokens. A preceding run omitted
final confidence and correctly failed closed; compatibility was then narrowed
to deriving confidence only from the successful deterministic tool output.

Earlier attempts produced different provider JSON shapes and one repeated tool
call. Those failures were preserved as design evidence: the agent initially
failed closed, then gained narrow normalization and one-turn self-correction
without weakening the tool or evidence controls.

## 2. AI Evaluation Lab

Status: implemented prototype.

The Streamlit workspace can:

- discover versioned evaluation runs;
- inspect engine, Prompt, retrieval, and dataset versions;
- launch a reproducible offline Golden Set v4 rules run;
- compare baseline and candidate metrics;
- show quality-gate regressions and resolutions;
- display Failure Taxonomy aggregation;
- compare tokens, cost, and latency when the run contains telemetry.

The lab reuses the same immutable evaluation artifacts as the command-line
runner. It does not invent live-model telemetry for older offline runs.

## 2.1 Prompt and Skill Studio

Status: implemented prototype.

The workspace exposes versioned Prompt presets for model-backed workflows and
reusable Skill definitions. A Skill is not merely a shorter prompt: it binds a
Prompt version to an allowed-tool list, policy scope, maximum planning steps,
and structured output contract. Custom versions are stored locally outside Git.
The host's evidence, confidence, permission, and schema checks remain fixed and
cannot be disabled by an edited Prompt.

For knockoff review, the built-in Skill uses `POL-KO-001`, calls the new
`classify_policy_signals` tool, and distinguishes imitation language plus a
protected-brand reference from a brand mention alone. This provides an
inspectable answer to why the Agent made the classification.

## 3. MCP Integration

Status: implemented and protocol-verified.

The project uses the official MCP Python SDK 2.x and exposes:

| Primitive | Name | Access |
| --- | --- | --- |
| Tool | `search_policy` | read-only |
| Tool | `lookup_brand` | read-only |
| Tool | `classify_policy_signals` | read-only |
| Tool | `review_product` | read-only |
| Tool | `get_review_queue` | read-only, metadata only |
| Tool | `submit_human_review` | guarded write |
| Resource | `policy://catalog` | read-only |
| Resource | `review://pending` | read-only, metadata only |
| Prompt | review-case prompt | user-controlled template |

`scripts/smoke_mcp.py` launches the MCP server as an independent stdio child
process. The official client completes the protocol handshake, discovers six
tools and two resources, and receives structured output from `search_policy`.
An optional Docker Compose profile exposes the same server over Streamable HTTP.

## Verification

- Complete unit and regression suite: 112 tests passed.
- Streamlit AppTest: eleven top-level workspaces loaded with zero exceptions.
- Evaluation Lab smoke: created a new 210-case v4 run and wrote artifacts.
- MCP stdio smoke: independent child process, six tools, two resources, and one
  successful structured tool call.
- Bailian Agent smoke: completed a grounded tool-selected reject decision.

## Claim Boundary

Implemented and supportable in a resume:

- bounded model-selected tool workflow;
- deterministic host-side guardrails;
- reproducible evaluation-run comparison;
- official MCP server/client interoperability;
- local and Docker execution;
- real Bailian Agent smoke call.

Not yet implemented or not production-validated:

- unrestricted autonomous planning;
- provider-native function calling;
- multi-agent collaboration;
- OAuth or enterprise MCP authorization;
- production database, concurrent queue, or tenant isolation;
- large-scale live-model benchmark on independently labeled business data.

## Product Significance

The new modules demonstrate a different capability from the original platform
governance internship. The domain experience supplies the problem and safety
constraints; Phase 6 demonstrates turning that problem into an AI product with
controlled agency, measurable release decisions, and reusable protocol tools.
