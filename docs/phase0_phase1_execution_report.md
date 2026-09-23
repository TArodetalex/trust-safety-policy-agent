# Phase 0 and Phase 1 Execution Report

## Scope

This record separates implemented and verified behavior from planned work. It
is intended to support project review, demonstration, and resume claims.

## Phase 0: Complete

- Created an isolated Python virtual environment and installed the project and
  development dependencies.
- Preserved the frozen Day 4-7 baselines while making baseline verification
  portable across LF and CRLF checkouts.
- Added Shadow Evaluation v2 so comparison metrics use the production decision
  as the reference rather than treating shadow output as ground truth.
- Verified the complete local suite after Phase 1: `82 passed`.
- Rebuilt the Docker image, started the service, and confirmed a healthy
  container with an HTTP 200 health response.
- Verified the Streamlit application from inside the Linux container, including
  all four tabs and the Docker `/app` project path.

## Bailian API Line: Complete

- Loads the key at runtime from an ignored `.env` file or environment variable.
- Provides a public `.env.example` with no credential.
- Supports OpenAI-compatible structured output, JSON fallback parsing, model
  fallback, redacted provider errors, and request/token metadata.
- Includes a one-call smoke-test script that never prints the key.
- Completed a real Windows call and a real Docker call against Bailian. The
  tested fallback path moved from an invalid primary-model JSON response to a
  successful backup-model response.
- Secret scans found no Bailian key in tracked files or in workspace files
  outside `.env`.

## Phase 1: Complete

- Added an independent, strict `ProductReviewInput`/`ProductReviewResult` data
  contract rather than overloading the original case-decision schema.
- Added a versioned Controlled Brand Library with canonical names, aliases,
  region, policy category, status, ambiguity flags, and notes.
- Added boundary-safe candidate recall across seller brand, title, description,
  OCR text, and supplied visual marks.
- Added deterministic context validation for explicit counterfeit language,
  compatibility, second-hand resale, common-word brands, and unknown brands.
- Kept brand matching separate from infringement judgment. A brand hit alone
  routes to review rather than causing an automatic rejection.
- Added policy retrieval, auditable evidence, exemptions, decision confidence,
  and reviewer checkpoints.
- Added focused tests for library loading, short aliases, counterfeit rejection,
  compatibility approval, ambiguous brands, image-evidence gaps, and strict
  input validation.
- Added a Chinese-first Product IPR workspace and localized the existing
  Streamlit controls while preserving English technical identifiers.

## Verified Demo Path

The default Product IPR example (`PR-001`) recalls `Gucci`, detects explicit
`1:1` and `mirror copy` language, retrieves `POL-CF-001`, and recommends
`reject` with a high confidence label and a reviewer checkpoint. This exact
path was exercised in the Docker-hosted browser UI.

## Current Boundary

Phase 1 accepts OCR text and visual marks as explicit inputs but does not yet
extract them automatically inside the Product IPR pipeline. A submitted image
without reliable visual evidence fails closed to `manual_review`. Automated
multimodal extraction and calibrated evaluation belong to the next phase and
must not be claimed as completed behavior.
