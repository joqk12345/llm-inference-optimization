# Changelog

All notable editorial and structural changes to this manuscript are documented in this file.

## 2026-04-26

### Added

- Added [`docs/book-longevity-review.md`](docs/book-longevity-review.md), a chapter-by-chapter review and migration plan for keeping the book useful as models, hardware, and runtimes change.
- Added [ADR-0001](docs/adr/0001-stable-core-and-versioned-adapters.md), documenting the decision to separate a stable conceptual core from versioned runtime, hardware, model, and deployment adapters.

### Changed

- Fixed the Chapter 6 numbering gap by renumbering the practical comparison and prefix-caching sections to 6.7 and 6.8.
- Replaced production `latest` image examples with digest placeholders that make validation and rollback explicit.
- Repaired repository-local absolute links and updated generated knowledge-graph metadata.

### Verification

- `npm run knowledge:lint`
- `npm run knowledge:lint-citations`
- `npm run docs:build`
- `git diff --check`

## 2026-04-17

### Changed

- Tightened chapter boundaries across the main path:
  - Chapter 5 now frames the inference problem space instead of pre-consuming Chapter 6 and 7 details
  - Chapter 6 now centers on KV management, block organization, fragmentation control, and prefix reuse
  - Chapter 7 now centers on scheduler responsibilities, iteration-level decisions, and budget allocation
  - Chapter 10 now centers on production deployment and runtime governance
  - Chapter 11 now centers on advanced and frontier topics rather than general production concerns
- Added explicit "what this chapter answers / does not answer" scope statements to Chapters 6, 7, 10, and 11
- Reworked chapter-end transitions so the reading flow between Chapters 2-11 is clearer and more consistent
- Reduced overlap between Chapter 7.3 and 7.4 by separating principle-level explanation from implementation-level details
- Standardized summary and lead-in formatting across chapters:
  - unified `关键要点：`
  - unified `指标口径`
  - unified `核心洞察`
  - unified full-width punctuation for bold lead-in labels
- Updated navigation-facing docs to match the revised manuscript structure:
  - `README.md`
  - `index.md`
  - `docs/content-summary.md`
  - `docs/word-counts.md`
  - `docs/word-counts.json`

### Notes

- Runnable code examples remain concentrated in the foundational chapters.
- Later chapters continue to include explanatory code blocks, but not every chapter currently ships full runnable companion code.
