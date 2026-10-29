# Changelog

All notable editorial and structural changes to this manuscript are documented in this file.

## 2026-10-29

- Put diagnosis and selection methods before trends in Chapter 2; replaced framework rankings with evidence requirements and conditioned PD selection on measured interference.
- Moved model-specific KV and DeepInfer settings into dated implementation records, retaining source provenance and explicit verification limits.
- Replaced unsupported Appendix C benchmark and customer results with experiment worksheets and ROI formulas, including a labelled hypothetical arithmetic example.
- Promoted the executable prefix-cache experiment in the reader path; updated FAQ, contribution notes and code coverage, and archived the initial launch checklist.
- Synchronized the Chapter 5 title and regenerated navigation and chapter counts. Dated publisher DOCX exports require separate regeneration and visual review.

## 2026-10-22

### Review corrections

- Corrected the Berkeley dissertation and DFlash bibliographic metadata, and added the sources referenced by manuscript citation markers.
- Added explicit citation-to-bibliography mapping with fail-fast validation during DOCX generation.
- Fixed linked-footer duplication and prefilled the TOC field with bookmarked three-level headings; final TOC page numbers still require WPS/Word pagination.
- Documented NVIDIA-only DeepInfer determinism, its required settings, and the distinction between vLLM's 128-token replay and DeepInfer's default 140-token retained suffix.
- Expanded the V4.1 discussion with CSA2, sparse indexers, Engram host tables, and mHC execution dependencies; replaced unconditional attention/MoE conclusions with workload-dependent comparisons.
- Generated a separately named reviewed DOCX; structural checks completed, visual pagination review remains pending.

### Added

- Added a DeepSeek-V4.1-Flash model-architecture section to Chapter 6 covering CED execution, Global KV and SWA KV, FP4/FP8 storage, bounded replay, split CUDA Graph execution, model-specific kernels, and TP/DP/EP deployment trade-offs.
- Added a DeepInfer section to Chapter 11 covering its DeepSeek-V4.1-Flash focus, SM100/Ascend 950 hardware boundary, specialized kernels, DSpark, SWA replay, structured output, and the service capabilities that remain outside the open-source release.
- Added publisher-formatted references in [`docs/publisher-references.md`](docs/publisher-references.md), including the official DeepInfer repository and vLLM DeepSeek-V4.1-Flash analysis.
- Added a dated Tsinghua University Press-style manuscript at `publish/LLM推理性能优化_正文_清华出版社体例_最终版_2026-10-22.docx`.

### Changed

- Updated the book metadata and manuscript title to `LLM 推理性能优化`.
- Compressed the content summary and clarified that Appendix C remains an unpublished validation draft rather than part of the formal manuscript.
- Updated the DOCX generator to include the publisher reference list, a WPS-updatable TOC field, dated output naming, front-matter metadata fields, and separate Roman/Arabic page numbering.

### Sources

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- [DeepSeek-V4.1-Flash on vLLM](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)

## 2026-09-26

### Added

- Added a 30-second, 16:9 introduction video covering the book's core path from inference economics and GPU fundamentals through KV cache, scheduling, quantization, speculative sampling, production deployment, and vLLM 0.30.0 practice.
- Added the playable MP4 at [`media/intro_video/llm-inference-optimization-rewired-style-30s.mp4`](media/intro_video/llm-inference-optimization-rewired-style-30s.mp4).
- Added the editable Canvas source at `media/intro_video/canvas_rewired_style.html`.

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
