---
id: "readme"
title: "LLM推理性能优化"
slug: "readme"
date: "2026-03-11"
type: "article"
topics:
  - "llm-inference"
  - "inference-economics"
concepts:
  - "cost-optimization"
tools: []
architecture_layer:
  - "motivation-and-economics"
learning_stage: "orientation"
optimization_axes:
  - "cost"
  - "latency"
related:
  - "chapters-chapter01-introduction"
  - "chapters-chapter02-technology-landscape"
references: []
status: "published"
display_order: 1
---
# LLM推理性能优化

> 从原理到生产环境的性能优化实战

**LLM Inference Optimization: A Practical Guide to Performance Optimization from Principles to Production Environment**

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 🧭 Reading Guide

- Main reading path: [`SUMMARY.md`](SUMMARY.md)
- Repo landing page: [`index.md`](index.md)
- Back-cover style summary: [`docs/content-summary.md`](docs/content-summary.md)
- Editorial changelog: [`CHANGELOG.md`](CHANGELOG.md)

---

## 📖 What This Book Teaches

This book teaches you how to optimize Large Language Model (LLM) inference through practical, hands-on examples. No deep learning background required - just Python and basic programming skills.

You'll learn:

- **11 Chapters in 4 Parts**: From motivation and GPU basics to production deployment and advanced topics
- **GPU Fundamentals**: Understand how GPUs work and how to leverage them for inference
- **Selected Code Examples**: The repo currently ships runnable examples for the foundational chapters, with more chapter code added incrementally
- **Docker Ready Foundations**: Environment checks and base examples are designed to be reproducible
- **Production Grade**: Real-world deployment strategies

---

## 📝 Current Editorial Status

- The manuscript is currently organized into **4 parts / 11 chapters**
- Chapter boundaries have been tightened so the main path now reads more cleanly:
  - Chapter 5 frames the problem space
  - Chapter 6 focuses on KV management and reuse
  - Chapter 7 focuses on request scheduling and budget decisions
  - Chapter 10 focuses on production deployment and runtime governance
  - Chapter 11 focuses on advanced and frontier topics
- Cross-chapter transitions, summary labels, and section lead-ins have been standardized for consistency
- Runnable code is still concentrated in the foundational chapters, with later chapter examples being added incrementally

See [`CHANGELOG.md`](CHANGELOG.md) for the latest structural and editorial updates.

---

## 🎯 Who This Is For

✅ **Perfect for you if**:
- You know Python
- You've heard of ChatGPT and want to understand how it works
- You want to run LLMs locally or in production
- You care about inference speed and cost optimization

❌ **Not for you if**:
- You're completely new to programming
- You want deep learning theory (this is practical, not theoretical)
- You want research papers (this is engineering, not academia)

---

## 📚 Table of Contents

### Chapters
- **Chapter 1: Introduction** - Why inference optimization matters ([`chapters/chapter01-introduction.md`](chapters/chapter01-introduction.md))
- **Chapter 2: Technology Landscape** - Why this problem got hard, and what changed ([`chapters/chapter02-technology-landscape.md`](chapters/chapter02-technology-landscape.md))
- **Chapter 3: GPU Basics** - GPU architecture, memory, and bandwidth ([`chapters/chapter03-gpu-basics.md`](chapters/chapter03-gpu-basics.md))
- **Chapter 4: Environment Setup** - Docker, CUDA, and sanity checks ([`chapters/chapter04-environment-setup.md`](chapters/chapter04-environment-setup.md))
- **Chapter 5: LLM Inference Basics** - What actually happens in prefill/decode ([`chapters/chapter05-llm-inference-basics.md`](chapters/chapter05-llm-inference-basics.md))
- **Chapter 6: KV Cache Optimization** - How KV state is stored, reused, and kept compact ([`chapters/chapter06-kv-cache-optimization.md`](chapters/chapter06-kv-cache-optimization.md))
- **Chapter 7: Request Scheduling** - How the scheduler makes admission, iteration, and budget decisions ([`chapters/chapter07-request-scheduling.md`](chapters/chapter07-request-scheduling.md))
- **Chapter 8: Quantization** - INT8/INT4 trade-offs and pitfalls ([`chapters/chapter08-quantization.md`](chapters/chapter08-quantization.md))
- **Chapter 9: Speculative Sampling** - Draft/verify, acceptance, rollback ([`chapters/chapter09-speculative-sampling.md`](chapters/chapter09-speculative-sampling.md))
- **Chapter 10: Production Deployment** - Deployment, observability, capacity, and cost governance in production ([`chapters/chapter10-production-deployment.md`](chapters/chapter10-production-deployment.md))
- **Chapter 11: Advanced Topics** - Agent infra, heterogeneous systems, MoE, and frontier topics ([`chapters/chapter11-advanced-topics.md`](chapters/chapter11-advanced-topics.md))

### Appendices
- **Appendix A: Tools and Resources** - A curated list of helpful tools
- **Appendix B: Troubleshooting** - Common issues and solutions
- **Appendix C: Performance Benchmarks** - Real-world numbers

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/joqk12345/llm-inference-optimization.git
cd llm-inference-optimization

# Generate word-count reports (chapter01..chapter11)
python3 scripts/wordcount.py --write

# Environment sanity check (optional)
bash code/chapter04/check_env.sh

# Run Chapter 3 code examples (Docker + NVIDIA runtime required)
cd code/chapter03
docker build -t llm-book-chapter03 .
docker run --gpus all -it llm-book-chapter03 python memory_calculator.py --help
```

Word-count outputs:
- `docs/word-counts.md`
- `docs/word-counts.json`

---

## 📊 Word Count Automation

This repo ships a GitHub Actions workflow that runs word-count stats on every push and updates the reports automatically:
- Workflow: `.github/workflows/wordcount.yml`
- Script: `scripts/wordcount.py`

If `docs/word-counts.md` / `docs/word-counts.json` changes, the workflow commits the updated files back to the same branch (and avoids infinite loops by skipping `github-actions[bot]` pushes).

---

## 📚 Changelog

Recent manuscript and structure updates are tracked in [`CHANGELOG.md`](CHANGELOG.md).

---

## 🤝 Contributing

We welcome contributions! Here are ways you can help:

### Easy Contributions
- 📝 Fix typos and grammar
- 🐛 Report bugs
- 💡 Suggest improvements
- 📖 Answer questions in issues/PRs

### Code Contributions
- 🔧 Fix bugs in code examples
- ✅ Add tests
- 🌐 Improve documentation
- 🆕 Add new examples

### Share Your Story
- 📝 Write a success story (we'll feature it!)
- 🎥 Record a video case study
- 💬 Share on social media

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Top contributors** get recognized:
- Monthly Top 10: Free Pro membership
- Quarterly Top 3: 1:1 consulting session
- Annual contributors: Your name in the book acknowledgments

---

## 📄 License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
