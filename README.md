# LLM推理性能优化

> 从原理到生产环境的性能优化实战

**LLM Inference Optimization: A Practical Guide to Performance Optimization from Principles to Production Environment**

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 📖 What This Book Teaches

This book teaches you how to optimize Large Language Model (LLM) inference through practical, hands-on examples. No deep learning background required - just Python and basic programming skills.

You'll learn:

- **5 Core Techniques**: KV Cache, Request Scheduling, Quantization, Speculative Sampling, Production Deployment
- **GPU Fundamentals**: Understand how GPUs work and how to leverage them for inference
- **Complete Code Examples**: Every concept comes with working code
- **Docker Ready**: One-command setup, no environment hassles
- **Production Grade**: Real-world deployment strategies

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
- **Chapter 6: KV Cache Optimization** - Memory, caching, and hit-rate ([`chapters/chapter06-kv-cache-optimization.md`](chapters/chapter06-kv-cache-optimization.md))
- **Chapter 7: Request Scheduling** - Batching, prioritization, and tail latency ([`chapters/chapter07-request-scheduling.md`](chapters/chapter07-request-scheduling.md))
- **Chapter 8: Quantization** - INT8/INT4 trade-offs and pitfalls ([`chapters/chapter08-quantization.md`](chapters/chapter08-quantization.md))
- **Chapter 9: Speculative Sampling** - Draft/verify, acceptance, rollback ([`chapters/chapter09-speculative-sampling.md`](chapters/chapter09-speculative-sampling.md))
- **Chapter 10: Production Deployment** - Kubernetes, observability, ops ([`chapters/chapter10-production-deployment.md`](chapters/chapter10-production-deployment.md))
- **Chapter 11: Advanced Topics** - MoE, compilation, kernels, trends ([`chapters/chapter11-advanced-topics.md`](chapters/chapter11-advanced-topics.md))

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
