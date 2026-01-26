# Frequently Asked Questions

## General Questions

### Is this book free to read?

Yes! **90% of the content is completely free**:
- All chapter text (100% free)
- 18 basic videos (~4 hours, free)
- Complete code repository (free)

The remaining 10% (advanced topics) is available through our [Pro membership](https://TODO).

### Do I need a GPU to follow along?

Not necessarily! You can learn the concepts without a GPU. However, to run the code examples:
- **Minimum**: Any NVIDIA GPU with 8+ GB VRAM
- **Recommended**: RTX 4090 (24 GB) or better
- **Alternative**: Use cloud GPU services (Lambda Labs, RunPod, etc.)

### Do I need deep learning experience?

No! This book is designed for developers who:
- Know Python
- Have heard of ChatGPT/LLMs
- Want to learn practical optimization techniques

We don't assume any deep learning background.

### How long does it take to complete?

At **10 hours/week**, you can complete the book in about **5 months**:
- Reading: ~6 hours/week
- Hands-on code: ~3 hours/week
- Community participation: ~1 hour/week

## Technical Questions

### What vLLM version does this book use?

We test all code with **vLLM v0.6.0**. Other versions may work but aren't guaranteed.

### Can I use this for other models besides Llama?

Yes! The techniques apply to any transformer-based LLM:
- Llama (2, 3, etc.)
- Mistral/Mixtral
- Falcon
- Qwen
- And many more

### What about multimodal models?

Chapter 9 covers multimodal model inference optimization. The core techniques (Chapters 4-7) apply to all models.

### Will this work with AMD GPUs or Apple Silicon?

The focus is on NVIDIA GPUs (CUDA). However, the concepts apply to any hardware. We don't provide AMD/M1 specific examples.

## Progression Questions

### What should I read first?

Read in order! Each chapter builds on the previous:
1. Chapter 1: Introduction
2. **Chapter 2: GPU Basics** ← Start here if you're new to GPUs
3. Chapter 3: Environment Setup
4. Chapters 4-7: Core techniques (KV Cache, Scheduling, Quantization, Speculative Sampling)
5. Chapter 8: Production Deployment

### Can I skip ahead?

Technically yes, but we don't recommend it. Chapters are designed to be read sequentially.

### I'm stuck on Chapter X. What should I do?

1. Re-read the chapter carefully
2. Run the code examples
3. Ask in the [Discord community](https://discord.gg/TODO)
4. Open a GitHub Issue with your question

## Community Questions

### How do I join the Discord?

[Click here to join](https://discord.gg/TODO). It's free and open to everyone!

### What are the Office Hours?

Every **Wednesday and Friday, 20:00-21:00 UTC** in Discord #office-hour.

### How can I contribute?

See our [Contributing Guide](../CONTRIBUTING.md). Ways to help:
- Fix typos
- Report bugs
- Answer questions
- Share success stories
- Write code examples

## Business Questions

### Why is some content paid?

To sustain the project:
- Free content covers everything you need
- Paid content goes deeper (advanced topics)
- Paid memberships fund continued development

### Can I get a refund?

Yes! If you're not satisfied with the Pro membership, contact us within 30 days for a full refund.

### Do you offer discounts?

Yes! Available for:
- Students (50% off)
- Open source contributors (free)
- Bulk purchases (contact us)

## Still Have Questions?

- **GitHub Issues**: [Open a question](https://github.com/joqk12345/llm-inference-optimization/issues/new?template=question.md)
- **Discord**: Join and ask in #questions
- **Email**: TODO

---

**Last updated**: 2025-01-26
