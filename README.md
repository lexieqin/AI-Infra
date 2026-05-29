# AI Infra Notes

Personal knowledge base and hands-on lab notes for transitioning from Platform / Cloud Native Engineering into AI Infrastructure Engineering.

This repository documents my learning journey across:

- LLM inference systems
- GPU serving infrastructure
- KV cache and memory management
- vLLM and high-throughput serving
- Kubernetes-based AI platforms
- Agent runtimes and orchestration
- Distributed inference systems
- AI platform engineering concepts

The goal is not just to learn how to use LLMs, but to understand:

- how inference actually works
- how models are served at scale
- how GPU resources are scheduled and optimized
- how AI systems map to traditional systems and cloud-native concepts

---

# Repository Structure

```text
AI-Infra-Notes/
├── Foundations/       # Core concepts and mental models
├── Serving/           # Inference engines and serving systems
├── GPU/               # GPU architecture and memory concepts
├── Kubernetes/        # AI workloads on Kubernetes
├── Agent-Systems/     # Agents, tool calling, memory systems
├── Projects/          # Hands-on experiments and mini projects
└── Glossary/          # Acronyms and terminology
```

---

# Current Focus

## Foundations
- Training vs Inference
- Tokenization
- Transformer basics
- Streaming vs non-streaming inference
- KV Cache
- Attention mechanisms

## Serving Systems
- Ollama
- vLLM
- Continuous batching
- PagedAttention
- OpenAI-compatible APIs

## Platform Engineering for AI
- GPU scheduling
- Autoscaling
- Model routing
- Observability
- Cost optimization

---

# Notes Philosophy

Most AI learning material focuses on:
- prompt engineering
- model usage
- application demos

This repository focuses instead on:
- runtime behavior
- systems design
- memory management
- scheduling
- distributed serving
- production infrastructure

The perspective is intentionally:
> Systems / Platform / Infrastructure Engineering first.

---

# Long-Term Goal

Build a strong mental model of modern AI infrastructure systems and transition from traditional cloud-native engineering into production AI infrastructure engineering.
