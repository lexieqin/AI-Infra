# AI Infra Foundations — Runtime / Inference Mental Model

> Notes for transitioning from Platform/Cloud Native Engineering → AI Infrastructure Engineering

---

# 1. What Is LLM Inference?

LLM inference is:

> The process of generating tokens step-by-step using a trained Transformer model.

Simplified flow:

```
User Prompt
    ↓
Tokenizer
    ↓
Tokens
    ↓
Embeddings
    ↓
Transformer Forward Pass
    ↓
Predict Next Token
    ↓
Append Token To Context
    ↓
Repeat Until STOP
    ↓
Return Response
```

Key insight:

# LLM inference is an iterative runtime loop.

The model:
- does NOT generate the entire response at once
- generates one token at a time

---

# 2. Tokenizer

## What is a tokenizer?

Tokenizer converts human-readable text into token IDs.

Example:

```
"Explain Kubernetes briefly"
    ↓
["Explain", " Kubernetes", " briefly"]
    ↓
[840, 20772, 26753]
```

The reverse operation is called decoding:

```
[840, 20772, 26753]
    ↓
"Explain Kubernetes briefly"
```

---

# 3. Tokens

## Tokens are NOT words.

A token may be:
- part of a word
- a full word
- punctuation
- whitespace + word
- Chinese character

Examples:

text "Kubernetes" → ["K", "ubernetes"] 

text "Hello world" → ["Hello", " world"] 

Important:

These are conceptual examples.

Actual token splits depend on:
- the specific tokenizer
- the vocabulary
- the model family

---

# Why not use characters?

Character-level sequences become extremely long.

This would make:
- attention expensive
- inference slow
- context windows impractical

---

# Why not use whole words?

Vocabulary would explode.

Problems:
- infinite words
- typos
- code
- multilingual text
- new words

---

# Tokenization is a compromise

Tokenizer:
```
infinite text
    ↓
finite vocabulary
```

Key idea:
# Tokenization = compression + standardization

---

# 4. Embeddings

After tokenization:

```
token IDs 
```

are converted into:

```
dense vectors 
```

Example:

```
20772
    ↓
[0.12, -0.55, 0.91, ...]

```
These vectors represent semantic meaning.

Transformer models operate on vectors, not raw token IDs.

---

# Why embeddings exist

Token IDs themselves have no semantic meaning.

Example:

```
20772 
```

does NOT inherently mean:
- Kubernetes
- cloud-native
- containers

Embeddings place tokens into:

# semantic vector space

Example intuition:

```
Kubernetes ↔ Docker
(close vectors)

Kubernetes ↔ banana
(far vectors)
```

---
# 5. Transformer Forward Pass

## What is a forward pass?

A forward pass is:

> Running the current token context through the Transformer model to predict the next token probabilities.

Simplified flow:
```
Token IDs
    ↓
Embeddings
    ↓
Transformer Layers
    ↓
Attention Computation
    ↓
Feed Forward Networks
    ↓
Logits
    ↓
Probability Distribution
```

Output:

```
possible next tokens + probabilities
```

---

# Example

Input:

```
Explain Kubernetes briefly 
```

Model predicts this kind of probability distribution:

| Token | Probability |
|---|---|
| "K" | 20% |
| "It" | 10% |
| "The" | 8% |

These probability numbers are illustrative, not real model outputs.

Model samples one token.

Example:

```
"K"  
```

Then appends it to context:

```
Explain Kubernetes briefly K 
```

Then repeats.

---

# Key Insight

# Inference is iterative prediction over a growing context.

---

# 6. Attention

## What is attention?

Attention determines:

> Which previous tokens are important when generating the next token.

---

# Important clarification

“Previous tokens” means:

# The ENTIRE current context window.

Including:
- user prompt
- system prompt
- conversation history
- previously generated output tokens

---

# Example

Current context:

```
[Explain]
[Kubernetes]
[briefly]
[K]
[ubernetes]
```

When generating the next token:

attention can see ALL previous tokens.

---

# Attention is NOT equal weighting

The model dynamically decides:
- which tokens matter more
- which matter less

---

# Example intuition

Prompt:

```
The capital of France is
```

When generating next token:

attention focuses heavily on:

```
France
``` 

instead of:

```
The
```

---

# 7. QKV (Query / Key / Value)

Attention is implemented using:

# QKV mechanism

Each token generates:

| Component | Meaning |
|---|---|
| Query (Q) | What information am I looking for? |
| Key (K) | What information do I contain? |
| Value (V) | What is my actual content? |

---

# Attention / QKV Flow

```
Current Token
    ↓
Generate Query Vector

Previous Tokens
    ↓
Generate Key Vectors
    ↓
Generate Value Vectors

Query + Keys
    ↓
Attention Score Computation
    ↓
Attention Weights

Attention Weights + Values
    ↓
Weighted Combination
    ↓
Context-Aware Representation
```

---

# Search-engine analogy

Query:
- what you're searching for

Key:
- searchable metadata

Value:
- actual document content

---

# Important insight

Q/K/V semantics are NOT hardcoded.

They are:
# learned during training

The model statistically learns patterns from massive datasets.

---

# Example

Prompt:

```
The capital of France is 
``` 

The model learns through training that:
- "capital" often attends to countries
- "France" is highly relevant to predicting the next token

This behavior emerges statistically from training data.

---

# 8. KV Cache

## Problem

Without caching:

every generated token would require recomputing attention over all previous tokens.

Very expensive.

---

# Solution

Store intermediate attention states.

This is called:

# KV Cache

Full name:
- Key Cache
- Value Cache

---

# Why only K/V are cached?

Because:
- previous token K/V values do not change
- they can be reused

Q is different.

Each new token only needs:

# its own current Query

to attend over:

# all historical Keys and Values

Historical Q values are not reused in later decode steps, so caching Q does not help.

---

# KV Cache Runtime Flow

```
Current Context
    ↓
Reuse Existing KV Cache
    ↓
Compute New Token Q/K/V
    ↓
Attention
    ↓
Generate Next Token
    ↓
Append New K/V To Cache
    ↓
Append Token To Context
    ↓
Repeat
```

---

# Key Insight

# Context grows continuously during inference.

Longer context means:
- larger KV cache
- more memory usage
- higher latency

---

# Multi-Head Attention Variants

KV cache size depends heavily on:

# how many K/V heads the model stores

Common variants:

| Variant | Idea | KV Cache Impact |
|---|---|---|
| MHA | Each query head has its own K/V heads | largest KV cache |
| MQA | Many query heads share one K/V head | much smaller KV cache |
| GQA | Groups of query heads share K/V heads | middle ground |

GQA is common in modern LLMs such as Llama-family models.

Key idea:

# Sharing K/V heads reduces KV cache memory.

This improves:
- memory efficiency
- max batch size
- decode throughput

---

# 9. Streaming

## Why streaming exists

Without streaming:
- users wait for the full response

With streaming:
- tokens are sent incrementally

Example:

```
JSON
{"response":"K"}
{"response":"ubernetes"}
{"response":" is"}
```

---

# Streaming Flow

```
Generate Token
    ↓
Decode Token
    ↓
Send Token To Client
    ↓
Append Token To Context
    ↓
Repeat
```

---

# Important insight

Streaming does NOT change inference.

It only changes:
# how results are delivered to users.

---

# 10. Sampling

LLMs are probabilistic systems.

The model predicts:
- probability distributions
- not deterministic outputs

---

# Common sampling methods

| Method | Description |
|---|---|
| Greedy | Always choose highest probability |
| Temperature | Control randomness |
| Top-k | Sample from top K tokens |
| Top-p | Sample from cumulative probability |

---

# Why same prompt can produce different outputs

Because sampling introduces randomness.

---

# 11. GPU Fundamentals

## Why GPUs are used for AI

Transformers are dominated by:

# matrix multiplication

Example:

```
A × B
``` 

---

# GPU strengths

GPUs are optimized for:

# massive parallel computation

Originally built for:
- graphics rendering
- pixel processing

---

# GPU vs CPU

```
CPU
    ↓
Few Powerful Cores
    ↓
Complex Logic / Branching / Sequential Tasks


GPU
    ↓
Thousands Of Small Cores
    ↓
Massive Parallel Math
    ↓
Matrix Multiplication
    ↓
High Throughput
```

---

# Why Transformers fit GPUs perfectly

Transformer workloads are:
- highly parallel
- mathematically uniform
- matrix-heavy

---

# Important clarification

LLM workloads are NOT “simple math”.

They are:

# massively parallelizable math

The key advantage of GPUs is:
- huge parallelism
- extremely high throughput

not:
- strong single-thread logic

---

# Why AI exploded with GPUs

Deep learning workloads turned out to be:
- huge matrix multiplications
- massively parallelizable

GPUs happened to be perfect for this.

---

# CUDA

NVIDIA later introduced:

# CUDA (Compute Unified Device Architecture)

CUDA allows developers to:
- program GPUs directly
- use GPUs for general-purpose computing

Modern AI frameworks:
- PyTorch
- TensorFlow

heavily rely on CUDA.

---

# 12. Quantization

## Why quantization matters

LLM inference is often limited by:
- GPU memory capacity
- GPU memory bandwidth

Quantization reduces the number of bits used to store model weights.

---

# Common numeric formats

| Format | Common Use |
|---|---|
| FP32 | training baseline / high precision |
| FP16 | common inference format |
| BF16 | common training + inference format |
| FP8 | newer high-throughput inference/training format |
| INT8 | smaller, faster inference |
| INT4 | much smaller model weights |

---

# Why it helps

Smaller weights mean:
- lower VRAM usage
- less memory bandwidth pressure
- potentially higher throughput
- ability to run larger models on the same GPU

---

# Tradeoff

Quantization can reduce quality if done poorly.

Common techniques:
- GPTQ
- AWQ
- smooth quantization methods

Key idea:

# Quantization trades precision for efficiency.

---

# 13. Ollama vs vLLM

Both are:

# inference serving systems

But optimized for different goals.

---

# Ollama

Focus:
- local development
- simplicity
- developer experience

Analogy:

```
Docker Desktop for LLMs  
```

---

# vLLM

Focus:
- production serving
- GPU efficiency
- throughput optimization

Analogy:

```
high-performance inference runtime
``` 

---

# Key difference

| Ollama | vLLM |
|---|---|
| Local-friendly | Datacenter-oriented |
| Easy UX | GPU optimization |
| Simple runtime | Advanced scheduling |

---

# 14. Prefill vs Decode

LLM inference has two very different phases:

# prefill

and:

# decode

This is one of the most important runtime distinctions.

---

# Prefill phase

Prefill processes the input prompt.

```
Prompt Tokens
    ↓
Process Many Tokens In Parallel
    ↓
Build Initial KV Cache
    ↓
Generate First Token
```

Prefill characteristics:
- processes the whole prompt at once
- uses large matrix multiplications
- tends to be compute-bound
- strongly affects first-token latency

Compute-bound means:

# GPU math throughput is the main bottleneck.

---

# Decode phase

Decode generates output tokens one at a time.

```
Current Token
    ↓
Use Existing KV Cache
    ↓
Generate Next Token
    ↓
Append New K/V
    ↓
Repeat
```

Decode characteristics:
- one token step at a time
- repeatedly reads KV cache from GPU memory
- tends to be memory-bound
- strongly affects output token latency

Memory-bound means:

# GPU memory bandwidth is the main bottleneck.

---

# Why this matters

Different phases need different optimizations:

| Phase | Main Bottleneck | Common Focus |
|---|---|---|
| Prefill | compute | batching long prompts |
| Decode | memory bandwidth | KV cache efficiency + scheduling |

This is why inference systems often optimize:
- prefill throughput
- decode throughput
- prefill/decode scheduling separately

---

# 15. Inference Performance Metrics

Production inference needs quantitative language.

Important metrics:

| Metric | Meaning | Mostly Affected By |
|---|---|---|
| TTFT | time to first token | prefill |
| TPOT | time per output token | decode |
| ITL | inter-token latency | decode |
| Throughput | tokens or requests per second | batching + GPU utilization |

---

# TTFT

TTFT means:

# Time To First Token

It measures how long the user waits before seeing the first streamed token.

TTFT is heavily affected by:
- prompt length
- prefill scheduling
- queueing delay
- model size

---

# TPOT / ITL

TPOT means:

# Time Per Output Token

ITL means:

# Inter-Token Latency

They measure how quickly tokens appear after generation starts.

They are heavily affected by:
- decode efficiency
- KV cache size
- memory bandwidth
- continuous batching

---

# Latency vs throughput

There is usually a tradeoff:

| Optimize For | Effect |
|---|---|
| Lower latency | smaller batches, less waiting |
| Higher throughput | larger batches, better GPU utilization |

Key idea:

# Serving is scheduling under constraints.

---

# 16. Batching

## Why batching matters

GPUs dislike tiny isolated workloads.

Better:
- large parallel computation

---

# Traditional batching

```
Incoming Requests
    ↓
Form Static Batch
    ↓
Run Batch On GPU
    ↓
Short Requests Finish Early
    ↓
GPU Slots Become Idle
    ↓
Must Wait For Longest Request
```

Problem:
- short requests finish early
- GPU slots become idle

---

# 17. Continuous Batching

vLLM introduced:

# Continuous Batching

---

# Core idea

Instead of:
- fixed immutable batches

Use:
- dynamically changing batches

---

# Continuous Batching Flow

```
Incoming Requests
    ↓
Dynamic Batch Scheduler
    ↓
Run Token Step On GPU
    ↓
Some Requests Finish
    ↓
Immediately Insert New Requests
    ↓
Keep GPU Fully Utilized
    ↓
Repeat
```

---

# Key insight

Continuous batching is:

# token-step scheduling

not:
# request-level scheduling

More precisely:

# continuous batching mainly schedules decode token steps.

Prefill can also be batched, but decode is where token-by-token scheduling becomes essential.

---

# OS analogy

Traditional batching:

```
run fixed jobs
``` 

Continuous batching:

```
modern scheduler
``` 

with:
- dynamic insertion
- resource reuse
- time-slice-like execution

---

# 18. Memory Fragmentation

## Problem

KV cache sizes are dynamic.

Requests:
- start at different times
- finish at different times
- have unpredictable lengths

This causes:
# GPU memory fragmentation

---

# Fragmentation Flow

```
|AA|BBBB|CCC|
    ↓

Request B Finishes

|AA|____|CCC|
    ↓

Fragmented Free Space
    ↓

Large Request Cannot Fit
```

---

# 19. PagedAttention

vLLM solves fragmentation using:

# PagedAttention

---

# Core idea

Instead of:

```
one request = one large memory block
``` 

Use:

```
request = many small pages
``` 

---

# PagedAttention Flow

```
KV Cache
    ↓
Split Into Small Pages
    ↓
Pages Stored Non-Contiguously
    ↓
Logical KV Blocks Map To Physical Pages
    ↓
Reduced Fragmentation
    ↓
Enables Continuous Batching
```

---

# OS analogy

PagedAttention is essentially:

# virtual memory for KV cache

---

# Benefits

- better memory utilization
- reduced fragmentation
- dynamic request scheduling
- enables continuous batching
- prefix sharing across requests with the same prompt prefix

---

# Prefix sharing

Some requests share the same prefix.

Examples:
- same system prompt
- same few-shot examples
- same chat template

PagedAttention can let requests share physical KV pages for the shared prefix.

Key idea:

# identical prefixes do not need duplicate KV cache memory.

---

# 20. AI Inference Servers ≈ GPU Operating Systems

Modern inference servers increasingly behave like:

# GPU operating systems

---

# Inference Server Responsibilities

```
Inference Server
    ↓
Request Scheduling
    ↓
KV Cache Management
    ↓
GPU Memory Management
    ↓
Continuous Batching
    ↓
Streaming Responses
    ↓
GPU Utilization Optimization
```

---

# They manage:

| AI Runtime | OS Analogy |
|---|---|
| KV cache | process memory |
| batch slots | CPU scheduling |
| GPU VRAM | RAM |
| token generation steps | time slices |
| request state | process state |

---

# Key responsibilities

Inference servers now handle:
- scheduling
- memory management
- cache management
- throughput optimization
- resource isolation
- GPU utilization

---

# 21. Inference On Kubernetes

Kubernetes is commonly used to run inference services, but GPUs behave differently from CPU and memory.

---

# GPU scheduling basics

Kubernetes discovers NVIDIA GPUs through:

# NVIDIA device plugin

The device plugin advertises GPUs as extended resources:

```
nvidia.com/gpu: 1
```

Pods request GPUs explicitly.

The scheduler then places pods on nodes with available GPU capacity.

---

# GPU Operator

NVIDIA GPU Operator helps manage the GPU software stack on Kubernetes:
- drivers
- container runtime integration
- device plugin
- DCGM monitoring
- MIG configuration

Key idea:

# device plugin exposes GPUs; GPU Operator manages the GPU node stack.

---

# Why GPUs cannot be oversold like CPUs

CPU can be time-shared naturally.

GPU memory cannot be safely oversold in the same way.

Problems:
- model weights must fit in VRAM
- KV cache grows dynamically
- GPU OOM usually kills work
- memory bandwidth is a hard bottleneck

So GPU scheduling often needs stricter capacity planning.

---

# GPU sharing options

| Option | What It Does | Best For |
|---|---|---|
| MIG | hardware partitions one GPU into isolated slices | stronger isolation |
| Time-slicing | multiple pods take turns on the same GPU | bursty/light workloads |
| MPS | CUDA-level multi-process sharing | better concurrent kernel execution |

Tradeoff:

# sharing improves utilization but weakens isolation.

---

# Platform serving layer

Common Kubernetes-native serving layers:
- KServe
- Ray Serve
- Knative

They help with:
- deployment
- autoscaling
- routing
- revisions
- model serving lifecycle

But the low-level GPU efficiency usually still depends on:
- vLLM
- TensorRT-LLM
- TGI
- SGLang

Platform layer:

# manages services and scale.

Inference runtime:

# manages tokens, memory, and GPU execution.

---

# 22. Core Mental Model

The most important takeaway:

# LLM inference is a systems problem.

---

# End-to-End Runtime Pipeline

```
Raw Text
    ↓
Tokenizer
    ↓
Tokens
    ↓
Embeddings
    ↓
Prefill
    ↓
Build Initial KV Cache
    ↓
First Token
    ↓
Decode Loop
    ↓
Current Q Attends To Historical K/V
    ↓
Next Token Probability
    ↓
Sampling
    ↓
Streaming
    ↓
Client Response
    ↓
Append To Context
    ↓
Repeat
```

---

# Final Insight

AI Infrastructure is increasingly becoming:

# distributed GPU runtime systems engineering

Core concerns:
- GPU utilization
- scheduling
- memory management
- KV cache efficiency
- quantization
- serving systems
- Kubernetes GPU orchestration
- distributed inference
- runtime optimization

This is why:
- platform engineers
- cloud-native engineers
- distributed systems engineers

often transition very successfully into:

# AI Infrastructure Engineering.
