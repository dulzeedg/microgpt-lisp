# GPT from Scratch in Racket
## Goal: Rewriting microgpt in lisp
## Why: Understand

A pure Racket implementation of a GPT-style transformer language model inspired by Andrej Karpathy's "The most atomic way to train and run inference for a GPT in pure, dependency-free Python."

This project reimplements the entire training and inference pipeline from first principles using only Racket. There are no machine learning frameworks, tensor libraries, or automatic differentiation packages.

The goal is educational: to demonstrate how modern transformer language models work internally by building every component from scratch.

## Features

* Character-level tokenization
* Custom autograd engine
* Computational graph construction
* Reverse-mode automatic differentiation
* RMSNorm
* Multi-head self-attention
* Feed-forward MLP blocks
* Residual connections
* Adam optimizer
* Autoregressive text generation
* Model save/load functionality

## Architecture

The model follows a simplified GPT-style architecture:

Token Embeddings
→ Positional Embeddings
→ RMSNorm
→ Multi-Head Self-Attention
→ Residual Connection
→ RMSNorm
→ MLP
→ Residual Connection
→ Language Modeling Head

The implementation intentionally prioritizes clarity over performance.

## Project Structure

The implementation contains five major sections:

### Block 1: Dataset and Tokenization

* Loads names from `names.txt`
* Builds character vocabulary
* Creates token mappings
* Defines BOS (Beginning of Sequence) token

### Block 2: Autograd Engine

Defines the `Value` type:

* Stores scalar values
* Tracks gradients
* Maintains computation graph
* Implements backpropagation

Supported operations:

* Addition
* Multiplication
* Division
* Exponentiation
* Logarithm
* Exponential
* ReLU

### Block 3: Parameter Initialization

Creates model parameters:

* Token embeddings (`wte`)
* Positional embeddings (`wpe`)
* Attention matrices
* MLP matrices
* Language modeling head

Weights are initialized using a Gaussian distribution.

### Block 4: Transformer Model

Implements:

* Linear layers
* Softmax
* RMSNorm
* Multi-head attention
* Feed-forward network
* GPT forward pass

### Block 5: Training and Inference

Includes:

* Cross-entropy loss
* Adam optimization
* Training loop
* Text generation
* Model serialization

## Hyperparameters

Current defaults:

```racket
(define n-layer 1)
(define n-embd 16)
(define block-size 16)
(define n-head 4)

(define learning-rate 0.01)
(define beta1 0.85)
(define beta2 0.99)

(define num-steps 1000)
```

## Running

Create a file named:

```text
names.txt
```

Example:

```text
emma
olivia
liam
noah
ava
sophia
```

Run:

```bash
racket main.rkt
```

Training output:

```text
step 1 / 1000 | loss 3.14
step 2 / 1000 | loss 3.07
...
```

Generated samples:

```text
sample 1: emma
sample 2: olia
sample 3: sopha
```

## Saving and Loading

Save trained parameters:

```racket
(save-model "model.dat")
```

Load existing parameters:

```racket
(load-model "model.dat")
```

## Educational Goals

This project demonstrates:

* How automatic differentiation works
* How gradients propagate through a computational graph
* How transformer attention is computed
* How language models learn next-token prediction
* How Adam optimization updates parameters
* How autoregressive generation works

Every operation is represented as scalar computations, making the mechanics of training completely visible.

## Limitations

This implementation is intentionally simple.

Compared to production GPT systems:

* No GPU support
* No tensor operations
* No batching
* No LayerNorm parameters
* No mixed precision
* No optimized kernels
* Significantly slower training

The focus is correctness and understanding rather than speed.

## Inspiration

Inspired by Andrej Karpathy's minimal GPT implementation and the philosophy:

> "Everything else is just efficiency."

This project explores how much of a transformer language model can be implemented using only basic language features and a few hundred lines of code.
