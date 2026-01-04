# Concepts & Architecture

## Core idea: Simulation-Based Inference for Symbolic Regression
- We train on millions of **synthetic equations** sampled from a prior over operators and expression depth. Each equation generates a **simulated dataset** by drawing support points.
- The model learns a **likelihood over expressions given data**: the encoder ingests sets of \(X, y\) pairs and the decoder generates prefix expressions that explain them.
- At inference, we reuse this learned conditional distribution to solve new symbolic regression tasks **without gradient-based finetuning**. *Just decoding and constant refinement.*
- This amortized approach turns symbolic regression into a language-like inference problem: data → tokens → refined expression.

## Generation methods (how we search the expression space)
- **Softmax sampling**: stochastic draws from the model distribution over next tokens (via top-k / top-p sampling).
- **Beam search**: deterministic, width-limited search with simplified-expression deduplication of completed beams.
- **MCTS**: tree search with learned policy/value signals; better exploration when local decisions are ambiguous (highly experimental).

As baselines, we also implement:

- **Prior sampling**: samples from operator priors without model guidance.
- **Brute-force**: exhaustive search up to a certain expression size.

## Architecture (Training + Inference Workflow)
- **Data synthesis**: During training, a `SkeletonPool` samples expressions, their constants, and data to use for training from a configured prior distribution.
- **Pre-Encoder**: Encodes float input tensors of shape \((B, M, D)\) into their IEEE-754 bit representations, yielding \((B, M, D')\) binary tensors with batch size \(B\), support points \(M\), original variables \(D\), and bit-encoded variables \(D' = 32 \times D\).
- **Encoder**: The SetTransformer component consumes unordered support points of shape \((B, M, D')\) and produces an embedding of shape \((B, S, d)\) summarizing the dataset. Here, \(S\) is the number of seeds in the Multihead Attention based pooling and \(d\) is the internal model size.
- **Decoder**: The Transformer decoder autoregressively emits tokens for expressions in prefix order, conditioned on the encoder output and optional prompts via cross-attention.
- **Refiner**: The Refiner class optimizes the constants of predicted skeletons using one of many available methods (e.g., LM, BFGS, Nelder-Mead).
- **Orchestrator** The main `FlashANSR` class ties together preprocessing, decoding (with a chosen generation method), refinement, scoring, and output formatting for a user-friendly end-to-end experience.
- **Configs**: YAML bundles (`model`, `tokenizer`, `dataset`, `train`, `evaluation run`) keep paths portable via `load_config(..., resolve_paths=True)`.

### Training loop

1. Sample skeletons and support points to build batches via `FlashANSRDataset`.
2. Encode sets with the SetTransformer; decode prefix tokens with teacher forcing.
3. Optimize cross-entropy loss; validate on held-out synthetic pools; checkpoint `model.yaml`, `tokenizer.yaml`, `state_dict.pt` together.

### Inference loop

1. Load a checkpoint and select a generation config (beam, softmax, MCTS).
2. Encode the provided \(X, y\) pairs; decode candidate expressions.
3. Refine constants; score candidates by \(\log_{10}(\text{FVU}) + \text{parsimony penalty}\); return the best expression and predictions. All diverse results are accessible.