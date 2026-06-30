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

Comparison baselines (such as prior sampling from operator priors without model guidance, and brute-force exhaustive search) moved out of flash-ansr in v0.6 into the standalone `srbf` package (Symbolic Regression Benchmark Framework: `pip install srbf`, https://github.com/psaegert/srbf). See the srbf repository for the available baselines and how to run them.

## Architecture (Training + Inference Workflow)
- **Data synthesis**: During training, a `symbolic_data` catalog (a generative recipe over operators, expression length, and literal/support priors) is sampled into problems by a `ProblemSource` and wrapped by `FlashANSRDataset`; each problem is an expression skeleton plus its constants and support points.
- **Pre-Encoder**: Encodes float input tensors of shape \((B, M, D)\) into their IEEE-754 bit representations, yielding \((B, M, D')\) binary tensors with batch size \(B\), support points \(M\), original variables \(D\), and bit-encoded variables \(D' = 32 \times D\).
- **Encoder**: The SetTransformer component consumes unordered support points of shape \((B, M, D')\) and produces an embedding of shape \((B, S, d)\) summarizing the dataset. Here, \(S\) is the number of seeds in the Multihead Attention based pooling and \(d\) is the internal model size.
- **Decoder**: The Transformer decoder autoregressively emits tokens for expressions in prefix order, conditioned on the encoder output and optional prompts via cross-attention.
- **Refiner**: The Refiner class optimizes the constants of predicted skeletons using one of many available methods (e.g., LM, BFGS, Nelder-Mead).
- **Orchestrator** The main `FlashANSR` class ties together preprocessing, decoding (with a chosen generation method), refinement, scoring, and output formatting for a user-friendly end-to-end experience.
- **Configs**: YAML bundles (`model`, `tokenizer`, `dataset`, `train`) keep paths portable via `load_config(..., resolve_paths=True)`. (Evaluation-run configs moved to the standalone `srbf` package in v0.6: `pip install srbf`, https://github.com/psaegert/srbf.)

### Training loop

1. Stream skeletons and support points from a `symbolic_data` catalog (via a `ProblemSource`) and build batches with `FlashANSRDataset`.
2. Encode sets with the SetTransformer; decode prefix tokens with teacher forcing.
3. Optimize cross-entropy loss; validate on a held-out synthetic catalog; checkpoint `model.yaml`, `tokenizer.yaml`, `state_dict.pt` together.

### Inference loop

1. Load a checkpoint and select a generation config (beam, softmax, MCTS).
2. Encode the provided \(X, y\) pairs; decode candidate expressions.
3. Refine constants; score candidates by $\log_{10}(\text{FVU}) + \lambda_\ell \cdot \text{length} + \lambda_c \cdot \text{n\_constants} + \lambda_L \cdot (-\log p)$; return the best expression and predictions. All diverse results are accessible.

`model.infer(X, y)` returns an `InferenceResult` that exposes this directly: the score-sorted refined `Candidate`s (`result.candidates`, best at `result.best`) plus the full `CandidateLedger` (`result.ledger`) -- the entire generation pool joined with the refined survivors, each classified `FIT_OK` / `FIT_FAILED` / `INVALID`. See [Getting started](getting_started.md) and the [API reference](api.md).