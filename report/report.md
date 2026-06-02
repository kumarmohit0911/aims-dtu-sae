# Do Sparse Autoencoder Features Drift After Narrow-Domain Fine-Tuning?

## Abstract

This project studies whether narrow-domain fine-tuning changes only the internal
features one would expect, or whether it also shifts apparently unrelated
features. I use `EleutherAI/pythia-160m` as the base language model, collect
middle-layer MLP activations, and train a sparse autoencoder (SAE) to decompose
those dense activations into sparse features. I then fine-tune the base model on
Python code, train a second SAE on the same layer, and compare the learned
feature dictionaries by decoder-vector similarity, activation-frequency shift,
mean activation shift, and qualitative top-token inspection. The expected result
is that code-aligned features such as indentation, punctuation, imports,
function definitions, and comments become more frequent or more sharply
represented, while the main interpretability risk is that unrelated prose
features may also drift because fine-tuning changes the model's shared residual
stream geometry.

## 1. Motivation

Language-model activations are dense and superposed: individual neurons rarely
correspond cleanly to individual human concepts. Mechanistic interpretability
tries to recover the computational structure of these models rather than only
measuring input-output behavior. The transformer-circuits line of work argues
that transformer internals can be studied with circuit-level tools, while later
dictionary-learning work shows that sparse autoencoders can identify features
that are often more interpretable than raw neurons.

An SAE is trained on frozen model activations. Given an activation vector
`x in R^d`, the encoder maps it into an overcomplete feature vector
`f in R^m`, where `m > d`, and a sparsity penalty encourages only a small number
of features to activate for any token. The decoder reconstructs the original
activation from those sparse features. If the SAE succeeds, each decoder
direction can be treated as an approximate feature direction in activation
space.

The research question here is:

> Does narrow-domain fine-tuning mainly change domain-relevant SAE features, or
> does it also reorganize unrelated parts of the feature space?

## 2. Experimental Setup

### Base Model and Layer

The base model is `EleutherAI/pythia-160m`. Pythia is useful for interpretability
because it is public, has consistent checkpoints, and is small enough to run
experiments without cluster-scale compute. I use a middle MLP layer,
`gpt_neox.layers.6.mlp`, because middle layers are a reasonable compromise:
early layers often capture token-local and lexical information, while later
layers are more entangled with next-token prediction and output-logit behavior.

### Data

For the base SAE, activations are collected from OpenWebText-style general
internet text. For fine-tuning, I use Python code from a public GitHub code
corpus. Python code is a deliberately narrow domain with visible structure:
indentation, comments, imports, decorators, strings, function definitions,
class definitions, exceptions, and punctuation-heavy syntax. This makes
qualitative feature interpretation easier than in domains where the expected
shift is semantically subtle.

The fine-tuned SAE is trained on the same general-text distribution as the base
SAE. This choice asks a sharper question: after the model is specialized on code,
do its representations of general text change?

### SAE Architecture

Both SAEs use:

- Input dimension: 768
- Hidden feature count: 6144, equal to an 8x expansion
- Activation: ReLU
- Objective: reconstruction MSE plus L1 sparsity penalty
- Decoder normalization after each optimizer step

The loss is:

```text
L = MSE(x_hat, x) + lambda * mean(abs(f))
```

where `f` is the sparse feature vector and `lambda` controls sparsity.

## 3. Method

The experiment has five stages.

1. Collect activations from `gpt_neox.layers.6.mlp` in the pretrained model.
2. Train the first SAE on those cached activations.
3. Fine-tune Pythia-160M on Python code.
4. Collect activations from the same layer in the fine-tuned model and train a
   second SAE with the same architecture and hyperparameters.
5. Compare base-SAE and fine-tuned-SAE features.

Feature matching is done by comparing decoder directions. For each base SAE
feature and fine-tuned SAE feature, I compute cosine similarity between their
decoder vectors. I then use Hungarian matching to construct a one-to-one mapping
that maximizes total cosine similarity. This gives a direct way to classify
features as stable, shifted, or reoriented.

The quantitative metrics are:

- Decoder cosine similarity after matching
- Activation frequency difference
- Mean activation difference
- Mean active activation difference
- Number of stable, frequency-shifted, and reoriented features

The qualitative metric is top activating token-context inspection. For selected
features, I record the top activating token and surrounding context before and
after fine-tuning. This is necessary because decoder similarity alone cannot say
what a feature means.

## 4. Results

After running the repository scripts, fill this section from:

- `artifacts/comparisons/base_vs_python/summary.json`
- `artifacts/comparisons/base_vs_python/feature_matches.csv`
- `artifacts/inspections/base/top_feature_examples.json`
- `artifacts/inspections/finetuned/top_feature_examples.json`

### Quantitative Summary

| Metric | Value |
| --- | ---: |
| Number of matched features | TODO |
| Median decoder cosine | TODO |
| Mean decoder cosine | TODO |
| Stable features | TODO |
| Frequency-shifted features | TODO |
| Reoriented features | TODO |

### Expected Domain-Aligned Changes

The most interpretable expected changes are code-oriented shifts:

- Features that activate on leading whitespace or indentation should become more
  frequent or more sharply activated.
- Features for punctuation such as `:`, `.`, `(`, `)`, `[`, `]`, and `,` may
  shift because Python code relies heavily on syntax tokens.
- Features for import statements, function definitions, class definitions, and
  comments may become more common.
- String-literal and identifier-like token features may separate from ordinary
  prose-token features.

### Unexpected Shifts

Unexpected shifts should be defined conservatively. A feature is a candidate
unexpected shift if:

- It has high decoder similarity to a base feature, but its activation frequency
  changes substantially on general text.
- It has low decoder similarity but its top activating contexts are not related
  to code.
- Its qualitative interpretation changes even though it appears to represent
  common prose tokens such as names, dates, sentiment words, or discourse
  markers.

These shifts matter because they would suggest that narrow fine-tuning changes
shared internal geometry, not only domain-specific circuits.

## 5. Interpretation

If most low-similarity or high-frequency-shift features are code-related, the
result supports a localized-change story: fine-tuning mostly alters features
connected to the new domain. If many unrelated prose features also shift, the
result supports a broader-drift story: fine-tuning adjusts shared representational
directions in a way that affects the model outside the target task.

A mixed result is plausible. The fine-tuning objective updates all model weights,
so even a narrow corpus can perturb residual-stream directions used for many
contexts. At the same time, Python code has strong token-level regularities, so
some feature changes should be easy to attribute to the domain.

## 6. Limitations

SAE comparisons are not perfectly identifiable. Two SAEs trained with different
random seeds can rotate, split, merge, or duplicate features even on the same
model. Therefore, a low decoder cosine does not automatically mean that the
model's internal concept disappeared. It may mean the second SAE represented the
same structure differently.

This project also uses one model size, one layer, one fine-tuning domain, and one
SAE architecture. A stronger study would repeat the experiment across multiple
layers, random seeds, sparsity coefficients, fine-tuning durations, and domains.
The top-context inspection is qualitative and can be biased by tokenization,
dataset choice, and the number of inspected examples.

Finally, training both SAEs on general text after code fine-tuning answers a
specific question about representational drift on the original distribution. A
separate analysis should train and evaluate on code activations too, because
some domain-relevant features may only be visible when the model is processing
code.

## 7. Future Work

Future work should include seed sweeps, cross-layer comparison, and direct
causal tests. For example, after identifying a code-specific feature, one could
ablate or stimulate it during code completion and measure whether syntax-related
next-token probabilities change. Another extension is to compare full
fine-tuning with LoRA fine-tuning to see whether parameter-efficient adaptation
causes less unrelated feature drift.

## References

- Nelson Elhage et al., "A Mathematical Framework for Transformer Circuits",
  Transformer Circuits, 2021:
  https://transformer-circuits.pub/2021/framework/index.html
- Anthropic, "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning", 2023:
  https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Stella Biderman et al., "Pythia: A Suite for Analyzing Large Language Models
  Across Training and Scaling", 2023:
  https://arxiv.org/abs/2304.01373
- `EleutherAI/pythia-160m` model card:
  https://huggingface.co/EleutherAI/pythia-160m
- Joseph Bloom et al., SAELens sparse autoencoder training codebase:
  https://github.com/jbloomAus/SAELens
