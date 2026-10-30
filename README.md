# 🧠 HSGauss


**Hierarchical Semantics-Enhanced Gaussian Representation Learning**

![Task](https://img.shields.io/badge/Task-Taxonomy%20Expansion-167D9A?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Representation](https://img.shields.io/badge/Representation-Gaussian-008F79?style=flat-square)
![Learning](https://img.shields.io/badge/Learning-End--to--End-B33C6B?style=flat-square)

> **Connecting concept text, hierarchical semantics, and probabilistic representations.**
>
> Hierarchy-aware concept understanding through Gaussian representations.

---

## Overview

**HSGauss** is a Gaussian representation learning framework for domain knowledge understanding and taxonomy expansion. Given a new concept, it ranks potential parents in an existing taxonomy to identify an appropriate attachment position.

The framework brings concept names, textual definitions, and visible hierarchical relations into a shared probabilistic representation. Parent–child semantic differences and sibling context inform the location and spread of candidate Gaussians. Parent candidates are ranked separately using the Bhattacharyya coefficient (BC) and directed KL divergence.

Three connected components define the framework: **S extracts hierarchical context, N incorporates context into Gaussian representations, and R constrains relation ranking across context views.**


## Core Components

| Component | Core idea                                                    | Role in the framework                                        |
| :-------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|   **S**   | **Hierarchical semantic differential selective state propagation** | Aggregates parent–child and sibling semantic differences through shared selective state updates. |
|   **N**   | **Context-guided Gaussian natural-parameter fusion**         | Translates hierarchical context into bounded mean and precision adjustments, then combines text and context in natural-parameter space. |
|   **R**   | **Sibling-context intervention and dual-view ranking constraints** | Preserves correct BC ordering and strengthens the weaker KL margin across factual and perturbed context views. |

### S · Understand Hierarchical Context

Parent–child relations express semantic inheritance and specialization, while sibling concepts provide a reference for distinctions within the same level. S builds state inputs from these semantic differences and extracts candidate context through selective propagation and attention readout.

### N · Shape Gaussian Representations

N converts hierarchical context into adjustments to Gaussian means and precisions. Natural-parameter fusion combines these adjustments with the base text Gaussian, allowing structural information to shape both a candidate's semantic location and its distributional spread.

### R · Learn Across Context Views

R perturbs selected sibling messages to construct factual and alternative context views within the same shared model. BC order preservation protects established correct rankings, while a weaker-margin KL objective constrains relation discrimination across both views.

## Design at a Glance

- **Text and hierarchy together** — Connects a concept's textual meaning with its visible structural position.
- **Asymmetric concept roles** — Uses text-based query representations and enhances candidates through query-masked hierarchical context.
- **Probabilistic relation modeling** — Represents concept semantics through Gaussian centers and precisions, with separate BC and KL ranking criteria.
- **Joint representation learning** — Optimizes text encoding, state propagation, Gaussian fusion, and relation constraints end to end.

---

The Gaussian relation modeling backbone builds on [TaxoBell](https://github.com/sahilmishra0012/TaxoBell).
