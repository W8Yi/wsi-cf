# Paper Planning (Updated Framing)

This document is the working outline for constructing the paper around `wsi_cf`.

It tracks:

* paper story
* main claims
* current evidence
* required experiments
* figures and tables

---

## Working Title

Primary:

* **Pathology-grounded counterfactual steering of whole-slide image regions**

Backup:

* **Concept-grounded counterfactual steering of whole-slide image regions**

---

## Scope

We frame the paper as a **general, multi-scale counterfactual framework** for pathology, with:

* HNSCC as the primary development setting
* additional cohorts as validation across different task granularities

The method supports:

* **tile-level and region-level counterfactual analysis**
* **slide-aware region selection (optional, not required)**
* **concept-driven, reusable interventions**

Target tasks:

* HNSCC: HPV status
* LUAD: growth-pattern transitions
* COAD/STAD: MSI
* KIRC: tumor grade
* CESC: cross-cancer HPV
* multi-cohort: tumor vs normal

---

## Problem

We aim to generate counterfactual edits in histopathology that are:

* spatially coherent
* biologically interpretable
* localized when needed
* strong enough to be meaningful
* aligned with downstream model behavior

Key limitations of prior work:

* tile-level methods lack sufficient context
* existing approaches are **task-specific and not reusable**
* large-scale generation lacks control and locality

---

## Core Question

Can we build a **reusable counterfactual framework** that:

* supports both tile-level and region-level edits
* uses a shared concept space across tasks
* produces localized, interpretable, and prediction-consistent changes

---

## Main Hypothesis

Counterfactual steering is more effective when:

* edits operate at **appropriate spatial scale (tile or region)**
* conditioning is **concept-grounded and reusable**
* generation is **spatially constrained**
* interventions are **aligned with downstream model behavior**

---

## Key Definition

**Pathology-grounded counterfactuals** are edits driven by:

* data-derived morphological concepts (SAE)
* grounded through representative prototype patterns
* statistically associated with clinical labels

rather than arbitrary latent perturbations.

---

## Correct Framing

### Multi-scale capability

The framework supports:

* **tile-level counterfactuals** (fine-grained, local)
* **region-level counterfactuals** (context-aware, structured)

Region-level editing provides improved morphological coherence, but both scales are part of the method.

---

### Slide-aware component

Whole-slide models are used for:

* identifying important regions
* guiding where to intervene

They are:

* **optional selection mechanisms**
* not required for the framework to operate
* not part of the generative model

---

### Generator

The generator is used for:

* producing localized counterfactual edits
* preserving surrounding structure

Key point:

* generation is **local**, not whole-slide

---

### Locality formulation

Counterfactual edits are spatially constrained:

* applied only within selected regions or tiles
* surrounding tissue is preserved
* transitions are smooth

---

### What we claim

* multi-scale (tile + region) counterfactual framework
* reusable concept space across tasks
* concept-grounded, interpretable interventions
* compatibility with downstream classifiers
* controlled locality and strength

---

### What we avoid

* full-slide generation
* universal claims
* causal biological conclusions
* architecture-level novelty

---

## One-sentence Summary

We introduce a multi-scale, concept-grounded framework for counterfactual steering in pathology, where reusable morphological concepts enable localized and interpretable edits across tasks.

---

## Contributions

1. **Multi-scale counterfactual framework**

   * supports both tile-level and region-level interventions
   * unifies local and contextual editing

2. **Reusable concept space**

   * concepts learned once from TCGA via SAE
   * reused across tasks and cancer types

3. **Concept-grounded steering**

   * interpretable, prototype-backed edits
   * structured manipulation of morphology

4. **Modular design**

   * decouples concept space, generator, and classifier
   * compatible with downstream models without retraining the generator

5. **Cross-task validation**

   * evaluated across tasks with different morphologic scales

---

## Core Comparison

* **Donor feature replacement** → baseline (instance-level editing)
* **SAE prototype steering** → main method (concept-level editing)

Goal:

* demonstrate that concept-level steering yields more coherent, interpretable, and stable counterfactuals

---

## Method Overview

1. (Optional) whole-slide attention → select region

2. extract tile or region

3. construct feature representation

4. apply steering:

   * donor replacement (baseline)
   * SAE concept steering (main)

5. generate localized counterfactual

6. evaluate visual, feature, and prediction changes

---

## Experimental Insight

We do NOT claim resolution as innovation.

We show:

* tile-level edits → limited context
* region-level edits → improved structure
* both scales are necessary depending on task

---

## Evaluation

### Visual

* target region changes
* spatial coherence
* absence of artifacts

### Feature-space

* movement toward target concept
* preservation outside edited region

### Downstream model

* prediction shift
* class flip rate

### Locality (primary metric)

* ratio of change inside vs outside edited region

---

## Pan-Cancer Strategy

Main paper:

* HNSCC (primary)
* LUAD (structural)
* MSI or grade (subtle)

Others → supplement

---

## Figures

### Fig 1 — Method

* WSI (optional) + region selection
* tile vs region editing
* concept steering
* before/after

### Fig 2–5

* donor vs SAE
* tile vs region comparison
* spatial scale experiments

### Fig 6

* quantitative metrics (locality + prediction shift)

### Fig 7

* cross-task validation

---

## Key Insight

The problem is not generation quality.

It is:

> how to **structure and constrain edits using reusable pathology concepts across spatial scales**

---

## Main Story

* concepts define what
* scale defines how much context
* (optional) attention defines where

---

## Bottom Line

This is a **multi-scale, concept-reusable, modular counterfactual framework**, not a task-specific or architecture-driven method.
## Method Notes

### What Is Actually Being Edited

The current method does **not** directly edit pixels. It edits selected UNI conditioning cells and then lets PixCell generate the image under that edited conditioning.

Current pipeline:

1. sample a real source region
2. encode the region into a UNI feature grid
3. choose selected cells in that grid
4. edit only those selected UNI cells in SAE latent space toward a prototype
5. decode the edited SAE representation back into UNI feature space
6. run PixCell diffusion with the edited conditioning grid
7. optionally preserve non-edited image regions in VAE latent space

### Exact Prototype Edit

For the current `wsi_cf` SAE runners, the edit mode is full-code prototype interpolation:

- source UNI features are flattened to `[N, D]`
- SAE encoding gives latent codes `z_lat`
- for selected tiles:
  - `z_edit = (1 - s) * z_lat + s * prototype`
- then decode back to UNI feature space
- then blend edited UNI features back into the original UNI feature grid

Where:

- `s = --prototype-strength`
- `prototype` is a full SAE latent vector from the prototype bundle
- `--steer-blend` controls UNI-space replacement after SAE decoding

So the method currently edits the **entire SAE code vector** for selected cells, not only a single latent neuron.

### Current Preservation Mechanism

Optional preservation is currently done in image latent space, not UNI space:

- encode the real source image into VAE latents
- build a spatial mask from the selected edited cells
- outside that mask, pull denoised latents back toward the noised trajectory of the original source image latents

Interpretation:

- conditioning edit says what should change
- latent preservation says what should stay the same

Current limitation:

- the preserve mask is hard and grid-aligned
- this can create visible rectangular boundaries

### Actual vs Generated Source

For figures and comparisons:

- `source_region_actual.png` = real tissue crop
- `source_region_generated.png` = baseline diffusion regeneration

This distinction matters and should be explicit in figures.

### Magnification / Representation Caveat

This is an important current limitation:

- the default prototype bundle and default SAE are from `20x`
- the new local `10x` experiments encode actual `10x` crops

So default `10x` steering currently uses:

- `20x` concept prototypes
- actual `10x` source features

That is a representation mismatch.

We do have a closer existing alternative:

- `/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2`

This uses `10x_pool2x2`, which is closer to the current `10x` setup than the default `20x` SAE, but still not identical to actual optical `10x`.

Paper implication:

- we should not overclaim magnification-agnostic concept transfer
- we should explicitly report the representation used for:
  - SAE training
  - prototype construction
  - steering source features

### Honest Positioning

The current method is best described as:

- region-level generation with coarse spatial cell-level control

not:

- dense structure-boundary editing
- exact object-level pathology editing

This is still scientifically useful, but the granularity and representation caveats should be stated clearly.
