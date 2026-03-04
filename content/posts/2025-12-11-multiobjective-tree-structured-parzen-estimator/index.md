---
title: 'Multiobjective Tree-Structured Parzen Estimator'
date: 2025-12-11T13:41:18+00:00
draft: false
description: 'Paper-reading notes: MOTPE'
tag: 'Posts'
ShowWordCount: true
ShowReadingTime: false
tags:
  - optimization
---


# 1. Problem

## 1.1 Real-world optimization setting

Many practical optimization problems require **simultaneous optimization of multiple conflicting objectives**, for example:

- maximize accuracy and minimize inference time
- minimize error and minimize model size
- engineering design trade-offs (performance vs cost)

Formally:

$$
\min_{x \in \mathcal{X}} f(x) = (f_1(x), \dots, f_m(x))
$$

where objectives are **black-box,** evaluations are **expensive** and analytical gradients are **unavailable**

## 1.2 Challenges with standard Bayesian Optimization (GP-based)

Traditional multiobjective BO typically relies on **Gaussian Processes (GPs)**, but they struggle with:

1. **Complex search spaces**
    - categorical variables, integer variables, conditional (tree-structured) parameters
2. **Scalability**
    - GP inference scales with number of observations, expensive in medium/high dimensions
3. **Parallel evaluation**
    - batch GP-BO requires complex coordination, synchronization overhead is high

## 1.3 Why TPE is attractive (but insufficient)

Tree-structured Parzen Estimator (TPE):

- models **parameter distributions** instead of the objective function
- naturally handles:
    - conditional parameters
    - categorical / integer variables
- scales well to many trials
- easy to parallelize

**Limitation**:

TPE is **single-objective only** and cannot directly handle Pareto optimization.

## 1.4 Goal of the paper

Design a **multiobjective Bayesian optimization algorithm** that:

- handles **tree-structured, mixed-type search spaces**
- works with **limited evaluation budgets**
- approximates the **Pareto front efficiently**
- supports **asynchronous parallelization**

# 2. Method (MOTPE)

## 2.1 Core idea

MOTPE extends TPE by:

- redefining **“good” vs “bad” observations** using **Pareto dominance**
- preserving TPE’s **density-ratio optimization principle**
- replacing Expected Improvement with **Expected Hypervolume Improvement**

## 2.2 Review: TPE (single objective)

TPE models: $p(x \mid y)$

using two densities:

- $l(x)$: density of **good** configurations
- $g(x)$: density of **bad** configurations

Split rule (single objective):

- good if ( y < $y^*$ ) (top γ quantile)
- bad otherwise

Acquisition (equivalent to EI):

$$
\arg\max_x \frac{l(x)}{g(x)}
$$

## 2.3 Key extension to multiobjective setting

### Observation classification

For multiobjective outcomes $y \in \mathbb{R}^m$:

- **Good observations**:
    - nondominated points
    - or points incomparable with a reference Pareto set $Y^*$
- **Bad observations**:
    - points **dominated** by $Y^*$

This replaces scalar thresholding with **Pareto-based ranking**.

## 2.4 Defining the “good” set

Procedure:

1. Perform **nondominated sorting**
2. Add Pareto fronts greedily until reaching γ proportion
3. If a front overflows:
    - select points maximizing **hypervolume contribution**

This ensures quality (near Pareto front) and diversity (spread along front).

## 2.5 Density modeling

For each parameter $x_i$, construct:

- $l(x_i)$ from good observations
- $g(x_i)$ from bad observations

Density estimation:

- Parzen estimators
- Gaussian kernels (real/integer)
- histograms (categorical)

**Important difference from TPE**:

- good samples are **weighted by hypervolume contribution**
- bad samples use uniform weight

## 2.6 Acquisition function (key result)

MOTPE uses **Expected Hypervolume Improvement (EHVI)**.

The paper shows:

$$
\text{EHVI}(x_i) \propto \left(\gamma + (1-\gamma)\frac{g(x_i)}{l(x_i)}\right)^{-1}
$$

Therefore:

$$
\arg\max_{x_i} \text{EHVI}(x_i)
\Longleftrightarrow
\arg\max_{x_i} \frac{l(x_i)}{g(x_i)}
$$

✅ Same optimization rule as TPE

✅ No explicit modeling of $p(y \mid x)$ needed

## 2.7 Sampling procedure

- sample each parameter **independently**
- traverse the **tree-structured space**
- sample candidates from $l(x_i)$
- select those with highest $l/g$ ratio
- evaluate full configuration

## 2.8 Parallelization

MOTPE supports **asynchronous parallel execution**:

- no batch synchronization
- unfinished evaluations ignored
- new candidates sampled stochastically

Result:

- large wall-clock speedups
- small loss in sample efficiency
- very practical for real systems

## One-line takeaway

MOTPE extends TPE to multiobjective optimization by replacing scalar “good/bad” splits with Pareto-based splits and using hypervolume-aware density modeling—while keeping the same simple $l(x)/g(x)$ decision rule.