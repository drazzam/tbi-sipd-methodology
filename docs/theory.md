# Statistical Theory and Mathematical Foundations

**Synthetic Individual Patient Data (sIPD) Generation Framework**

**Author**: Ahmed Azzam, MD  
**Institution**: Department of Neuroradiology, WVU Medicine  
**Date**: January 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Tetrachoric Correlation](#phase-1-tetrachoric-correlation)
3. [Phase 2: Iterative Proportional Fitting](#phase-2-iterative-proportional-fitting)
4. [Phase 3: Copula Theory](#phase-3-copula-theory)
5. [Phase 4: Bayesian Logistic Regression](#phase-4-bayesian-logistic-regression)
6. [Phase 5: Validation Theory](#phase-5-validation-theory)
7. [Mathematical Proofs](#mathematical-proofs)
8. [References](#references)

---

## Overview

This document provides the complete statistical and mathematical theory underlying the five-phase sIPD generation methodology. Each phase builds upon established statistical methods with rigorous theoretical foundations.

### Problem Statement

Given:
- **K studies** with 2×2 contingency tables for predictor-outcome relationships
- **P predictors** (binary variables)
- **Target prevalences** π₁, π₂, ..., πₚ for each predictor
- **Odds ratios** OR₁, OR₂, ..., ORₚ from meta-analysis

Generate:
- **N synthetic patients** with binary predictors X = [X₁, X₂, ..., Xₚ]
- **Binary outcomes** Y ∈ {0, 1}
- Preserve correlation structure, prevalences, and outcome associations

---

## Phase 1: Tetrachoric Correlation

### Theoretical Foundation

Tetrachoric correlation assumes that observed binary variables arise from an underlying bivariate normal distribution that has been discretized by thresholding.

**Assumption**: Binary variables (Xᵢ, Xⱼ) ∈ {0,1} arise from latent continuous variables (Zᵢ, Zⱼ) ~ N(μ, Σ) where:

$$
X_i = \begin{cases} 
1 & \text{if } Z_i > \tau_i \\
0 & \text{if } Z_i \leq \tau_i
\end{cases}
$$

### Mathematical Derivation

#### From 2×2 Table to Tetrachoric Correlation

Given 2×2 contingency table:

|       | Y=1 | Y=0 | Total |
|-------|-----|-----|-------|
| X=1   |  a  |  b  | a+b   |
| X=0   |  c  |  d  | c+d   |
| Total | a+c | b+d |   N   |

Define:
- π₁ = P(X=1) = (a+b)/N
- π₂ = P(Y=1) = (a+c)/N
- π₁₁ = P(X=1, Y=1) = a/N

The thresholds τ₁ and τ₂ for standard normal are:

$$
\tau_1 = \Phi^{-1}(1 - \pi_1)
$$

$$
\tau_2 = \Phi^{-1}(1 - \pi_2)
$$

The tetrachoric correlation ρ satisfies:

$$
\pi_{11} = \Phi_2(\tau_1, \tau_2, \rho)
$$

where Φ₂ is the bivariate normal CDF.

#### Estimation Method

We estimate ρ by finding the value that satisfies:

$$
\hat{\rho} = \arg\min_\rho \left| \Phi_2(\tau_1, \tau_2, \rho) - \pi_{11} \right|
$$

This is solved numerically using Brent's method with bounds [-0.999, 0.999].

#### Properties

1. **Range**: -1 ≤ ρ ≤ 1
2. **Interpretation**: Correlation between latent continuous variables
3. **Asymptotic Normality**: √N(ρ̂ - ρ) → N(0, V) where V depends on cell counts
4. **Consistency**: ρ̂ →ᵖ ρ as N → ∞

### Correlation Matrix Construction

Given P predictors, we estimate pairwise correlations to form **R** (P×P correlation matrix).

**Challenge**: Estimated R may not be positive semi-definite due to:
- Sampling variability
- Missing data (not all pairs observed)
- Inconsistent estimates across studies

**Solution**: Project R onto nearest PSD matrix:

$$
\hat{R}_{PSD} = \arg\min_{S \succeq 0} \|R - S\|_F
$$

where ‖·‖_F is Frobenius norm and S ⪰ 0 denotes positive semi-definite.

**Algorithm** (Higham 2002):
1. Eigen-decompose: R = QΛQᵀ
2. Truncate negative eigenvalues: Λ₊ = max(Λ, ε·I)
3. Reconstruct: R̂ = QΛ₊Qᵀ
4. Rescale diagonal to 1

---

## Phase 2: Iterative Proportional Fitting

### Theoretical Foundation

IPF (also called RAS algorithm or biproportional fitting) adjusts a multivariate distribution to match specified marginal distributions while preserving structure.

### Problem Formulation

Given:
- Initial sample Z ~ N(0, R) with R from Phase 1
- Target marginals: P(Xᵢ = 1) = πᵢ for i = 1, ..., P

Find transformation T: Z → X such that:
1. X preserves correlation structure from Z
2. Marginal prevalences match targets exactly

### Algorithm

**Step 1**: Generate latent continuous variables
$$
Z \sim N(0, R)
$$

**Step 2**: Initial binary conversion
$$
X^{(0)}_i = \mathbb{1}(Z_i > \Phi^{-1}(1-\pi_i))
$$

**Step 3**: Iterative adjustment (for t = 1, 2, ..., T):
$$
X^{(t+1)} = \text{adjust\_margins}(X^{(t)}, Z, \{\pi_i\})
$$

Adjustment preserves ordering from Z while matching marginals exactly.

### Convergence Theory

**Theorem** (IPF Convergence): Under mild regularity conditions, the IPF algorithm converges to a solution X̂ that satisfies:

$$
\|P_{\hat{X}}(X_i = 1) - \pi_i\| < \epsilon \quad \forall i
$$

for specified tolerance ε.

**Proof sketch**:
1. Define Kullback-Leibler divergence: D_KL(P‖Q) = Σ P log(P/Q)
2. Each IPF iteration decreases D_KL
3. D_KL is strictly convex
4. Therefore, algorithm converges to global minimum

**Convergence rate**: O(1/t) where t is iteration number.

### Properties

1. **Marginal Preservation**: Exact by construction
2. **Correlation Preservation**: Approximately preserved (typically >0.95)
3. **Uniqueness**: Solution is unique under typical conditions
4. **Computational Complexity**: O(NPT) where N=samples, P=predictors, T=iterations

---

## Phase 3: Copula Theory

### Sklar's Theorem

**Theorem** (Sklar 1959): Let F be a joint distribution function with marginals F₁, ..., Fₚ. Then there exists a copula C: [0,1]ᴾ → [0,1] such that:

$$
F(x_1, ..., x_p) = C(F_1(x_1), ..., F_p(x_p))
$$

If F₁, ..., Fₚ are continuous, C is unique.

### Gaussian Copula

The Gaussian copula with correlation matrix R is defined as:

$$
C_R(u_1, ..., u_p) = \Phi_R(\Phi^{-1}(u_1), ..., \Phi^{-1}(u_p))
$$

where Φ_R is the multivariate normal CDF with correlation R.

### Properties

1. **Tail Dependence**: Gaussian copula has zero tail dependence
2. **Symmetry**: C(u₁, u₂) = C(u₂, u₁)
3. **Fréchet Bounds**: W(u) ≤ C(u) ≤ M(u) for all copulas

### Application to Binary Data

For binary variables with prevalences π₁, ..., πₚ:

1. **Generate**: (U₁, ..., Uₚ) ~ Gaussian copula with correlation R
2. **Transform**: Xᵢ = 𝟙(Uᵢ > 1 - πᵢ)
3. **Result**: Binary variables with approximately correct correlations

### Validation

Measure correlation preservation:

$$
\rho_{preservation} = \text{cor}(\text{vec}(R_{target}), \text{vec}(R_{observed}))
$$

Typically requires ρ_preservation > 0.95 for acceptable fit.

---

## Phase 4: Bayesian Logistic Regression

### Model Specification

The outcome Y follows a Bernoulli distribution:

$$
Y_i \sim \text{Bernoulli}(p_i)
$$

with logit link:

$$
\text{logit}(p_i) = \beta_0 + \sum_{j=1}^P \beta_j X_{ij}
$$

### Prior Distribution

Weakly informative priors on coefficients:

$$
\beta_j \sim N(\log(\text{OR}_j), \sigma^2) \quad j = 1, ..., P
$$

$$
\beta_0 \sim N(\mu_0, \sigma_0^2)
$$

where ORⱼ are published odds ratios and μ₀ calibrates intercept for target prevalence.

### Posterior Inference

Using Bayes' theorem:

$$
p(\beta | X, Y) \propto p(Y | X, \beta) \cdot p(\beta)
$$

Sampling via Hamiltonian Monte Carlo (HMC):
- **Chains**: 4 parallel chains
- **Warmup**: 1,000 iterations
- **Sampling**: 1,000 iterations per chain
- **Thinning**: None (HMC has low autocorrelation)

### Convergence Diagnostics

**R̂ statistic** (Gelman-Rubin):

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

where:
- Ŵ = within-chain variance
- B̂ = between-chain variance  
- V̂ = (W + B)/2

**Criterion**: R̂ < 1.01 for all parameters indicates convergence.

### Predictive Distribution

For new patient X*:

$$
p(Y^* = 1 | X^*, X, Y) = \int p(Y^* = 1 | X^*, \beta) \cdot p(\beta | X, Y) d\beta
$$

Approximated using posterior samples:

$$
\hat{p}(Y^* = 1 | X^*) = \frac{1}{S} \sum_{s=1}^S \text{logit}^{-1}(\beta_0^{(s)} + \sum_{j=1}^P \beta_j^{(s)} X^*_j)
$$

---

## Phase 5: Validation Theory

### Discrimination

**C-statistic** (equivalent to AUC):

$$
C = P(p_i > p_j | Y_i = 1, Y_j = 0)
$$

Interpretation: Probability that a randomly selected case has higher predicted risk than a randomly selected control.

### Calibration

**Brier Score**:

$$
BS = \frac{1}{N} \sum_{i=1}^N (Y_i - p_i)^2
$$

Lower is better; perfect calibration gives BS = 0.

**Calibration Slope**: From logistic regression of Y on logit(p):

$$
\text{logit}(Y_i) \sim \alpha + \gamma \cdot \text{logit}(p_i)
$$

Perfect calibration: γ = 1, α = 0.

### Clinical Decision Rules

For CDR with predictors S ⊂ {1, ..., P}:

$$
\text{Recommend CT} = \bigvee_{j \in S} X_j
$$

**Sensitivity**:

$$
\text{Sens} = P(\text{CT recommended} | Y = 1)
$$

**Specificity**:

$$
\text{Spec} = P(\text{CT not recommended} | Y = 0)
$$

---

## Mathematical Proofs

### Proof 1: IPF Convergence

**Claim**: IPF algorithm converges to a solution satisfying marginal constraints.

**Proof**:
1. Define objective: minimize D_KL(P_X ‖ P_Z) subject to marginal constraints
2. D_KL is strictly convex
3. Feasible set is non-empty (continuous Z can be discretized to match any marginals)
4. Each IPF step decreases D_KL monotonically
5. D_KL ≥ 0 with minimum at 0
6. By monotone convergence theorem, algorithm converges to minimum

∎

### Proof 2: Tetrachoric Correlation Consistency

**Claim**: Tetrachoric correlation estimator ρ̂ is consistent.

**Proof**:
1. Cell proportions π̂₁, π̂₂, π̂₁₁ converge to true values by LLN
2. Threshold estimates τ̂₁, τ̂₂ converge by continuity of Φ⁻¹
3. Bivariate normal CDF Φ₂ is continuous in ρ
4. Therefore, solution to Φ₂(τ̂₁, τ̂₂, ρ̂) = π̂₁₁ converges to solution with true parameters

∎

---

## References

1. **Tetrachoric Correlation**: Pearson, K. (1900). *Mathematical contributions to the theory of evolution*

2. **Iterative Proportional Fitting**: Deming & Stephan (1940). *On a least squares adjustment of a sampled frequency table*

3. **Copula Theory**: Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges*

4. **Gaussian Copula**: Song, P. X.-K. (2000). *Multivariate dispersion models generated from Gaussian copula*

5. **Bayesian Logistic Regression**: Gelman et al. (2013). *Bayesian Data Analysis, 3rd Edition*

6. **PSD Projection**: Higham, N. J. (2002). *Computing the nearest correlation matrix*

7. **Calibration**: Steyerberg, E. W. (2009). *Clinical Prediction Models*

---

**Document Version**: 1.0  
**Last Updated**: January 2025
