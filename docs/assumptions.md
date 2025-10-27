# Assumptions and Limitations

**Statistical Assumptions for sIPD Generation**

---

## Table of Contents

1. [Overview](#overview)
2. [Data Assumptions](#data-assumptions)
3. [Statistical Assumptions](#statistical-assumptions)
4. [Model Assumptions](#model-assumptions)
5. [Violations and Impact](#violations-and-impact)
6. [Robustness Testing](#robustness-testing)
7. [Limitations](#limitations)
8. [Recommendations](#recommendations)

---

## Overview

This document explicitly states all assumptions underlying the five-phase sIPD generation methodology.

### Key Principle

> **All models are wrong, but some are useful** (Box, 1976)

---

## Data Assumptions

### DA1: Study Population Homogeneity

**Assumption**: Patients across K studies come from exchangeable populations.

**When This May Fail**:
- Different case definitions across studies
- Different injury severity distributions
- Different healthcare settings

**Impact if Violated**: Moderate - averaged correlations may not represent any single population

---

### DA2: Representative 2×2 Tables

**Assumption**: Published tables accurately represent underlying relationships.

**Mitigation**:
- Use comprehensive systematic review
- Include grey literature
- Conduct sensitivity analyses

---

## Statistical Assumptions

### SA1: Latent Bivariate Normality

**Assumption**: Binary predictors arise from latent normal variables.

**Evidence Supporting**:
- Central Limit Theorem
- Tetrachoric correlation widely validated
- Robust to moderate departures

**When This Fails**: Extreme prevalences (<1% or >99%)

---

### SA2: Gaussian Copula Adequacy

**Assumption**: Gaussian copula captures dependence structure.

**Alternatives if Violated**:
- t-copula (tail dependence)
- Clayton copula (lower tail)
- Gumbel copula (upper tail)

---

## Model Assumptions

### MA1: Logistic Link Function

**Assumption**: Log-odds linear in predictors.

**Robustness**: Moderate - fairly robust to mild non-linearity

---

### MA2: No Unmeasured Confounding

**Known Limitations**:
- Neurological exam details
- Mechanism specifics
- Time from injury

**Mitigation**: Acknowledge in reporting, conduct sensitivity analysis

---

## Violations and Impact

| Assumption | Severity | Detection | Mitigation |
|------------|----------|-----------|------------|
| Population homogeneity | Moderate | I² statistic | Subgroup analysis |
| Bivariate normality | Low-Moderate | Prevalence check | Polychoric correlation |
| Gaussian copula | Moderate | Model comparison | Alternative copulas |
| Logistic link | Low-Moderate | Residual plots | Add interactions |

---

## Robustness Testing

### Recommended Sensitivity Analyses

1. **Correlation Strength**: ±20% adjustment
2. **Prevalence**: ±2% adjustment
3. **Odds Ratios**: Use 95% CI bounds
4. **Sample Size**: N = 10K, 50K, 100K
5. **Missing Data**: Exclude studies with >30% missing

---

## Limitations

### Use Case Restrictions

**✓ Appropriate**:
- Model development
- Sample size calculations
- Educational demonstrations

**❌ Inappropriate**:
- Regulatory submissions alone
- Definitive treatment recommendations
- Publication as primary evidence

---

## Recommendations

1. **State assumptions explicitly**
2. **Conduct sensitivity analyses**
3. **Validate externally when possible**
4. **Acknowledge limitations transparently**

---

**Document Version**: 1.0  
**Last Updated**: January 2025
