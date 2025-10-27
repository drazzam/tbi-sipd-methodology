# Best Practices for sIPD Generation

**Practical Implementation Guidelines**

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Model Building](#model-building)
4. [Validation Strategies](#validation-strategies)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Reporting Standards](#reporting-standards)
7. [Common Pitfalls](#common-pitfalls)
8. [Workflow Checklist](#workflow-checklist)

---

## Overview

This document provides practical guidance for implementing the sIPD generation methodology. These best practices emerged from extensive testing and reflect lessons learned.

### Guiding Principles

1. **Validate at every step** - Don't wait until the end
2. **Document all decisions** - Reproducibility matters
3. **Test sensitivity** - Assumptions matter
4. **Be transparent** - Acknowledge limitations
5. **Compare to reality** - External validation is key

---

## Data Preparation

### Extracting 2×2 Tables

**Best Practice**: Create standardized extraction template

```python
# Template for each study
study_data = {
    'study_id': 'Haydel_2000',
    'n_total': 520,
    'outcome_prevalence': 0.08,
    'predictors': {
        'age_65_plus': {
            'a': 15,  # outcome+, predictor+
            'b': 45,  # outcome-, predictor+
            'c': 27,  # outcome+, predictor-
            'd': 433  # outcome-, predictor-
        },
        # ... other predictors
    }
}
```

**Check**: Always verify a+b+c+d = N

---

### Handling Missing Data

**Rule 1**: Document which correlations are missing
**Rule 2**: Never silently impute without justification

**Recommended Approach**:
```python
# 1. Calculate proportion missing
missing_pct = (n_missing_pairs / total_pairs) * 100

# 2. Decision tree
if missing_pct < 10:
    # Use available data only
    method = "complete_case"
elif missing_pct < 30:
    # Impute conservatively (use r=0 or population average)
    method = "conservative_imputation"
else:
    # Too much missing - warn user
    raise ValueError(f"Too much missing data ({missing_pct}%)")
```

---

### Data Quality Checks

**Checklist Before Phase 1**:

- [ ] All 2×2 tables have positive counts (no zeros)
- [ ] Study sizes are plausible (N > 100)
- [ ] Prevalences are reasonable (0.01 < π < 0.99)
- [ ] No obvious data entry errors (e.g., a+b+c+d ≠ N)
- [ ] Odds ratios are in expected range (0.1 < OR < 100)

**Automated Check**:
```python
def validate_2x2_table(table):
    assert table.sum() > 100, "Sample size too small"
    assert np.all(table >= 0), "Negative counts"
    
    prevalence = (table[0,0] + table[1,0]) / table.sum()
    assert 0.01 < prevalence < 0.99, "Extreme prevalence"
    
    return True
```

---

## Model Building

### Phase 1: Correlation Matrix

**Best Practice**: Use multiple methods and compare

```python
# Method 1: Direct tetrachoric
corr_tetrachoric = estimate_tetrachoric(tables)

# Method 2: Pooled using meta-analysis
corr_pooled = meta_analysis_correlation(tables)

# Method 3: Bayesian hierarchical model
corr_bayesian = bayesian_correlation(tables)

# Compare
print(f"Agreement: {np.corrcoef(corr_tetrachoric.ravel(), 
                               corr_bayesian.ravel())[0,1]:.3f}")
```

**Red Flags**:
- Agreement < 0.90 → Investigate discrepancies
- Large differences suggest heterogeneity

---

### Phase 2: IPF

**Key Parameters**:
- `max_iter`: Set to 100 (usually converges in 10-30)
- `tolerance`: 0.01 (1% deviation acceptable)
- `n_samples`: Start small (1K), scale up after validation

**Convergence Monitoring**:
```python
ipf_gen = IPFGenerator(corr_matrix, prevalences, verbose=True)
X = ipf_gen.generate(1000, random_seed=42)

# Check convergence
for i, name in enumerate(predictor_names):
    obs_prev = X[:, i].mean()
    target_prev = prevalences[name]
    print(f"{name}: {abs(obs_prev - target_prev):.4f}")
    
# If any diff > 0.02, increase max_iter or check data quality
```

**Pro Tip**: Run with multiple seeds to check stability

---

### Phase 3: Copula

**When to Use vs. Skip**:
- ✅ Use: Complex correlation structure, validation needed
- ⏭️ Skip: Simple structure, IPF sufficient, time constraints

**Parameter Selection**:
```python
# For most cases: Gaussian copula
copula = CopulaModel(copula_type='gaussian')

# For heavy tails: t-copula (requires additional parameter)
# copula = TcopulaModel(df=5)  # df = degrees of freedom
```

**Validation Threshold**: correlation_preservation > 0.95

---

### Phase 4: Bayesian Model

**Prior Selection**:

**Weakly Informative (Recommended)**:
```python
# Center on published OR, moderate uncertainty
prior_sd = 0.5  # Allows OR to vary ±30% from published
```

**Very Weak (Use if skeptical of published ORs)**:
```python
prior_sd = 1.0  # Allows wide variation
```

**Strong (Use if very confident in published values)**:
```python
prior_sd = 0.2  # Tight around published values
```

**MCMC Settings**:
```python
# Development (fast)
n_samples = 1000
n_chains = 2
target_accept = 0.8

# Production (thorough)
n_samples = 2000
n_chains = 4
target_accept = 0.95
```

**Convergence Diagnostics**:
```python
# R-hat < 1.01 for all parameters
rhat_values = model.summary()['r_hat']
assert np.all(rhat_values < 1.01), "MCMC not converged"

# Effective sample size > 1000
ess = model.summary()['ess_bulk']
assert np.all(ess > 1000), "Insufficient effective samples"
```

---

## Validation Strategies

### Train/Test Split

**Standard Split**: 70/30 or 60/40
**Stratified**: Essential for rare outcomes

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    stratify=y,  # IMPORTANT for rare outcomes
    random_state=42
)
```

**Check Class Balance**:
```python
print(f"Train prevalence: {y_train.mean():.4f}")
print(f"Test prevalence: {y_test.mean():.4f}")
# Should be within 20% of each other
```

---

### External Validation

**Gold Standard**: Compare to held-out real data

```python
# Validate on real external dataset
c_stat_real = calculate_c_statistic(X_real, y_real, model)
c_stat_synthetic = calculate_c_statistic(X_test, y_test, model)

print(f"Real data C-stat: {c_stat_real:.3f}")
print(f"Synthetic C-stat: {c_stat_synthetic:.3f}")
print(f"Difference: {abs(c_stat_real - c_stat_synthetic):.3f}")

# Acceptable if difference < 0.05
```

---

### Bootstrap Confidence Intervals

**Best Practice**: Use bootstrap for robust CIs

```python
def bootstrap_ci(X, y, model, n_boot=1000):
    c_stats = []
    for i in range(n_boot):
        indices = np.random.choice(len(X), len(X), replace=True)
        c_stat = calculate_c_statistic(
            X[indices], y[indices], model
        )
        c_stats.append(c_stat)
    
    return np.percentile(c_stats, [2.5, 97.5])

ci = bootstrap_ci(X_test, y_test, model)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## Interpretation Guidelines

### Understanding Synthetic Data

**What Synthetic Data Represents**:
- ✅ Plausible data consistent with published summaries
- ✅ Useful for model development and comparison
- ✅ Educational and hypothesis-generating

**What Synthetic Data Does NOT Represent**:
- ❌ Actual real patient data
- ❌ Ground truth for clinical decisions
- ❌ Substitute for prospective validation

---

### Comparing Models

**Fair Comparison Requires**:
1. Same synthetic dataset for all models
2. Same train/test split
3. Same outcome definition
4. Same performance metrics

```python
# Compare multiple models on same data
models = ['logistic', 'random_forest', 'xgboost']
results = []

for model_name in models:
    model = fit_model(X_train, y_train, model_name)
    c_stat = evaluate_model(X_test, y_test, model)
    results.append({'model': model_name, 'c_stat': c_stat})

comparison_df = pd.DataFrame(results)
```

---

### Statistical Testing

**Appropriate Tests**:
- DeLong test for comparing C-statistics
- McNemar test for comparing binary predictions
- Calibration curves for visual comparison

**Sample Code**:
```python
from scipy.stats import chi2

# DeLong test for C-statistic comparison
def delong_test(y_true, pred1, pred2):
    """
    Test if two C-statistics are significantly different
    """
    # Implementation details...
    return p_value

p = delong_test(y_test, pred_model1, pred_model2)
if p < 0.05:
    print("Models significantly different")
```

---

## Reporting Standards

### Minimum Reporting Requirements

**In Methods Section**:
1. Number of studies and total patients
2. Correlation estimation method
3. Missing data handling
4. Priors used in Bayesian model
5. Sample size generated
6. Random seeds for reproducibility

**In Results Section**:
1. Correlation matrix summary statistics
2. Prevalence validation (observed vs. target)
3. Model performance metrics (C-stat, Brier, calibration)
4. CDR comparison results

**In Discussion Section**:
1. Limitations of synthetic data
2. Sensitivity to assumptions
3. External validation results (if available)
4. Intended use cases

---

### Reproducibility Checklist

- [ ] All random seeds documented
- [ ] Software versions reported
- [ ] Code made available (GitHub)
- [ ] Data extraction sheet shared
- [ ] Parameter choices justified

**Example Methods Statement**:

> "We generated synthetic individual patient data for 50,000 patients using a five-phase methodology. Phase 1 estimated tetrachoric correlations from 2×2 tables from K=9 studies (N=61,955). Phase 2 used iterative proportional fitting (max_iter=100, tolerance=0.01) to generate predictors with target prevalences. Phase 4 applied Bayesian logistic regression with weakly informative priors (SD=0.5) centered on published odds ratios. All analyses used Python 3.10 with PyMC 5.0. Random seed was set to 42 for reproducibility. Code is available at github.com/..."

---

## Common Pitfalls

### Pitfall 1: Ignoring Convergence

**Problem**: Using non-converged MCMC chains

**Solution**: Always check R-hat and ESS

---

### Pitfall 2: Overfitting to Training Data

**Problem**: Reporting training set performance

**Solution**: ALWAYS report test set performance

---

### Pitfall 3: Cherry-Picking Results

**Problem**: Running multiple seeds, reporting best

**Solution**: Pre-specify seed or report all results

---

### Pitfall 4: Forgetting Limitations

**Problem**: Treating synthetic = real data

**Solution**: Always caveat findings with "synthetic data"

---

### Pitfall 5: Ignoring Missingness

**Problem**: Assuming MCAR without testing

**Solution**: Conduct sensitivity analyses

---

## Workflow Checklist

**Before Starting**:
- [ ] Systematic review complete
- [ ] 2×2 tables extracted and validated
- [ ] Software environment set up
- [ ] Random seed chosen

**Phase 1**:
- [ ] Correlation matrix estimated
- [ ] Matrix is positive semi-definite
- [ ] Correlations are plausible

**Phase 2**:
- [ ] IPF converged (all diffs < tolerance)
- [ ] Correlation structure preserved (r > 0.95)

**Phase 3** (Optional):
- [ ] Copula fitted successfully
- [ ] Validation metrics acceptable

**Phase 4**:
- [ ] Bayesian model converged (R-hat < 1.01)
- [ ] Outcome prevalence matches target

**Phase 5**:
- [ ] Train/test split stratified
- [ ] Model performance evaluated
- [ ] CDR validation complete

**Reporting**:
- [ ] Methods fully documented
- [ ] Results reported with CIs
- [ ] Limitations acknowledged
- [ ] Code shared publicly

---

## Conclusion

Following these best practices will maximize the quality and utility of synthetic data while maintaining scientific rigor and transparency.

> "In God we trust, all others must bring data" - W. Edwards Deming

And when the data is synthetic, we must bring even more rigor.

---

**Document Version**: 1.0  
**Last Updated**: January 2025
