"""
Custom Correlation Example
===========================

Demonstrates how to specify custom correlation structures and perform
sensitivity analysis by varying correlation strengths. Shows how correlation
assumptions affect synthetic data generation.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

from phase1_correlation_matrix import nearest_positive_semidefinite
from phase2_ipf import IPFGenerator
from phase3_copula import CopulaModel, compare_copula_to_ipf

print("=" * 80)
print("CUSTOM CORRELATION STRUCTURES - SENSITIVITY ANALYSIS")
print("=" * 80)

# ============================================================================
# Example 1: Manually Specified Correlation Matrix
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: Manually Specified Correlation Matrix")
print("=" * 80)

# Define 5 predictors with custom correlations
predictor_names = ['age', 'symptom_a', 'symptom_b', 'injury_type', 'test_result']

# Custom correlation matrix (representing clinical knowledge)
# For example:
# - age correlated with symptom_a (older patients have more symptoms)
# - symptom_a and symptom_b moderately correlated
# - injury_type independent of age but correlated with test results
custom_correlation = np.array([
    [1.00, 0.40, 0.30, 0.10, 0.05],  # age
    [0.40, 1.00, 0.50, 0.15, 0.20],  # symptom_a
    [0.30, 0.50, 1.00, 0.10, 0.15],  # symptom_b
    [0.10, 0.15, 0.10, 1.00, 0.45],  # injury_type
    [0.05, 0.20, 0.15, 0.45, 1.00]   # test_result
])

# Ensure positive semi-definite
custom_correlation = nearest_positive_semidefinite(custom_correlation)

print("\nCustom Correlation Matrix:")
print(pd.DataFrame(custom_correlation, 
                  columns=predictor_names, 
                  index=predictor_names).round(2))

# Define prevalences
custom_prevalences = {
    'age': 0.25,
    'symptom_a': 0.30,
    'symptom_b': 0.20,
    'injury_type': 0.15,
    'test_result': 0.10
}

print("\nTarget Prevalences:")
for name, prev in custom_prevalences.items():
    print(f"  {name}: {prev:.2%}")

# Generate data
print("\nGenerating 10,000 samples...")
ipf_gen = IPFGenerator(
    correlation_matrix=custom_correlation,
    target_prevalences=custom_prevalences,
    max_iter=100,
    tolerance=0.01
)

X_custom = ipf_gen.generate(n_samples=10000, random_seed=42)

print(f"✓ Generated {X_custom.shape[0]:,} samples")

# Validate correlation preservation
observed_corr = np.corrcoef(X_custom.T)
print("\nCorrelation Preservation:")
for i in range(len(predictor_names)):
    for j in range(i+1, len(predictor_names)):
        target = custom_correlation[i, j]
        observed = observed_corr[i, j]
        diff = abs(target - observed)
        print(f"  {predictor_names[i]:15s} <-> {predictor_names[j]:15s}: "
              f"target={target:.3f}, observed={observed:.3f}, diff={diff:.3f}")

# ============================================================================
# Example 2: Sensitivity Analysis - Varying Correlation Strength
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Sensitivity Analysis - Correlation Strength")
print("=" * 80)

# Test different correlation strengths
correlation_strengths = [0.1, 0.3, 0.5, 0.7]

# Simple 3-predictor setup
simple_predictors = ['predictor_1', 'predictor_2', 'predictor_3']
simple_prevalences = {
    'predictor_1': 0.30,
    'predictor_2': 0.25,
    'predictor_3': 0.20
}

print("\nComparing outcomes across different correlation strengths...")
print(f"Predictors: {simple_predictors}")
print(f"Prevalences: {list(simple_prevalences.values())}")

results = []

for strength in correlation_strengths:
    print(f"\n--- Correlation Strength: {strength:.1f} ---")
    
    # Create correlation matrix with specified strength
    corr_matrix = np.eye(3)
    corr_matrix[0, 1] = corr_matrix[1, 0] = strength
    corr_matrix[0, 2] = corr_matrix[2, 0] = strength * 0.7
    corr_matrix[1, 2] = corr_matrix[2, 1] = strength * 0.8
    
    # Generate data
    ipf_gen = IPFGenerator(
        correlation_matrix=corr_matrix,
        target_prevalences=simple_prevalences,
        max_iter=100,
        tolerance=0.01
    )
    
    X_sens = ipf_gen.generate(n_samples=5000, random_seed=42)
    
    # Calculate observed correlations
    obs_corr = np.corrcoef(X_sens.T)
    
    # Calculate co-occurrence rate (how often predictors appear together)
    co_occurrence = np.sum(np.all(X_sens == 1, axis=1)) / len(X_sens)
    
    result = {
        'strength': strength,
        'corr_12': obs_corr[0, 1],
        'corr_13': obs_corr[0, 2],
        'corr_23': obs_corr[1, 2],
        'co_occurrence': co_occurrence
    }
    results.append(result)
    
    print(f"  Observed correlations: "
          f"r12={obs_corr[0,1]:.3f}, r13={obs_corr[0,2]:.3f}, r23={obs_corr[1,2]:.3f}")
    print(f"  Co-occurrence rate: {co_occurrence:.3%}")

# Summary table
print("\n" + "-" * 80)
print("SENSITIVITY ANALYSIS SUMMARY")
print("-" * 80)

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))

# ============================================================================
# Example 3: Comparing Independence vs. Correlation
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Impact of Independence Assumption")
print("=" * 80)

# Generate under independence assumption
print("\nScenario A: Independent Predictors")
indep_corr = np.eye(3)
ipf_indep = IPFGenerator(
    correlation_matrix=indep_corr,
    target_prevalences=simple_prevalences,
    max_iter=100,
    tolerance=0.01
)
X_indep = ipf_indep.generate(n_samples=5000, random_seed=42)

# Generate under correlation assumption
print("\nScenario B: Correlated Predictors (r=0.5)")
corr_corr = np.array([[1.0, 0.5, 0.5],
                      [0.5, 1.0, 0.5],
                      [0.5, 0.5, 1.0]])
ipf_corr = IPFGenerator(
    correlation_matrix=corr_corr,
    target_prevalences=simple_prevalences,
    max_iter=100,
    tolerance=0.01
)
X_corr = ipf_corr.generate(n_samples=5000, random_seed=43)

# Compare outcomes
print("\n" + "-" * 80)
print("COMPARISON: Independence vs. Correlation")
print("-" * 80)

# Number of predictors present per patient
n_predictors_indep = X_indep.sum(axis=1)
n_predictors_corr = X_corr.sum(axis=1)

print(f"\nDistribution of number of predictors present per patient:")
print(f"\n{'Scenario':<25s} {'0 pred':>10s} {'1 pred':>10s} {'2 pred':>10s} {'3 pred':>10s}")
print("-" * 65)

for name, n_pred in [('Independent', n_predictors_indep), 
                     ('Correlated (r=0.5)', n_predictors_corr)]:
    dist = [(n_pred == i).sum() / len(n_pred) for i in range(4)]
    print(f"{name:<25s} {dist[0]:>9.1%} {dist[1]:>9.1%} {dist[2]:>9.1%} {dist[3]:>9.1%}")

print("\nKey Insight:")
print("  - Independent predictors: More patients with exactly 1 predictor")
print("  - Correlated predictors: More patients with 0 or multiple predictors")
print("  - This affects clinical decision rules that use 'ANY' logic!")

# ============================================================================
# Example 4: Comparing Copula vs IPF Methods
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Copula vs IPF Method Comparison")
print("=" * 80)

print("\nGenerating data using both methods...")

# IPF method (already have X_corr from above)
X_ipf = X_corr

# Copula method
X_df = pd.DataFrame(X_ipf, columns=simple_predictors)
copula = CopulaModel(copula_type='gaussian')
copula.fit(X_df, feature_names=simple_predictors, verbose=False)
X_copula = copula.sample(n_samples=5000, random_seed=44)

# Compare methods
print("\nComparing IPF vs Copula approaches...")
comparison = compare_copula_to_ipf(X_ipf, X_copula, simple_predictors)

print(f"\nMethod Agreement (correlation of correlations): {comparison['method_agreement']:.4f}")
print(f"Max Correlation Difference: {comparison['max_correlation_diff']:.4f}")

print("\nPrevalence Comparison:")
print(comparison['prevalence_comparison'].round(4).to_string(index=False))

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)

print("""
1. **Correlation Matters**:
   - Stronger correlations → more co-occurrence of predictors
   - Affects clinical decision rule performance
   - Independence assumption may underestimate risk in some patients

2. **Sensitivity Analysis is Critical**:
   - Test multiple correlation scenarios (weak, moderate, strong)
   - Validate against clinical knowledge and observed data
   - Document correlation assumptions clearly

3. **Method Comparison**:
   - IPF: Fast, exact prevalences, good correlation preservation
   - Copula: Flexible, handles complex dependencies, slightly slower
   - Both methods produce similar results when correlations are moderate

4. **Best Practices**:
   - Always validate correlation matrices (positive semi-definite)
   - Check that observed correlations match targets (tolerance <0.05)
   - Compare generated data to real data when available
   - Report correlation assumptions in publications

5. **Clinical Implications**:
   - Correlation structure affects:
     * Distribution of risk across patients
     * Performance of CDRs using 'ANY' logic
     * Sample size requirements for studies
   - Always test sensitivity to correlation assumptions
""")

print("=" * 80)
print("✓ Custom Correlation Examples Complete")
print("=" * 80)
