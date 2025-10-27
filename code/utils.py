"""
Utility Functions for sIPD Generation
======================================

Common utility functions used across all phases of synthetic individual
patient data generation.

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings


def validate_binary_matrix(X: np.ndarray, name: str = "X") -> None:
    """
    Validate that matrix contains only binary (0/1) values.
    
    Args:
        X: Matrix to validate
        name: Name of matrix for error messages
        
    Raises:
        ValueError: If matrix is not binary
    """
    if not np.all(np.isin(X, [0, 1])):
        unique_vals = np.unique(X)
        raise ValueError(
            f"{name} must contain only 0 and 1. Found values: {unique_vals}"
        )


def check_prevalences(X: np.ndarray, target_prevalences: Dict[str, float],
                     feature_names: List[str], tolerance: float = 0.02) -> Dict[str, float]:
    """
    Check if observed prevalences match target prevalences within tolerance.
    
    Args:
        X: Binary data matrix
        target_prevalences: Dict of feature_name -> target prevalence
        feature_names: List of feature names
        tolerance: Maximum acceptable deviation
        
    Returns:
        differences: Dict of feature_name -> absolute difference
        
    Raises:
        Warning: If any prevalence exceeds tolerance
    """
    observed = X.mean(axis=0)
    differences = {}
    issues = []
    
    for i, name in enumerate(feature_names):
        target = target_prevalences[name]
        diff = abs(observed[i] - target)
        differences[name] = diff
        
        if diff > tolerance:
            issues.append(
                f"{name}: observed={observed[i]:.4f}, target={target:.4f}, "
                f"diff={diff:.4f}"
            )
    
    if issues:
        warnings.warn(
            f"Prevalence mismatches exceed tolerance ({tolerance}):\n" + 
            "\n".join(issues)
        )
    
    return differences


def calculate_2x2_table(X: np.ndarray, y: np.ndarray, 
                       predictor_idx: int) -> np.ndarray:
    """
    Calculate 2×2 contingency table for predictor-outcome relationship.
    
    Args:
        X: Predictor matrix (n_samples, n_features)
        y: Outcome vector (n_samples,)
        predictor_idx: Index of predictor column
        
    Returns:
        table: 2×2 array [[a, b], [c, d]] where:
            a = both predictor and outcome present
            b = predictor present, outcome absent
            c = predictor absent, outcome present
            d = both absent
    """
    predictor = X[:, predictor_idx]
    
    a = np.sum((predictor == 1) & (y == 1))  # Both present
    b = np.sum((predictor == 1) & (y == 0))  # Predictor only
    c = np.sum((predictor == 0) & (y == 1))  # Outcome only
    d = np.sum((predictor == 0) & (y == 0))  # Both absent
    
    table = np.array([[a, b], [c, d]])
    return table


def odds_ratio_from_2x2(table: np.ndarray, correction: float = 0.5) -> Tuple[float, float]:
    """
    Calculate odds ratio and standard error from 2×2 table.
    
    Args:
        table: 2×2 contingency table [[a, b], [c, d]]
        correction: Continuity correction for zero cells
        
    Returns:
        or_value: Odds ratio
        se_log_or: Standard error of log odds ratio
    """
    a, b, c, d = table.ravel()
    
    # Apply continuity correction if any cell is 0
    if any(cell == 0 for cell in [a, b, c, d]):
        a += correction
        b += correction
        c += correction
        d += correction
    
    # Calculate OR
    or_value = (a * d) / (b * c)
    
    # Calculate SE of log(OR)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    return or_value, se_log_or


def confidence_interval_or(or_value: float, se_log_or: float, 
                          confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for odds ratio.
    
    Args:
        or_value: Odds ratio
        se_log_or: Standard error of log odds ratio
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
    """
    from scipy import stats
    
    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # CI on log scale
    log_or = np.log(or_value)
    log_ci_lower = log_or - z * se_log_or
    log_ci_upper = log_or + z * se_log_or
    
    # Convert back to OR scale
    ci_lower = np.exp(log_ci_lower)
    ci_upper = np.exp(log_ci_upper)
    
    return ci_lower, ci_upper


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.3,
                    random_seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
    """
    Split data into training and test sets with stratification.
    
    Args:
        X: Predictor matrix
        y: Outcome vector
        test_size: Proportion of data for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def create_summary_table(X: pd.DataFrame, y: np.ndarray,
                        feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create summary statistics table for dataset.
    
    Args:
        X: Predictor DataFrame or array
        y: Outcome array
        feature_names: Optional list of feature names
        
    Returns:
        summary_df: DataFrame with summary statistics
    """
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    summary_data = []
    
    for col in X.columns:
        # Overall prevalence
        prevalence = X[col].mean()
        
        # Prevalence among cases and controls
        cases_prev = X.loc[y == 1, col].mean() if (y == 1).any() else 0
        controls_prev = X.loc[y == 0, col].mean() if (y == 0).any() else 0
        
        # Calculate 2×2 table and OR
        predictor_idx = X.columns.get_loc(col)
        table = calculate_2x2_table(X.values, y, predictor_idx)
        or_value, se_log_or = odds_ratio_from_2x2(table)
        ci_lower, ci_upper = confidence_interval_or(or_value, se_log_or)
        
        summary_data.append({
            'Feature': col,
            'Overall_Prevalence': f"{prevalence:.3f}",
            'Cases_Prevalence': f"{cases_prev:.3f}",
            'Controls_Prevalence': f"{controls_prev:.3f}",
            'Odds_Ratio': f"{or_value:.2f}",
            '95%_CI': f"({ci_lower:.2f}, {ci_upper:.2f})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str,
                feature_names: Optional[List[str]] = None) -> None:
    """
    Save synthetic dataset to CSV file.
    
    Args:
        X: Predictor matrix
        y: Outcome vector
        filepath: Path to save CSV
        feature_names: Optional list of feature names
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['outcome'] = y
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Outcome prevalence: {y.mean():.4f}")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load synthetic dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        X: Predictor matrix
        y: Outcome vector
        feature_names: List of feature names
    """
    df = pd.read_csv(filepath)
    
    # Assume last column is outcome
    feature_names = df.columns[:-1].tolist()
    X = df[feature_names].values
    y = df.iloc[:, -1].values
    
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {X.shape}")
    print(f"Outcome prevalence: {y.mean():.4f}")
    
    return X, y, feature_names


def format_performance_metrics(metrics: Dict[str, float], 
                               decimals: int = 4) -> pd.DataFrame:
    """
    Format performance metrics as a presentable DataFrame.
    
    Args:
        metrics: Dictionary of metric_name -> value
        decimals: Number of decimal places
        
    Returns:
        metrics_df: Formatted DataFrame
    """
    formatted = []
    
    for key, value in metrics.items():
        # Format key (replace underscores, title case)
        formatted_key = key.replace('_', ' ').title()
        
        # Format value
        if isinstance(value, float):
            if 0 < value < 0.01:
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = f"{value:.{decimals}f}"
        else:
            formatted_value = str(value)
        
        formatted.append({
            'Metric': formatted_key,
            'Value': formatted_value
        })
    
    metrics_df = pd.DataFrame(formatted)
    return metrics_df


if __name__ == "__main__":
    """
    Example usage of utility functions.
    """
    print("=" * 70)
    print("Utility Functions Example")
    print("=" * 70)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.binomial(1, 0.3, (n_samples, n_features))
    y = np.random.binomial(1, 0.05, n_samples)
    
    feature_names = [f'predictor_{i+1}' for i in range(n_features)]
    
    print(f"\nGenerated {n_samples} samples with {n_features} features")
    
    # Validate binary matrix
    print("\n" + "-" * 70)
    print("Binary Validation...")
    print("-" * 70)
    try:
        validate_binary_matrix(X, "X")
        print("✓ X is binary")
        validate_binary_matrix(y, "y")
        print("✓ y is binary")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Create summary table
    print("\n" + "-" * 70)
    print("Summary Statistics...")
    print("-" * 70)
    
    summary = create_summary_table(
        pd.DataFrame(X, columns=feature_names), y, feature_names
    )
    print(summary.to_string(index=False))
    
    # Train-test split
    print("\n" + "-" * 70)
    print("Train-Test Split...")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.3)
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train outcome prevalence: {y_train.mean():.4f}")
    print(f"Test outcome prevalence: {y_test.mean():.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Utilities Example Complete")
    print("=" * 70)
