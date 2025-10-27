"""
Bootstrap Confidence Intervals
==============================

Provides bootstrap methods for calculating confidence intervals on model
performance metrics. Uses resampling to account for uncertainty.

Author: Ahmed Azzam, MD
Institution: Department of Neuroradiology, WVU Medicine
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss


def bootstrap_metric(X: np.ndarray, y: np.ndarray,
                    metric_func: Callable,
                    n_bootstrap: int = 1000,
                    confidence_level: float = 0.95,
                    random_seed: Optional[int] = None,
                    show_progress: bool = True) -> Dict:
    """
    Calculate bootstrap confidence interval for a performance metric.
    
    Args:
        X: Feature matrix or predictions
        y: True outcomes
        metric_func: Function that takes (X, y) and returns scalar metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        show_progress: Whether to show progress bar
        
    Returns:
        results: Dict with point estimate and CI
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(y)
    bootstrap_values = []
    
    # Bootstrap resampling
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if show_progress else range(n_bootstrap)
    
    for _ in iterator:
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices] if X.ndim > 1 else X[indices]
        y_boot = y[indices]
        
        # Calculate metric
        try:
            metric_value = metric_func(X_boot, y_boot)
            bootstrap_values.append(metric_value)
        except Exception as e:
            # Skip if metric calculation fails (e.g., all one class)
            continue
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Calculate point estimate and CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {
        'point_estimate': metric_func(X, y),
        'bootstrap_mean': np.mean(bootstrap_values),
        'bootstrap_std': np.std(bootstrap_values),
        'ci_lower': np.percentile(bootstrap_values, lower_percentile),
        'ci_upper': np.percentile(bootstrap_values, upper_percentile),
        'confidence_level': confidence_level,
        'n_bootstrap': len(bootstrap_values)
    }
    
    return results


def bootstrap_c_statistic(y_true: np.ndarray, y_pred: np.ndarray,
                          n_bootstrap: int = 1000,
                          confidence_level: float = 0.95,
                          random_seed: Optional[int] = None) -> Dict:
    """
    Bootstrap confidence interval for C-statistic (AUC).
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
        
    Returns:
        results: Dict with C-statistic and CI
    """
    def c_stat_func(X, y):
        return roc_auc_score(y, X)
    
    return bootstrap_metric(
        y_pred, y_true, c_stat_func, 
        n_bootstrap, confidence_level, random_seed
    )


def bootstrap_brier_score(y_true: np.ndarray, y_pred: np.ndarray,
                         n_bootstrap: int = 1000,
                         confidence_level: float = 0.95,
                         random_seed: Optional[int] = None) -> Dict:
    """
    Bootstrap confidence interval for Brier score.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
        
    Returns:
        results: Dict with Brier score and CI
    """
    def brier_func(X, y):
        return brier_score_loss(y, X)
    
    return bootstrap_metric(
        y_pred, y_true, brier_func,
        n_bootstrap, confidence_level, random_seed
    )


def bootstrap_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray,
                                     threshold: float = 0.5,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95,
                                     random_seed: Optional[int] = None) -> Dict:
    """
    Bootstrap CIs for sensitivity and specificity at given threshold.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
        
    Returns:
        results: Dict with sensitivity and specificity CIs
    """
    def sens_func(X, y):
        y_class = (X >= threshold).astype(int)
        tp = np.sum((y == 1) & (y_class == 1))
        fn = np.sum((y == 1) & (y_class == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def spec_func(X, y):
        y_class = (X >= threshold).astype(int)
        tn = np.sum((y == 0) & (y_class == 0))
        fp = np.sum((y == 0) & (y_class == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    sens_results = bootstrap_metric(
        y_pred, y_true, sens_func,
        n_bootstrap, confidence_level, random_seed, show_progress=False
    )
    
    spec_results = bootstrap_metric(
        y_pred, y_true, spec_func,
        n_bootstrap, confidence_level, random_seed, show_progress=False
    )
    
    return {
        'sensitivity': sens_results,
        'specificity': spec_results,
        'threshold': threshold
    }


def bootstrap_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95,
                                  random_seed: Optional[int] = None) -> Dict:
    """
    Bootstrap CIs for calibration slope and intercept.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
        
    Returns:
        results: Dict with calibration metrics and CIs
    """
    from sklearn.linear_model import LogisticRegression
    
    def calibration_slope_func(X, y):
        # Convert probabilities to log-odds
        epsilon = 1e-10
        logit = np.log((X + epsilon) / (1 - X + epsilon))
        
        # Fit calibration model
        try:
            cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            cal_model.fit(logit.reshape(-1, 1), y)
            return cal_model.coef_[0][0]
        except:
            return 1.0
    
    def calibration_intercept_func(X, y):
        epsilon = 1e-10
        logit = np.log((X + epsilon) / (1 - X + epsilon))
        
        try:
            cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            cal_model.fit(logit.reshape(-1, 1), y)
            return cal_model.intercept_[0]
        except:
            return 0.0
    
    slope_results = bootstrap_metric(
        y_pred, y_true, calibration_slope_func,
        n_bootstrap, confidence_level, random_seed, show_progress=False
    )
    
    intercept_results = bootstrap_metric(
        y_pred, y_true, calibration_intercept_func,
        n_bootstrap, confidence_level, random_seed, show_progress=False
    )
    
    return {
        'calibration_slope': slope_results,
        'calibration_intercept': intercept_results
    }


def comprehensive_bootstrap_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95,
                                  random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate comprehensive bootstrap report for all metrics.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
        
    Returns:
        report_df: DataFrame with all metrics and CIs
    """
    print("Calculating bootstrap confidence intervals...")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Confidence level: {confidence_level*100}%\n")
    
    results = []
    
    # C-statistic
    print("→ C-statistic...")
    c_stat = bootstrap_c_statistic(y_true, y_pred, n_bootstrap, 
                                   confidence_level, random_seed)
    results.append({
        'Metric': 'C-statistic',
        'Point_Estimate': f"{c_stat['point_estimate']:.4f}",
        'Bootstrap_Mean': f"{c_stat['bootstrap_mean']:.4f}",
        'CI_Lower': f"{c_stat['ci_lower']:.4f}",
        'CI_Upper': f"{c_stat['ci_upper']:.4f}",
        'CI_Width': f"{c_stat['ci_upper'] - c_stat['ci_lower']:.4f}"
    })
    
    # Brier score
    print("→ Brier score...")
    brier = bootstrap_brier_score(y_true, y_pred, n_bootstrap,
                                  confidence_level, random_seed)
    results.append({
        'Metric': 'Brier Score',
        'Point_Estimate': f"{brier['point_estimate']:.4f}",
        'Bootstrap_Mean': f"{brier['bootstrap_mean']:.4f}",
        'CI_Lower': f"{brier['ci_lower']:.4f}",
        'CI_Upper': f"{brier['ci_upper']:.4f}",
        'CI_Width': f"{brier['ci_upper'] - brier['ci_lower']:.4f}"
    })
    
    # Calibration metrics
    print("→ Calibration metrics...")
    calibration = bootstrap_calibration_metrics(y_true, y_pred, n_bootstrap,
                                               confidence_level, random_seed)
    
    results.append({
        'Metric': 'Calibration Slope',
        'Point_Estimate': f"{calibration['calibration_slope']['point_estimate']:.4f}",
        'Bootstrap_Mean': f"{calibration['calibration_slope']['bootstrap_mean']:.4f}",
        'CI_Lower': f"{calibration['calibration_slope']['ci_lower']:.4f}",
        'CI_Upper': f"{calibration['calibration_slope']['ci_upper']:.4f}",
        'CI_Width': f"{calibration['calibration_slope']['ci_upper'] - calibration['calibration_slope']['ci_lower']:.4f}"
    })
    
    results.append({
        'Metric': 'Calibration Intercept',
        'Point_Estimate': f"{calibration['calibration_intercept']['point_estimate']:.4f}",
        'Bootstrap_Mean': f"{calibration['calibration_intercept']['bootstrap_mean']:.4f}",
        'CI_Lower': f"{calibration['calibration_intercept']['ci_lower']:.4f}",
        'CI_Upper': f"{calibration['calibration_intercept']['ci_upper']:.4f}",
        'CI_Width': f"{calibration['calibration_intercept']['ci_upper'] - calibration['calibration_intercept']['ci_lower']:.4f}"
    })
    
    print("✓ Bootstrap analysis complete\n")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    """
    Example usage of bootstrap confidence intervals.
    """
    print("=" * 80)
    print("Bootstrap Confidence Intervals - Example")
    print("=" * 80)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions and outcomes
    y_true = np.random.binomial(1, 0.1, n_samples)
    logit = np.random.randn(n_samples) * 2 - 2
    y_pred = 1 / (1 + np.exp(-logit))
    
    print(f"\nSample size: {n_samples}")
    print(f"Outcome prevalence: {y_true.mean():.3f}")
    print(f"Mean predicted risk: {y_pred.mean():.3f}\n")
    
    # Generate comprehensive report
    report = comprehensive_bootstrap_report(
        y_true, y_pred,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_seed=42
    )
    
    print("=" * 80)
    print("BOOTSTRAP RESULTS")
    print("=" * 80)
    print(report.to_string(index=False))
    print("=" * 80)
