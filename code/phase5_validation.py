"""
Phase 5: Clinical Decision Rule (CDR) Validation
=================================================

Validates synthetic data by applying clinical decision rules and comparing
performance to published studies. Evaluates model predictions across multiple
thresholds.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


# Clinical Decision Rule Definitions
CDR_DEFINITIONS = {
    'CCHR': {
        'name': 'Canadian CT Head Rule',
        'predictors': ['age_65_plus', 'dangerous_mechanism', 'vomiting_2_plus', 
                      'skull_fracture', 'gcs_less_than_15'],
        'logic': 'ANY',  # Recommend CT if ANY predictor present
        'published_sensitivity': 0.984,
        'published_specificity': None,
        'published_ct_rate': 0.540
    },
    'NOC': {
        'name': 'New Orleans Criteria',
        'predictors': ['headache', 'vomiting_2_plus', 'age_65_plus', 
                      'intoxication', 'seizure', 'skull_fracture', 
                      'gcs_less_than_15'],
        'logic': 'ANY',
        'published_sensitivity': 0.977,
        'published_specificity': None,
        'published_ct_rate': 0.860
    },
    'NEXUS_II': {
        'name': 'NEXUS-II',
        'predictors': ['age_65_plus', 'skull_fracture', 'gcs_less_than_15', 
                      'vomiting_2_plus', 'dangerous_mechanism'],
        'logic': 'ANY',
        'published_sensitivity': 0.990,
        'published_specificity': 0.192,
        'published_ct_rate': 0.808
    },
    'CHIP': {
        'name': 'CT in Head Injury Patients',
        'predictors': ['gcs_less_than_15', 'skull_fracture', 'vomiting_2_plus', 
                      'age_65_plus'],
        'logic': 'ANY',
        'published_sensitivity': 0.988,
        'published_specificity': None,
        'published_ct_rate': 0.519
    }
}


def apply_cdr(X: pd.DataFrame, cdr_name: str) -> np.ndarray:
    """
    Apply clinical decision rule to predict CT recommendation.
    
    Args:
        X: DataFrame with predictor columns
        cdr_name: Name of CDR ('CCHR', 'NOC', 'NEXUS_II', 'CHIP')
        
    Returns:
        predictions: Binary array (1 = recommend CT, 0 = no CT)
        
    Raises:
        ValueError: If CDR name is invalid or required columns missing
    """
    if cdr_name not in CDR_DEFINITIONS:
        raise ValueError(f"Unknown CDR: {cdr_name}. Valid options: {list(CDR_DEFINITIONS.keys())}")
        
    cdr = CDR_DEFINITIONS[cdr_name]
    required_cols = cdr['predictors']
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {cdr_name}: {missing_cols}")
    
    # Apply logic (currently only 'ANY' logic implemented)
    if cdr['logic'] == 'ANY':
        # Recommend CT if ANY predictor is positive
        predictions = X[required_cols].any(axis=1).astype(int).values
    else:
        raise NotImplementedError(f"Logic type '{cdr['logic']}' not implemented")
        
    return predictions


def calculate_cdr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for CDR predictions.
    
    Args:
        y_true: True outcomes (1 = ciTBI, 0 = no ciTBI)
        y_pred: CDR predictions (1 = recommend CT, 0 = no CT)
        
    Returns:
        metrics: Dictionary of performance metrics
            - sensitivity: True positive rate
            - specificity: True negative rate
            - npv: Negative predictive value
            - ppv: Positive predictive value
            - ct_rate: Proportion recommended for CT
            - missed_injuries: Number of missed ciTBI cases
            - missed_injury_rate: Proportion of ciTBI cases missed
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    n_total = len(y_true)
    n_positive = y_true.sum()
    n_negative = n_total - n_positive
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    ct_rate = (tp + fp) / n_total
    missed_injuries = fn
    missed_injury_rate = fn / n_positive if n_positive > 0 else 0
    
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'npv': npv,
        'ppv': ppv,
        'ct_rate': ct_rate,
        'missed_injuries': missed_injuries,
        'missed_injury_rate': missed_injury_rate,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    return metrics


def compare_to_published(synthetic_metrics: Dict[str, float], 
                        cdr_name: str) -> Dict[str, float]:
    """
    Compare synthetic CDR performance to published results.
    
    Args:
        synthetic_metrics: Metrics calculated on synthetic data
        cdr_name: Name of CDR
        
    Returns:
        comparison: Dictionary with differences from published values
    """
    published = CDR_DEFINITIONS[cdr_name]
    
    comparison = {
        'cdr_name': cdr_name,
        'synthetic_sensitivity': synthetic_metrics['sensitivity'],
        'published_sensitivity': published['published_sensitivity'],
        'sensitivity_difference': abs(synthetic_metrics['sensitivity'] - 
                                     published['published_sensitivity']),
        'synthetic_ct_rate': synthetic_metrics['ct_rate'],
        'published_ct_rate': published['published_ct_rate'],
        'ct_rate_difference': abs(synthetic_metrics['ct_rate'] - 
                                 published['published_ct_rate'])
    }
    
    if published['published_specificity'] is not None:
        comparison['synthetic_specificity'] = synthetic_metrics['specificity']
        comparison['published_specificity'] = published['published_specificity']
        comparison['specificity_difference'] = abs(synthetic_metrics['specificity'] - 
                                                  published['published_specificity'])
    
    return comparison


def validate_all_cdrs(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Validate all CDRs on synthetic data.
    
    Args:
        X: DataFrame with predictor columns
        y: True outcomes
        
    Returns:
        results_df: DataFrame with validation results for all CDRs
    """
    results = []
    
    for cdr_name in CDR_DEFINITIONS.keys():
        try:
            # Apply CDR
            y_pred = apply_cdr(X, cdr_name)
            
            # Calculate metrics
            metrics = calculate_cdr_metrics(y, y_pred)
            
            # Compare to published
            comparison = compare_to_published(metrics, cdr_name)
            
            # Combine results
            result = {
                'CDR': CDR_DEFINITIONS[cdr_name]['name'],
                'Sensitivity': f"{metrics['sensitivity']:.3f}",
                'Specificity': f"{metrics['specificity']:.3f}" if metrics['specificity'] else 'N/A',
                'NPV': f"{metrics['npv']:.3f}",
                'CT_Rate': f"{metrics['ct_rate']:.1%}",
                'Published_Sens': f"{comparison['published_sensitivity']:.3f}",
                'Sens_Diff': f"{comparison['sensitivity_difference']:.3f}",
                'Published_CT_Rate': f"{comparison['published_ct_rate']:.1%}",
                'CT_Rate_Diff': f"{abs(metrics['ct_rate'] - comparison['published_ct_rate']):.1%}"
            }
            
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"Error validating {cdr_name}: {str(e)}")
            
    results_df = pd.DataFrame(results)
    return results_df


def calculate_model_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        
    Returns:
        metrics: Dictionary with performance metrics
            - c_statistic: Area under ROC curve
            - brier_score: Brier score (lower is better)
            - calibration_slope: Slope of calibration curve
            - calibration_intercept: Intercept of calibration curve
    """
    # C-statistic (AUC)
    c_stat = roc_auc_score(y_true, y_prob)
    
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Calibration (fit logistic regression of outcomes on logit of predictions)
    from sklearn.linear_model import LogisticRegression
    
    # Convert probabilities to log-odds
    epsilon = 1e-10  # Avoid log(0)
    logit_pred = np.log((y_prob + epsilon) / (1 - y_prob + epsilon))
    logit_pred = logit_pred.reshape(-1, 1)
    
    # Fit calibration model
    cal_model = LogisticRegression(penalty=None, solver='lbfgs')
    cal_model.fit(logit_pred, y_true)
    
    calibration_slope = cal_model.coef_[0][0]
    calibration_intercept = cal_model.intercept_[0]
    
    # O/E ratio (observed/expected)
    oe_ratio = y_true.mean() / y_prob.mean()
    
    metrics = {
        'c_statistic': c_stat,
        'brier_score': brier,
        'calibration_slope': calibration_slope,
        'calibration_intercept': calibration_intercept,
        'oe_ratio': oe_ratio
    }
    
    return metrics


def calculate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                thresholds: List[float]) -> pd.DataFrame:
    """
    Calculate metrics at multiple probability thresholds.
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        thresholds: List of probability thresholds (as percentages, e.g., [1, 2, 5, 10])
        
    Returns:
        threshold_df: DataFrame with metrics at each threshold
    """
    results = []
    
    n_positive = y_true.sum()
    
    for threshold_pct in thresholds:
        threshold = threshold_pct / 100  # Convert to probability
        
        # Apply threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_cdr_metrics(y_true, y_pred)
        
        # Calculate NNS (number needed to scan)
        nns = 1 / (y_prob[y_pred == 1].mean()) if y_pred.sum() > 0 else np.inf
        
        result = {
            'Threshold_%': threshold_pct,
            'Sensitivity_%': metrics['sensitivity'] * 100,
            'Specificity_%': metrics['specificity'] * 100,
            'NPV_%': metrics['npv'] * 100,
            'CT_Rate_%': metrics['ct_rate'] * 100,
            'NNS': nns
        }
        
        results.append(result)
        
    threshold_df = pd.DataFrame(results)
    return threshold_df


def validate_data_integrity(X: pd.DataFrame, y: np.ndarray) -> Dict[str, any]:
    """
    Perform integrity checks on synthetic data.
    
    Args:
        X: Predictor DataFrame
        y: Outcome array
        
    Returns:
        report: Dictionary with validation results
    """
    report = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'missing_values': X.isnull().sum().to_dict(),
        'outcome_prevalence': y.mean(),
        'feature_prevalences': X.mean().to_dict(),
        'all_binary': all((X[col].isin([0, 1]).all()) for col in X.columns),
        'outcome_binary': np.array_equal(np.unique(y), [0, 1])
    }
    
    # Check for any data quality issues
    issues = []
    
    if report['missing_values']:
        issues.append("Missing values detected")
        
    if not report['all_binary']:
        issues.append("Non-binary values in predictors")
        
    if not report['outcome_binary']:
        issues.append("Non-binary outcome values")
        
    if report['outcome_prevalence'] < 0.001 or report['outcome_prevalence'] > 0.05:
        issues.append(f"Unusual outcome prevalence: {report['outcome_prevalence']:.4f}")
        
    report['issues'] = issues if issues else ['No issues detected']
    report['valid'] = len(issues) == 0
    
    return report


if __name__ == "__main__":
    """
    Example usage of CDR validation.
    """
    print("=" * 70)
    print("Phase 5: CDR Validation Example")
    print("=" * 70)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 10000
    
    # Create example predictor data
    predictor_names = ['age_65_plus', 'vomiting_2_plus', 'gcs_less_than_15',
                      'skull_fracture', 'dangerous_mechanism', 'headache',
                      'intoxication', 'seizure', 'anticoagulant']
    
    prevalences = [0.15, 0.08, 0.12, 0.03, 0.22, 0.45, 0.18, 0.02, 0.06]
    
    X_dict = {}
    for name, prev in zip(predictor_names, prevalences):
        X_dict[name] = np.random.binomial(1, prev, n_samples)
    
    X = pd.DataFrame(X_dict)
    
    # Generate outcomes (using simplified model)
    log_odds = -4.23 + 0.884 * X['age_65_plus'] + 2.242 * X['skull_fracture']
    prob = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, prob)
    
    print(f"\nGenerated {n_samples} samples")
    print(f"Outcome prevalence: {y.mean():.4f}")
    
    # Data integrity check
    print("\n" + "-" * 70)
    print("Data Integrity Check...")
    print("-" * 70)
    
    integrity = validate_data_integrity(X, y)
    print(f"Valid: {integrity['valid']}")
    print(f"Issues: {', '.join(integrity['issues'])}")
    
    # Validate CDRs
    print("\n" + "-" * 70)
    print("CDR Validation Results...")
    print("-" * 70)
    
    cdr_results = validate_all_cdrs(X, y)
    print(cdr_results.to_string(index=False))
    
    # Model metrics
    print("\n" + "-" * 70)
    print("Model Performance Metrics...")
    print("-" * 70)
    
    model_metrics = calculate_model_metrics(y, prob)
    for key, value in model_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Threshold analysis
    print("\n" + "-" * 70)
    print("Threshold Analysis...")
    print("-" * 70)
    
    threshold_metrics = calculate_threshold_metrics(y, prob, [1, 2, 3, 5, 10])
    print(threshold_metrics.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ Phase 5 Example Complete")
    print("=" * 70)
