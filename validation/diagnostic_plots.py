"""
Diagnostic Plots for Model Validation
=====================================

Creates publication-quality diagnostic plots for validating synthetic data
and model performance: ROC curves, calibration plots, distribution plots.

Author: Ahmed Azzam, MD
Institution: Department of Neuroradiology, WVU Medicine
Date: January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, calibration_curve
from sklearn.calibration import CalibrationDisplay
from scipy import stats
from typing import Optional, Tuple
import warnings


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "ROC Curve",
                   save_path: Optional[str] = None,
                   show_plot: bool = True) -> Tuple[plt.Figure, float]:
    """
    Create ROC curve with AUC.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)
        show_plot: Whether to display plot
        
    Returns:
        fig: Matplotlib figure
        auc_score: Area under curve
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ROC curve
    ax.plot(fpr, tpr, color='#2563eb', linewidth=2, 
            label=f'ROC Curve (AUC = {auc_score:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', 
            linewidth=1, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, auc_score


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray,
                          n_bins: int = 10,
                          title: str = "Calibration Plot",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
    """
    Create calibration plot.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display plot
        
    Returns:
        fig: Matplotlib figure
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, 
                                             strategy='quantile')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
            color='#2563eb', label='Model Calibration')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', 
            linewidth=1, label='Perfect Calibration')
    
    # Formatting
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_risk_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Risk Distribution",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
    """
    Plot distribution of predicted risks by outcome.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display plot
        
    Returns:
        fig: Matplotlib figure
    """
    # Separate predictions by outcome
    risks_cases = y_pred[y_true == 1]
    risks_controls = y_pred[y_true == 0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograms
    ax.hist(risks_controls, bins=50, alpha=0.6, color='#3b82f6', 
            label=f'No Event (N={len(risks_controls):,})', density=True)
    ax.hist(risks_cases, bins=50, alpha=0.6, color='#ef4444',
            label=f'Event (N={len(risks_cases):,})', density=True)
    
    # Formatting
    ax.set_xlabel('Predicted Risk', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risk distribution plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_correlation_heatmap(correlation_matrix: np.ndarray,
                            predictor_names: list,
                            title: str = "Correlation Matrix",
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
    """
    Create correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix
        predictor_names: List of predictor names
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display plot
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12)
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(predictor_names)))
    ax.set_yticks(np.arange(len(predictor_names)))
    ax.set_xticklabels(predictor_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(predictor_names, fontsize=10)
    
    # Add correlation values
    for i in range(len(predictor_names)):
        for j in range(len(predictor_names)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_predictor_prevalences(X: pd.DataFrame,
                              target_prevalences: Optional[dict] = None,
                              title: str = "Predictor Prevalences",
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
    """
    Plot observed vs target predictor prevalences.
    
    Args:
        X: Predictor DataFrame
        target_prevalences: Dict of predictor -> target prevalence
        title: Plot title
        save_path: Path to save figure
        show_plot: Whether to display plot
        
    Returns:
        fig: Matplotlib figure
    """
    # Calculate observed prevalences
    observed = X.mean()
    predictor_names = X.columns.tolist()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(predictor_names))
    
    # Observed prevalences
    ax.bar(x_pos, observed, alpha=0.7, color='#3b82f6', 
           label='Observed', width=0.4)
    
    # Target prevalences (if provided)
    if target_prevalences is not None:
        target_vals = [target_prevalences.get(name, 0) for name in predictor_names]
        ax.bar(x_pos + 0.4, target_vals, alpha=0.7, color='#10b981',
               label='Target', width=0.4)
    
    # Formatting
    ax.set_xlabel('Predictor', fontsize=12)
    ax.set_ylabel('Prevalence', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + 0.2 if target_prevalences else x_pos)
    ax.set_xticklabels(predictor_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(observed.max(), 
                       max(target_prevalences.values()) if target_prevalences else 0) * 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prevalence plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_comprehensive_diagnostic_report(X: pd.DataFrame, y: np.ndarray, 
                                          y_pred: np.ndarray,
                                          correlation_matrix: Optional[np.ndarray] = None,
                                          target_prevalences: Optional[dict] = None,
                                          output_dir: str = "diagnostic_plots",
                                          show_plots: bool = False) -> None:
    """
    Generate comprehensive diagnostic report with all plots.
    
    Args:
        X: Predictor DataFrame
        y: True outcomes
        y_pred: Predicted probabilities
        correlation_matrix: Optional correlation matrix
        target_prevalences: Optional target prevalences
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    # ROC Curve
    print("\n1. ROC Curve...")
    plot_roc_curve(y, y_pred, 
                  save_path=f"{output_dir}/roc_curve.png",
                  show_plot=show_plots)
    
    # Calibration Plot
    print("2. Calibration Plot...")
    plot_calibration_curve(y, y_pred,
                          save_path=f"{output_dir}/calibration_plot.png",
                          show_plot=show_plots)
    
    # Risk Distribution
    print("3. Risk Distribution...")
    plot_risk_distribution(y, y_pred,
                          save_path=f"{output_dir}/risk_distribution.png",
                          show_plot=show_plots)
    
    # Predictor Prevalences
    print("4. Predictor Prevalences...")
    plot_predictor_prevalences(X, target_prevalences,
                              save_path=f"{output_dir}/predictor_prevalences.png",
                              show_plot=show_plots)
    
    # Correlation Heatmap (if provided)
    if correlation_matrix is not None:
        print("5. Correlation Heatmap...")
        plot_correlation_heatmap(correlation_matrix, X.columns.tolist(),
                                save_path=f"{output_dir}/correlation_heatmap.png",
                                show_plot=show_plots)
    
    print(f"\n✓ All diagnostic plots saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    """
    Example usage of diagnostic plots.
    """
    print("=" * 80)
    print("Diagnostic Plots - Example")
    print("=" * 80)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 5000
    
    # Create predictors
    predictor_names = ['age_65_plus', 'gcs_less_than_15', 'skull_fracture']
    X_dict = {
        'age_65_plus': np.random.binomial(1, 0.15, n_samples),
        'gcs_less_than_15': np.random.binomial(1, 0.12, n_samples),
        'skull_fracture': np.random.binomial(1, 0.03, n_samples)
    }
    X_df = pd.DataFrame(X_dict)
    
    # Generate outcomes and predictions
    log_odds = -4.5 + 0.88 * X_df['age_65_plus'] + 2.24 * X_df['skull_fracture']
    y_pred = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, y_pred)
    
    # Correlation matrix
    corr_matrix = np.corrcoef(X_df.T)
    
    # Target prevalences
    target_prev = {
        'age_65_plus': 0.15,
        'gcs_less_than_15': 0.12,
        'skull_fracture': 0.03
    }
    
    # Generate all plots
    create_comprehensive_diagnostic_report(
        X_df, y, y_pred,
        correlation_matrix=corr_matrix,
        target_prevalences=target_prev,
        output_dir="example_diagnostic_plots",
        show_plots=False
    )
    
    print("\n✓ Example complete - check 'example_diagnostic_plots/' directory")
