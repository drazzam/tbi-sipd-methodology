"""
Phase 3: Copula Modeling
=========================

Models joint distribution of binary predictors using Gaussian copula to handle
complex dependencies. Validates that generated data preserves correlation structure.

"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
import warnings
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class CopulaModel:
    """
    Gaussian copula model for multivariate binary data generation.
    
    Uses copula theory to model joint distribution while preserving marginal
    distributions and correlation structure.
    
    Attributes:
        copula_type (str): Type of copula ('gaussian' supported)
        copula: Fitted copula model object
        fitted (bool): Whether model has been fitted
        n_features (int): Number of features/predictors
        feature_names (list): Names of features
        
    Example:
        >>> copula = CopulaModel(copula_type='gaussian')
        >>> copula.fit(X_train)
        >>> X_synthetic = copula.sample(n_samples=10000)
        >>> metrics = copula.validate_fit(X_train, X_synthetic)
    """
    
    def __init__(self, copula_type: str = 'gaussian'):
        """
        Initialize copula model.
        
        Args:
            copula_type: Type of copula to use. Currently only 'gaussian' supported.
            
        Raises:
            ValueError: If copula_type is not supported.
        """
        if copula_type.lower() != 'gaussian':
            raise ValueError("Only 'gaussian' copula is currently supported")
            
        self.copula_type = copula_type.lower()
        self.copula = None
        self.fitted = False
        self.n_features = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None, 
            verbose: bool = True) -> 'CopulaModel':
        """
        Fit Gaussian copula to observed binary data.
        
        Args:
            X: Binary data matrix (n_samples, n_features)
            feature_names: Optional list of feature names
            verbose: Whether to print progress information
            
        Returns:
            self: Fitted copula model
            
        Raises:
            ValueError: If X is not binary or has invalid shape
        """
        # Validate input
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be numpy array or pandas DataFrame")
            
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
            
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
            
        # Check if binary
        unique_vals = np.unique(X)
        if not np.array_equal(unique_vals, [0, 1]):
            warnings.warn(f"X contains non-binary values: {unique_vals}")
            
        self.n_features = X.shape[1]
        self.feature_names = (feature_names if feature_names is not None 
                             else [f"feature_{i}" for i in range(self.n_features)])
        
        if verbose:
            print(f"Fitting {self.copula_type} copula to {X.shape[0]} samples "
                  f"with {self.n_features} features...")
        
        # Convert to DataFrame for copulas library
        df = pd.DataFrame(X, columns=self.feature_names)
        
        # Initialize and fit Gaussian copula
        self.copula = GaussianMultivariate()
        
        # Fit copula with progress indication
        if verbose:
            with tqdm(total=1, desc="Fitting copula") as pbar:
                self.copula.fit(df)
                pbar.update(1)
        else:
            self.copula.fit(df)
            
        self.fitted = True
        
        if verbose:
            print("✓ Copula fitted successfully")
            
        return self
        
    def sample(self, n_samples: int, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic samples from fitted copula.
        
        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
            
        Returns:
            X_synthetic: Binary synthetic data (n_samples, n_features)
            
        Raises:
            RuntimeError: If copula has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Copula must be fitted before sampling")
            
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Generate continuous samples from copula
        samples_continuous = self.copula.sample(n_samples)
        
        # Convert to binary by thresholding at median
        # This preserves marginal probabilities for binary data
        X_synthetic = np.zeros_like(samples_continuous.values)
        for i, col in enumerate(self.feature_names):
            threshold = samples_continuous[col].median()
            X_synthetic[:, i] = (samples_continuous[col] > threshold).astype(int)
            
        return X_synthetic
        
    def validate_fit(self, X_observed: np.ndarray, X_synthetic: Optional[np.ndarray] = None,
                    n_synthetic: int = 10000, random_seed: int = 42) -> Dict[str, float]:
        """
        Validate copula fit by comparing correlation structures.
        
        Args:
            X_observed: Original observed data
            X_synthetic: Optional pre-generated synthetic data
            n_synthetic: Number of synthetic samples to generate if not provided
            random_seed: Random seed for reproducibility
            
        Returns:
            metrics: Dictionary containing validation metrics
                - 'correlation_preservation': Correlation between observed and synthetic
                - 'max_correlation_difference': Maximum absolute difference
                - 'mean_correlation_difference': Mean absolute difference
                - 'frobenius_distance': Frobenius norm of correlation difference
                
        Raises:
            RuntimeError: If copula has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Copula must be fitted before validation")
            
        # Generate synthetic data if not provided
        if X_synthetic is None:
            X_synthetic = self.sample(n_synthetic, random_seed=random_seed)
            
        # Calculate correlation matrices
        corr_observed = np.corrcoef(X_observed.T)
        corr_synthetic = np.corrcoef(X_synthetic.T)
        
        # Calculate metrics
        # Correlation between correlation matrices (flattened)
        triu_indices = np.triu_indices_from(corr_observed, k=1)
        corr_obs_flat = corr_observed[triu_indices]
        corr_syn_flat = corr_synthetic[triu_indices]
        
        correlation_preservation = np.corrcoef(corr_obs_flat, corr_syn_flat)[0, 1]
        
        # Differences in correlations
        corr_diff = np.abs(corr_observed - corr_synthetic)
        max_diff = np.max(corr_diff[triu_indices])
        mean_diff = np.mean(corr_diff[triu_indices])
        
        # Frobenius distance
        frobenius_dist = np.linalg.norm(corr_observed - corr_synthetic, 'fro')
        
        metrics = {
            'correlation_preservation': correlation_preservation,
            'max_correlation_difference': max_diff,
            'mean_correlation_difference': mean_diff,
            'frobenius_distance': frobenius_dist
        }
        
        return metrics
        
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Extract correlation matrix from fitted copula.
        
        Returns:
            correlation_matrix: Estimated correlation matrix
            
        Raises:
            RuntimeError: If copula has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Copula must be fitted first")
            
        # Extract covariance matrix from Gaussian copula
        if hasattr(self.copula, 'covariance'):
            cov_matrix = self.copula.covariance
            # Convert covariance to correlation
            std_devs = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            return corr_matrix
        else:
            warnings.warn("Could not extract correlation matrix from copula")
            return None
            
    def compare_prevalences(self, X_observed: np.ndarray, X_synthetic: np.ndarray) -> pd.DataFrame:
        """
        Compare feature prevalences between observed and synthetic data.
        
        Args:
            X_observed: Original observed data
            X_synthetic: Synthetic data
            
        Returns:
            comparison_df: DataFrame with prevalence comparisons
        """
        prevalence_obs = X_observed.mean(axis=0)
        prevalence_syn = X_synthetic.mean(axis=0)
        prevalence_diff = np.abs(prevalence_obs - prevalence_syn)
        
        comparison = pd.DataFrame({
            'feature': self.feature_names,
            'observed_prevalence': prevalence_obs,
            'synthetic_prevalence': prevalence_syn,
            'absolute_difference': prevalence_diff,
            'relative_difference_pct': (prevalence_diff / prevalence_obs * 100)
        })
        
        return comparison


def compare_copula_to_ipf(X_ipf: np.ndarray, X_copula: np.ndarray, 
                         feature_names: list) -> Dict[str, any]:
    """
    Compare IPF-generated data to copula-generated data.
    
    Args:
        X_ipf: Data generated by IPF method
        X_copula: Data generated by copula method
        feature_names: Names of features
        
    Returns:
        comparison: Dictionary containing comparison metrics and DataFrames
    """
    # Correlation matrices
    corr_ipf = np.corrcoef(X_ipf.T)
    corr_copula = np.corrcoef(X_copula.T)
    
    # Correlation preservation between methods
    triu_indices = np.triu_indices_from(corr_ipf, k=1)
    corr_ipf_flat = corr_ipf[triu_indices]
    corr_copula_flat = corr_copula[triu_indices]
    
    method_agreement = np.corrcoef(corr_ipf_flat, corr_copula_flat)[0, 1]
    
    # Prevalence comparison
    prev_ipf = X_ipf.mean(axis=0)
    prev_copula = X_copula.mean(axis=0)
    
    prevalence_df = pd.DataFrame({
        'feature': feature_names,
        'ipf_prevalence': prev_ipf,
        'copula_prevalence': prev_copula,
        'difference': np.abs(prev_ipf - prev_copula)
    })
    
    comparison = {
        'method_agreement': method_agreement,
        'prevalence_comparison': prevalence_df,
        'correlation_ipf': corr_ipf,
        'correlation_copula': corr_copula,
        'max_correlation_diff': np.max(np.abs(corr_ipf - corr_copula))
    }
    
    return comparison


if __name__ == "__main__":
    """
    Example usage of copula modeling.
    """
    print("=" * 70)
    print("Phase 3: Copula Modeling Example")
    print("=" * 70)
    
    # Generate some example binary data
    np.random.seed(42)
    n_samples = 5000
    n_features = 5
    
    # Create correlated binary data
    mean = np.zeros(n_features)
    cov = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.0],
        [0.3, 1.0, 0.4, 0.2, 0.1],
        [0.2, 0.4, 1.0, 0.3, 0.2],
        [0.1, 0.2, 0.3, 1.0, 0.3],
        [0.0, 0.1, 0.2, 0.3, 1.0]
    ])
    
    # Generate continuous data then threshold
    Z = np.random.multivariate_normal(mean, cov, n_samples)
    X = (Z > 0).astype(int)
    
    feature_names = [f'predictor_{i+1}' for i in range(n_features)]
    
    print(f"\nGenerated {n_samples} samples with {n_features} features")
    print(f"Feature prevalences: {X.mean(axis=0)}")
    
    # Fit copula
    print("\n" + "-" * 70)
    print("Fitting Copula Model...")
    print("-" * 70)
    
    copula = CopulaModel(copula_type='gaussian')
    copula.fit(X, feature_names=feature_names, verbose=True)
    
    # Generate synthetic data
    print("\n" + "-" * 70)
    print("Generating Synthetic Data...")
    print("-" * 70)
    
    X_synthetic = copula.sample(n_samples=10000, random_seed=42)
    print(f"Generated {X_synthetic.shape[0]} synthetic samples")
    
    # Validate fit
    print("\n" + "-" * 70)
    print("Validation Metrics...")
    print("-" * 70)
    
    metrics = copula.validate_fit(X, X_synthetic)
    
    print(f"Correlation Preservation: {metrics['correlation_preservation']:.4f}")
    print(f"Max Correlation Difference: {metrics['max_correlation_difference']:.4f}")
    print(f"Mean Correlation Difference: {metrics['mean_correlation_difference']:.4f}")
    print(f"Frobenius Distance: {metrics['frobenius_distance']:.4f}")
    
    # Compare prevalences
    print("\n" + "-" * 70)
    print("Prevalence Comparison...")
    print("-" * 70)
    
    prev_comparison = copula.compare_prevalences(X, X_synthetic)
    print(prev_comparison.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ Phase 3 Example Complete")
    print("=" * 70)
