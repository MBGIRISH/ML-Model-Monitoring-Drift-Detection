"""
Data Drift Detection Module
Implements statistical drift metrics: PSI, KL-Divergence, and distribution comparisons

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import entropy
import logging
from typing import Dict, List, Tuple

from utils import load_artifact

logger = logging.getLogger(__name__)

class DataDriftDetector:
    """
    Detects data drift using multiple statistical methods:
    - Population Stability Index (PSI)
    - KL-Divergence
    - Distribution comparisons
    """
    
    def __init__(self, baseline_distributions_path, config):
        self.config = config
        self.baseline_distributions = load_artifact(baseline_distributions_path)
        self.psi_threshold = config['data_drift']['psi_threshold']
        self.kl_threshold = config['data_drift']['kl_divergence_threshold']
        
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI is a common metric in finance/ML for detecting distribution shifts.
        General rule of thumb:
        - PSI < 0.1: Looks good, no major changes
        - 0.1 <= PSI < 0.2: Something's changing, keep an eye on it
        - PSI >= 0.2: Significant drift - probably need to retrain
        """
        # Create bins based on expected distribution
        breakpoints = np.linspace(min(expected.min(), actual.min()), 
                                 max(expected.max(), actual.max()), bins + 1)
        
        # Calculate expected distribution
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        expected_percents = np.clip(expected_percents, 0.0001, 1.0)  # Avoid division by zero
        
        # Calculate actual distribution
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        actual_percents = np.clip(actual_percents, 0.0001, 1.0)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                    np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def calculate_kl_divergence(self, expected: np.ndarray, actual: np.ndarray, bins: int = 50) -> float:
        """
        Calculate KL-Divergence between two distributions
        
        KL-Divergence measures how different two probability distributions are.
        Higher values indicate greater drift.
        """
        # Create bins
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate histograms
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        
        # Normalize to probabilities
        expected_probs = expected_hist / (expected_hist.sum() + 1e-10)
        actual_probs = actual_hist / (actual_hist.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0) - learned this the hard way!
        epsilon = 1e-10
        expected_probs = expected_probs + epsilon
        actual_probs = actual_probs + epsilon
        
        # Normalize again
        expected_probs = expected_probs / expected_probs.sum()
        actual_probs = actual_probs / actual_probs.sum()
        
        # Calculate KL-Divergence
        kl_div = entropy(actual_probs, expected_probs)
        
        return float(kl_div)
    
    def calculate_statistical_test(self, expected: np.ndarray, actual: np.ndarray) -> Dict:
        """
        Perform statistical tests to compare distributions:
        - Kolmogorov-Smirnov test
        - Mann-Whitney U test (for non-parametric comparison)
        """
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(expected, actual)
        
        # Mann-Whitney U test
        mw_statistic, mw_pvalue = stats.mannwhitneyu(expected, actual, alternative='two-sided')
        
        return {
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'ks_significant': ks_pvalue < 0.05,
            'mw_statistic': float(mw_statistic),
            'mw_pvalue': float(mw_pvalue),
            'mw_significant': mw_pvalue < 0.05
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift for all features in current data
        
        Returns:
            Dictionary with drift scores and alerts for each feature
        """
        drift_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'features': {},
            'overall_drift_detected': False,
            'drift_summary': {
                'total_features': 0,
                'features_with_drift': 0,
                'high_drift_features': []
            }
        }
        
        for feature in current_data.columns:
            if feature not in self.baseline_distributions:
                logger.warning(f"Feature {feature} not found in baseline distributions. Skipping.")
                continue
            
            # Get baseline distribution info
            baseline_info = self.baseline_distributions[feature]
            current_values = current_data[feature].values
            
            # Reconstruct baseline distribution (approximate from histogram)
            baseline_hist = np.array(baseline_info['histogram'])
            baseline_edges = np.array(baseline_info['histogram_edges'])
            baseline_mean = baseline_info['mean']
            baseline_std = baseline_info['std']
            
            # Generate synthetic baseline data from distribution
            # Using normal approximation for continuous features
            n_samples = len(current_values)
            baseline_synthetic = np.random.normal(
                baseline_mean, 
                baseline_std, 
                size=n_samples
            )
            
            # Calculate drift metrics
            psi = self.calculate_psi(baseline_synthetic, current_values)
            kl_div = self.calculate_kl_divergence(baseline_synthetic, current_values)
            statistical_tests = self.calculate_statistical_test(baseline_synthetic, current_values)
            
            # Determine if drift is detected
            drift_detected = (
                psi >= self.psi_threshold or 
                kl_div >= self.kl_threshold or
                statistical_tests['ks_significant']
            )
            
            drift_results['features'][feature] = {
                'psi': psi,
                'kl_divergence': kl_div,
                'statistical_tests': statistical_tests,
                'drift_detected': drift_detected,
                'drift_severity': self._classify_drift_severity(psi, kl_div),
                'baseline_stats': {
                    'mean': baseline_mean,
                    'std': baseline_std
                },
                'current_stats': {
                    'mean': float(np.mean(current_values)),
                    'std': float(np.std(current_values))
                }
            }
            
            if drift_detected:
                drift_results['drift_summary']['features_with_drift'] += 1
                if psi >= self.psi_threshold * 1.5:  # High drift threshold
                    drift_results['drift_summary']['high_drift_features'].append(feature)
        
        drift_results['drift_summary']['total_features'] = len(drift_results['features'])
        drift_results['overall_drift_detected'] = (
            drift_results['drift_summary']['features_with_drift'] > 0
        )
        
        return drift_results
    
    def _classify_drift_severity(self, psi: float, kl_div: float) -> str:
        """Classify drift severity based on metrics"""
        if psi >= 0.2 or kl_div >= 0.5:
            return "HIGH"
        elif psi >= 0.1 or kl_div >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_drift_summary(self, drift_results: Dict) -> str:
        """Generate human-readable drift summary"""
        summary = f"\n{'='*60}\n"
        summary += f"DATA DRIFT DETECTION SUMMARY\n"
        summary += f"{'='*60}\n"
        summary += f"Timestamp: {drift_results['timestamp']}\n"
        summary += f"Total Features Monitored: {drift_results['drift_summary']['total_features']}\n"
        summary += f"Features with Drift: {drift_results['drift_summary']['features_with_drift']}\n"
        summary += f"High Drift Features: {len(drift_results['drift_summary']['high_drift_features'])}\n"
        summary += f"\nDrift Details:\n"
        summary += f"{'-'*60}\n"
        
        for feature, results in drift_results['features'].items():
            if results['drift_detected']:
                summary += f"\n{feature}:\n"
                summary += f"  PSI: {results['psi']:.4f} (threshold: {self.psi_threshold})\n"
                summary += f"  KL-Divergence: {results['kl_divergence']:.4f} (threshold: {self.kl_threshold})\n"
                summary += f"  Severity: {results['drift_severity']}\n"
                summary += f"  Baseline Mean: {results['baseline_stats']['mean']:.4f}\n"
                summary += f"  Current Mean: {results['current_stats']['mean']:.4f}\n"
        
        summary += f"\n{'='*60}\n"
        
        return summary


class ConceptDriftDetector:
    """
    Detects concept drift by analyzing changes in feature-target relationships
    and model performance over time.
    
    Concept Drift vs Data Drift:
    - Data Drift: Changes in feature distributions (X changes)
    - Concept Drift: Changes in the relationship between features and target (P(Y|X) changes)
    """
    
    def __init__(self, baseline_metrics_path, config):
        self.config = config
        self.baseline_metrics = load_artifact(baseline_metrics_path)
        self.performance_decay_threshold = config['concept_drift']['performance_decay_threshold']
        self.auc_drop_threshold = config['concept_drift']['auc_drop_threshold']
        
    def detect_concept_drift(self, current_metrics: Dict, predictions: np.ndarray = None, 
                            actual: np.ndarray = None) -> Dict:
        """
        Detect concept drift by comparing current performance with baseline
        
        Args:
            current_metrics: Dictionary with current model performance metrics
            predictions: Model predictions (optional, for error analysis)
            actual: Actual labels (optional, for error analysis)
        """
        concept_drift_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': current_metrics,
            'performance_changes': {},
            'concept_drift_detected': False,
            'drift_severity': 'NONE'
        }
        
        # Compare key metrics
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1_score']:
            if metric in self.baseline_metrics and metric in current_metrics:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics[metric]
                
                if baseline_value > 0:
                    change = (current_value - baseline_value) / baseline_value
                    change_pct = change * 100
                    
                    concept_drift_results['performance_changes'][metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'absolute_change': current_value - baseline_value,
                        'relative_change': change,
                        'change_percentage': change_pct,
                        'degraded': change < 0
                    }
        
        # Determine if concept drift is detected
        auc_change = concept_drift_results['performance_changes'].get('auc', {})
        if auc_change:
            auc_drop = abs(auc_change.get('relative_change', 0)) if auc_change.get('degraded', False) else 0
            
            if auc_drop >= self.performance_decay_threshold:
                concept_drift_results['concept_drift_detected'] = True
                if auc_drop >= self.performance_decay_threshold * 2:
                    concept_drift_results['drift_severity'] = 'HIGH'
                else:
                    concept_drift_results['drift_severity'] = 'MEDIUM'
        
        # Error distribution analysis (if predictions and actuals provided)
        if predictions is not None and actual is not None:
            errors = predictions != actual
            error_rate = np.mean(errors)
            baseline_error_rate = 1 - self.baseline_metrics.get('accuracy', 0)
            
            error_increase = error_rate - baseline_error_rate
            error_increase_pct = (error_increase / baseline_error_rate) * 100 if baseline_error_rate > 0 else 0
            
            concept_drift_results['error_analysis'] = {
                'baseline_error_rate': baseline_error_rate,
                'current_error_rate': error_rate,
                'error_increase': error_increase,
                'error_increase_percentage': error_increase_pct,
                'significant_increase': error_increase_pct >= self.config['concept_drift']['error_rate_increase'] * 100
            }
            
            if concept_drift_results['error_analysis']['significant_increase']:
                concept_drift_results['concept_drift_detected'] = True
                if concept_drift_results['drift_severity'] == 'NONE':
                    concept_drift_results['drift_severity'] = 'MEDIUM'
        
        return concept_drift_results
    
    def get_concept_drift_summary(self, concept_drift_results: Dict) -> str:
        """Generate human-readable concept drift summary"""
        summary = f"\n{'='*60}\n"
        summary += f"CONCEPT DRIFT DETECTION SUMMARY\n"
        summary += f"{'='*60}\n"
        summary += f"Timestamp: {concept_drift_results['timestamp']}\n"
        summary += f"Concept Drift Detected: {concept_drift_results['concept_drift_detected']}\n"
        summary += f"Drift Severity: {concept_drift_results['drift_severity']}\n"
        summary += f"\nPerformance Changes:\n"
        summary += f"{'-'*60}\n"
        
        for metric, changes in concept_drift_results['performance_changes'].items():
            summary += f"\n{metric.upper()}:\n"
            summary += f"  Baseline: {changes['baseline']:.4f}\n"
            summary += f"  Current: {changes['current']:.4f}\n"
            summary += f"  Change: {changes['change_percentage']:.2f}%\n"
            summary += f"  Status: {'DEGRADED' if changes['degraded'] else 'IMPROVED/STABLE'}\n"
        
        if 'error_analysis' in concept_drift_results:
            error_analysis = concept_drift_results['error_analysis']
            summary += f"\nError Analysis:\n"
            summary += f"  Baseline Error Rate: {error_analysis['baseline_error_rate']:.4f}\n"
            summary += f"  Current Error Rate: {error_analysis['current_error_rate']:.4f}\n"
            summary += f"  Error Increase: {error_analysis['error_increase_percentage']:.2f}%\n"
        
        summary += f"\n{'='*60}\n"
        
        return summary

