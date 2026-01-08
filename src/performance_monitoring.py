"""
Model Performance Monitoring Module
Tracks model performance metrics over time and identifies degradation patterns

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from collections import deque
import logging
from datetime import datetime
from typing import Dict, List, Optional

from utils import load_artifact, save_artifact

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors model performance over time.
    
    Tracks all the usual metrics and tries to catch when things are going
    wrong before they become a big problem.
    """
    
    def __init__(self, baseline_metrics_path, config):
        self.config = config
        self.baseline_metrics = load_artifact(baseline_metrics_path)
        self.performance_history = []
        self.metrics_window = deque(maxlen=config['performance']['evaluation_window'])
        self.min_auc = config['performance']['min_auc']
        self.min_accuracy = config['performance']['min_accuracy']
        
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: Optional[np.ndarray] = None,
                            timestamp: Optional[str] = None) -> Dict:
        """
        Evaluate model performance and add to monitoring history
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC calculation)
            timestamp: Optional timestamp for this evaluation
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'n_samples': len(y_true)
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Calculate error rate
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        # Compare with baseline
        metrics['vs_baseline'] = self._compare_with_baseline(metrics)
        
        # Add to history
        self.performance_history.append(metrics)
        self.metrics_window.append(metrics)
        
        return metrics
    
    def _compare_with_baseline(self, current_metrics: Dict) -> Dict:
        """Compare current metrics with baseline"""
        comparison = {}
        
        for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']:
            if metric in current_metrics and metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics[metric]
                
                comparison[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'difference': current_value - baseline_value,
                    'relative_change': (current_value - baseline_value) / baseline_value if baseline_value > 0 else 0,
                    'degraded': current_value < baseline_value
                }
        
        return comparison
    
    def check_performance_thresholds(self, metrics: Dict) -> Dict:
        """
        Check if performance metrics fall below acceptable thresholds
        
        Returns:
            Dictionary with threshold violations and alerts
        """
        violations = {
            'threshold_violations': [],
            'alerts_triggered': False,
            'degradation_detected': False
        }
        
        # Check AUC threshold
        if 'auc' in metrics and self.min_auc:
            if metrics['auc'] < self.min_auc:
                violations['threshold_violations'].append({
                    'metric': 'auc',
                    'value': metrics['auc'],
                    'threshold': self.min_auc,
                    'severity': 'HIGH'
                })
                violations['alerts_triggered'] = True
                violations['degradation_detected'] = True
        
        # Check accuracy threshold
        if self.min_accuracy:
            if metrics['accuracy'] < self.min_accuracy:
                violations['threshold_violations'].append({
                    'metric': 'accuracy',
                    'value': metrics['accuracy'],
                    'threshold': self.min_accuracy,
                    'severity': 'HIGH'
                })
                violations['alerts_triggered'] = True
                violations['degradation_detected'] = True
        
        return violations
    
    def detect_gradual_degradation(self, window_size: int = 10) -> Dict:
        """
        Detect gradual performance degradation over a rolling window
        
        Args:
            window_size: Number of recent evaluations to consider
        """
        if len(self.performance_history) < window_size:
            return {
                'degradation_detected': False,
                'reason': 'Insufficient history for degradation detection'
            }
        
        recent_metrics = self.performance_history[-window_size:]
        
        degradation_analysis = {
            'degradation_detected': False,
            'trend_analysis': {},
            'metric_trends': {}
        }
        
        # Analyze trends for each metric
        for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']:
            values = [m.get(metric) for m in recent_metrics if metric in m]
            
            if len(values) >= 3:
                # Calculate trend (slope of linear regression)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                # Calculate correlation to detect consistent decline
                correlation = np.corrcoef(x, values)[0, 1]
                
                degradation_analysis['metric_trends'][metric] = {
                    'slope': float(slope),
                    'correlation': float(correlation),
                    'trend': 'DECLINING' if slope < -0.001 and correlation < -0.5 else 
                            'IMPROVING' if slope > 0.001 and correlation > 0.5 else 'STABLE',
                    'first_value': float(values[0]),
                    'last_value': float(values[-1]),
                    'change': float(values[-1] - values[0])
                }
                
                # Detect degradation
                if (slope < -0.001 and correlation < -0.5 and 
                    abs(values[-1] - values[0]) > 0.02):  # At least 2% drop
                    degradation_analysis['degradation_detected'] = True
        
        return degradation_analysis
    
    def detect_silent_failures(self, metrics: Dict) -> Dict:
        """
        Detect silent model failures.
        
        These are tricky - the model looks like it's working but it's actually
        broken in subtle ways. Like predicting only one class, or having terrible
        false negative rates.
        """
        silent_failure_analysis = {
            'silent_failure_detected': False,
            'indicators': []
        }
        
        # Check for class imbalance issues
        cm = np.array(metrics['confusion_matrix'])
        total = cm.sum()
        
        if total > 0:
            # Check if model is predicting only one class
            pred_distribution = cm.sum(axis=0) / total
            if np.max(pred_distribution) > 0.95:
                silent_failure_analysis['silent_failure_detected'] = True
                silent_failure_analysis['indicators'].append({
                    'type': 'CLASS_IMBALANCE',
                    'description': 'Model predicting predominantly one class',
                    'severity': 'HIGH'
                })
            
            # Check for high false negative rate (missing important cases)
            fn_rate = metrics['false_negatives'] / (metrics['false_negatives'] + metrics['true_positives'] + 1e-10)
            if fn_rate > 0.5:
                silent_failure_analysis['indicators'].append({
                    'type': 'HIGH_FALSE_NEGATIVE_RATE',
                    'description': f'High false negative rate: {fn_rate:.2%}',
                    'severity': 'MEDIUM'
                })
            
            # Check for high false positive rate
            fp_rate = metrics['false_positives'] / (metrics['false_positives'] + metrics['true_negatives'] + 1e-10)
            if fp_rate > 0.5:
                silent_failure_analysis['indicators'].append({
                    'type': 'HIGH_FALSE_POSITIVE_RATE',
                    'description': f'High false positive rate: {fp_rate:.2%}',
                    'severity': 'MEDIUM'
                })
        
        return silent_failure_analysis
    
    def get_performance_summary(self, include_history: bool = False) -> Dict:
        """Get comprehensive performance summary"""
        summary = {
            'baseline_metrics': self.baseline_metrics,
            'total_evaluations': len(self.performance_history),
            'recent_performance': None,
            'performance_trend': None
        }
        
        if self.performance_history:
            summary['recent_performance'] = self.performance_history[-1]
            summary['performance_trend'] = self.detect_gradual_degradation()
        
        if include_history:
            summary['performance_history'] = self.performance_history
        
        return summary
    
    def save_performance_history(self, artifacts_dir: str):
        """Save performance history to disk"""
        history_path = f"{artifacts_dir}/performance_history.json"
        save_artifact(self.performance_history, history_path)
        logger.info(f"Performance history saved to {history_path}")

