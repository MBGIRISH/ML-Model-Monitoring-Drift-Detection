"""
Alerting and Retraining Logic Module
Defines thresholds, triggers alerts, and manages retraining decisions

Author: M B GIRISH
Date: January 2026
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

from utils import save_artifact

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages alerts for drift detection and performance degradation.
    
    This is where we decide what's worth alerting on and what's just noise.
    """
    
    def __init__(self, config):
        self.config = config
        self.alert_history = []
        self.drift_alert_threshold = config['alerts']['drift_alert_threshold']
        self.performance_alert_threshold = config['alerts']['performance_alert_threshold']
        self.alerts_enabled = config['alerts']['enabled']
    
    def check_data_drift_alerts(self, drift_results: Dict) -> List[Dict]:
        """
        Check if data drift results trigger alerts.
        
        Returns list of alerts if any drift is detected above thresholds.
        """
        alerts = []
        
        if not self.alerts_enabled:
            return alerts
        
        # Check overall drift
        if drift_results.get('overall_drift_detected', False):
            high_drift_features = drift_results['drift_summary'].get('high_drift_features', [])
            
            if high_drift_features:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH',
                    'message': f'High data drift detected in features: {", ".join(high_drift_features)}',
                    'details': {
                        'affected_features': high_drift_features,
                        'total_features_with_drift': drift_results['drift_summary']['features_with_drift']
                    }
                }
                alerts.append(alert)
                logger.warning(f"ALERT: {alert['message']}")
            
            # Check individual feature drift
            for feature, results in drift_results['features'].items():
                if results['drift_detected']:
                    psi = results['psi']
                    if psi >= self.drift_alert_threshold:
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'FEATURE_DRIFT',
                            'severity': 'MEDIUM' if psi < self.drift_alert_threshold * 1.5 else 'HIGH',
                            'message': f'Data drift detected in feature: {feature} (PSI: {psi:.4f})',
                            'details': {
                                'feature': feature,
                                'psi': psi,
                                'kl_divergence': results['kl_divergence'],
                                'severity': results['drift_severity']
                            }
                        }
                        alerts.append(alert)
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def check_concept_drift_alerts(self, concept_drift_results: Dict) -> List[Dict]:
        """
        Check if concept drift results trigger alerts
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if not self.alerts_enabled:
            return alerts
        
        if concept_drift_results.get('concept_drift_detected', False):
            severity = concept_drift_results.get('drift_severity', 'MEDIUM')
            
            # Get performance changes
            performance_changes = concept_drift_results.get('performance_changes', {})
            auc_change = performance_changes.get('auc', {})
            
            if auc_change and auc_change.get('degraded', False):
                change_pct = abs(auc_change.get('change_percentage', 0))
                
                if change_pct >= self.performance_alert_threshold * 100:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'CONCEPT_DRIFT',
                        'severity': severity,
                        'message': f'Concept drift detected: AUC dropped by {change_pct:.2f}%',
                        'details': {
                            'baseline_auc': auc_change.get('baseline', 0),
                            'current_auc': auc_change.get('current', 0),
                            'change_percentage': change_pct,
                            'drift_severity': severity
                        }
                    }
                    alerts.append(alert)
                    logger.warning(f"ALERT: {alert['message']}")
        
        # Check error rate increase
        error_analysis = concept_drift_results.get('error_analysis', {})
        if error_analysis.get('significant_increase', False):
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'ERROR_RATE_INCREASE',
                'severity': 'MEDIUM',
                'message': f'Significant error rate increase: {error_analysis.get("error_increase_percentage", 0):.2f}%',
                'details': error_analysis
            }
            alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def check_performance_alerts(self, performance_violations: Dict) -> List[Dict]:
        """
        Check if performance threshold violations trigger alerts
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if not self.alerts_enabled:
            return alerts
        
        if performance_violations.get('alerts_triggered', False):
            violations = performance_violations.get('threshold_violations', [])
            
            for violation in violations:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'PERFORMANCE_THRESHOLD_VIOLATION',
                    'severity': violation.get('severity', 'HIGH'),
                    'message': f'{violation["metric"].upper()} below threshold: {violation["value"]:.4f} < {violation["threshold"]:.4f}',
                    'details': violation
                }
                alerts.append(alert)
                logger.warning(f"ALERT: {alert['message']}")
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts"""
        summary = {
            'total_alerts': len(self.alert_history),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'recent_alerts': self.alert_history[-10:] if len(self.alert_history) > 10 else self.alert_history
        }
        
        # Count by type
        for alert in self.alert_history:
            alert_type = alert['type']
            summary['alerts_by_type'][alert_type] = summary['alerts_by_type'].get(alert_type, 0) + 1
            
            severity = alert['severity']
            summary['alerts_by_severity'][severity] = summary['alerts_by_severity'].get(severity, 0) + 1
        
        return summary
    
    def save_alerts(self, artifacts_dir: str):
        """Save alert history to disk"""
        alerts_path = f"{artifacts_dir}/alerts_history.json"
        save_artifact(self.alert_history, alerts_path)
        logger.info(f"Alert history saved to {alerts_path}")


class RetrainingManager:
    """
    Manages retraining logic and decisions
    """
    
    def __init__(self, config):
        self.config = config
        self.retraining_history = []
        self.auto_retrain = config['retraining']['auto_retrain']
        self.retrain_on_drift = config['retraining']['retrain_on_drift']
        self.retrain_on_performance_drop = config['retraining']['retrain_on_performance_drop']
        self.min_samples_for_retrain = config['retraining']['min_samples_for_retrain']
        self.model_comparison_metric = config['retraining']['model_comparison_metric']
    
    def should_retrain(self, drift_alerts: List[Dict], concept_drift_detected: bool,
                      performance_degraded: bool, available_samples: int) -> Dict:
        """
        Determine if model should be retrained based on various conditions
        
        Returns:
            Dictionary with retraining decision and reasoning
        """
        decision = {
            'should_retrain': False,
            'reason': '',
            'priority': 'LOW',
            'conditions_met': []
        }
        
        # Check if enough samples are available
        if available_samples < self.min_samples_for_retrain:
            decision['reason'] = f'Insufficient samples for retraining: {available_samples} < {self.min_samples_for_retrain}'
            return decision
        
        # Check data drift conditions
        if self.retrain_on_drift:
            high_drift_alerts = [a for a in drift_alerts if a.get('severity') == 'HIGH']
            if high_drift_alerts:
                decision['should_retrain'] = True
                decision['conditions_met'].append('HIGH_DATA_DRIFT')
                decision['priority'] = 'HIGH'
                decision['reason'] = 'High data drift detected in multiple features'
        
        # Check concept drift conditions
        if self.retrain_on_performance_drop and concept_drift_detected:
            decision['should_retrain'] = True
            decision['conditions_met'].append('CONCEPT_DRIFT')
            if decision['priority'] != 'HIGH':
                decision['priority'] = 'MEDIUM'
            decision['reason'] = 'Concept drift detected - model performance degraded'
        
        # Check performance degradation
        if self.retrain_on_performance_drop and performance_degraded:
            decision['should_retrain'] = True
            decision['conditions_met'].append('PERFORMANCE_DEGRADATION')
            if decision['priority'] != 'HIGH':
                decision['priority'] = 'MEDIUM'
            decision['reason'] = 'Model performance below acceptable thresholds'
        
        if not decision['should_retrain']:
            decision['reason'] = 'No retraining conditions met'
        
        return decision
    
    def compare_models(self, baseline_metrics: Dict, new_model_metrics: Dict) -> Dict:
        """
        Compare baseline model with newly trained model
        
        Returns:
            Dictionary with comparison results and deployment recommendation
        """
        comparison = {
            'baseline_metrics': baseline_metrics,
            'new_model_metrics': new_model_metrics,
            'improvements': {},
            'degradations': {},
            'deploy_recommendation': 'KEEP_BASELINE',
            'reason': ''
        }
        
        metric = self.model_comparison_metric
        
        if metric in baseline_metrics and metric in new_model_metrics:
            baseline_value = baseline_metrics[metric]
            new_value = new_model_metrics[metric]
            
            improvement = new_value - baseline_value
            improvement_pct = (improvement / baseline_value) * 100 if baseline_value > 0 else 0
            
            comparison['metric_comparison'] = {
                'metric': metric,
                'baseline': baseline_value,
                'new_model': new_value,
                'improvement': improvement,
                'improvement_percentage': improvement_pct
            }
            
            # Recommend deployment if significant improvement
            if improvement_pct >= 2.0:  # At least 2% improvement
                comparison['deploy_recommendation'] = 'DEPLOY_NEW_MODEL'
                comparison['reason'] = f'New model shows {improvement_pct:.2f}% improvement in {metric}'
            elif improvement_pct >= 0:
                comparison['deploy_recommendation'] = 'DEPLOY_NEW_MODEL'
                comparison['reason'] = f'New model shows marginal improvement in {metric}'
            else:
                comparison['deploy_recommendation'] = 'KEEP_BASELINE'
                comparison['reason'] = f'New model performs worse than baseline in {metric}'
        
        # Compare all metrics
        for metric_name in ['accuracy', 'auc', 'precision', 'recall', 'f1_score']:
            if metric_name in baseline_metrics and metric_name in new_model_metrics:
                baseline_val = baseline_metrics[metric_name]
                new_val = new_model_metrics[metric_name]
                
                if new_val > baseline_val:
                    comparison['improvements'][metric_name] = {
                        'baseline': baseline_val,
                        'new_model': new_val,
                        'improvement': new_val - baseline_val
                    }
                elif new_val < baseline_val:
                    comparison['degradations'][metric_name] = {
                        'baseline': baseline_val,
                        'new_model': new_val,
                        'degradation': new_val - baseline_val
                    }
        
        return comparison
    
    def log_retraining_decision(self, decision: Dict, comparison: Optional[Dict] = None):
        """Log retraining decision to history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'comparison': comparison
        }
        self.retraining_history.append(entry)
        logger.info(f"Retraining decision logged: {decision['reason']}")
    
    def get_retraining_summary(self) -> Dict:
        """Get summary of retraining history"""
        return {
            'total_retraining_events': len(self.retraining_history),
            'recent_decisions': self.retraining_history[-5:] if len(self.retraining_history) > 5 else self.retraining_history
        }
    
    def save_retraining_history(self, artifacts_dir: str):
        """Save retraining history to disk"""
        history_path = f"{artifacts_dir}/retraining_history.json"
        save_artifact(self.retraining_history, history_path)
        logger.info(f"Retraining history saved to {history_path}")

