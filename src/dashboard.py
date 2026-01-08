"""
Dashboard and Reporting Module
Creates visualizations and reports for model monitoring

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Optional
import logging

from utils import load_artifact

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class MonitoringDashboard:
    """
    Creates visualizations and reports for model monitoring
    """
    
    def __init__(self, config, artifacts_dir: str):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.dashboards_dir = config['paths']['dashboards_dir']
        os.makedirs(self.dashboards_dir, exist_ok=True)
    
    def plot_feature_drift(self, drift_results: Dict, top_n: int = 10):
        """
        Plot feature drift scores (PSI and KL-Divergence)
        """
        features_data = []
        
        for feature, results in drift_results['features'].items():
            features_data.append({
                'feature': feature,
                'psi': results['psi'],
                'kl_divergence': results['kl_divergence'],
                'drift_detected': results['drift_detected']
            })
        
        df = pd.DataFrame(features_data)
        df = df.sort_values('psi', ascending=False).head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSI plot
        ax1 = axes[0]
        colors = ['red' if d else 'blue' for d in df['drift_detected']]
        ax1.barh(df['feature'], df['psi'], color=colors)
        ax1.axvline(x=self.config['data_drift']['psi_threshold'], 
                    color='red', linestyle='--', label=f"Threshold: {self.config['data_drift']['psi_threshold']}")
        ax1.set_xlabel('PSI Score')
        ax1.set_title('Population Stability Index (PSI) by Feature')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # KL-Divergence plot
        ax2 = axes[1]
        colors = ['red' if d else 'blue' for d in df['drift_detected']]
        ax2.barh(df['feature'], df['kl_divergence'], color=colors)
        ax2.axvline(x=self.config['data_drift']['kl_divergence_threshold'], 
                    color='red', linestyle='--', 
                    label=f"Threshold: {self.config['data_drift']['kl_divergence_threshold']}")
        ax2.set_xlabel('KL-Divergence')
        ax2.set_title('KL-Divergence by Feature')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.dashboards_dir, f'feature_drift_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature drift plot saved to {filepath}")
        return filepath
    
    def plot_performance_trends(self, performance_history: List[Dict]):
        """
        Plot model performance metrics over time
        """
        if not performance_history:
            logger.warning("No performance history available for plotting")
            return None
        
        df = pd.DataFrame(performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            logger.warning("No performance metrics available for plotting")
            return None
        
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(14, 4 * len(available_metrics)))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        baseline_metrics = load_artifact(os.path.join(self.artifacts_dir, "baseline_metrics.json"))
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            ax.plot(df['timestamp'], df[metric], marker='o', linewidth=2, label=f'Current {metric}')
            
            # Add baseline line
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                ax.axhline(y=baseline_value, color='green', linestyle='--', 
                          linewidth=2, label=f'Baseline {metric}: {baseline_value:.4f}')
            
            # Add threshold line if applicable
            if metric == 'auc' and self.config['performance']['min_auc']:
                ax.axhline(y=self.config['performance']['min_auc'], 
                          color='red', linestyle='--', 
                          label=f'Min AUC Threshold: {self.config["performance"]["min_auc"]}')
            elif metric == 'accuracy' and self.config['performance']['min_accuracy']:
                ax.axhline(y=self.config['performance']['min_accuracy'], 
                          color='red', linestyle='--', 
                          label=f'Min Accuracy Threshold: {self.config["performance"]["min_accuracy"]}')
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Over Time')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.dashboards_dir, f'performance_trends_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance trends plot saved to {filepath}")
        return filepath
    
    def plot_confusion_matrix_heatmap(self, confusion_matrix_data: List[List[int]], 
                                     title: str = "Confusion Matrix"):
        """Plot confusion matrix as heatmap"""
        cm = np.array(confusion_matrix_data)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.dashboards_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_executive_summary(self, monitoring_results: Dict) -> str:
        """
        Create executive-friendly summary report
        """
        summary = []
        summary.append("=" * 80)
        summary.append("ML MODEL MONITORING EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Model Health Status
        summary.append("MODEL HEALTH STATUS")
        summary.append("-" * 80)
        
        # Performance Summary
        if monitoring_results.get('performance'):
            perf = monitoring_results['performance']
            summary.append(f"Current Performance Metrics:")
            summary.append(f"  - Accuracy: {perf.get('accuracy', 'N/A'):.4f}")
            if 'auc' in perf:
                summary.append(f"  - AUC: {perf.get('auc', 'N/A'):.4f}")
            summary.append(f"  - Precision: {perf.get('precision', 'N/A'):.4f}")
            summary.append(f"  - Recall: {perf.get('recall', 'N/A'):.4f}")
            summary.append("")
        
        # Drift Summary
        summary.append("DRIFT DETECTION SUMMARY")
        summary.append("-" * 80)
        
        if monitoring_results.get('data_drift'):
            data_drift = monitoring_results['data_drift']
            summary.append(f"Data Drift Detected: {'YES' if data_drift.get('overall_drift_detected') else 'NO'}")
            if data_drift.get('overall_drift_detected'):
                summary.append(f"  - Features with Drift: {data_drift['drift_summary']['features_with_drift']}")
                summary.append(f"  - High Drift Features: {len(data_drift['drift_summary']['high_drift_features'])}")
            summary.append("")
        
        if monitoring_results.get('concept_drift'):
            concept_drift = monitoring_results['concept_drift']
            summary.append(f"Concept Drift Detected: {'YES' if concept_drift.get('concept_drift_detected') else 'NO'}")
            if concept_drift.get('concept_drift_detected'):
                summary.append(f"  - Drift Severity: {concept_drift.get('drift_severity', 'N/A')}")
            summary.append("")
        
        # Alerts Summary
        alerts = monitoring_results.get('alerts', [])
        summary.append("ALERTS SUMMARY")
        summary.append("-" * 80)
        summary.append(f"Total Alerts: {len(alerts)}")
        
        if alerts:
            high_severity = [a for a in alerts if a.get('severity') == 'HIGH']
            medium_severity = [a for a in alerts if a.get('severity') == 'MEDIUM']
            summary.append(f"  - High Severity: {len(high_severity)}")
            summary.append(f"  - Medium Severity: {len(medium_severity)}")
            
            if high_severity:
                summary.append("")
                summary.append("High Priority Alerts:")
                for alert in high_severity[:5]:  # Show top 5
                    summary.append(f"  - {alert.get('message', 'N/A')}")
        summary.append("")
        
        # Retraining Recommendation
        if monitoring_results.get('retraining_decision'):
            retrain_decision = monitoring_results['retraining_decision']
            summary.append("RETRAINING RECOMMENDATION")
            summary.append("-" * 80)
            summary.append(f"Should Retrain: {'YES' if retrain_decision.get('should_retrain') else 'NO'}")
            summary.append(f"Priority: {retrain_decision.get('priority', 'N/A')}")
            summary.append(f"Reason: {retrain_decision.get('reason', 'N/A')}")
            summary.append("")
        
        summary.append("=" * 80)
        
        report_text = "\n".join(summary)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.dashboards_dir, f'executive_summary_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Executive summary saved to {report_path}")
        return report_text
    
    def create_detailed_report(self, monitoring_results: Dict) -> str:
        """Create detailed technical report"""
        report = []
        report.append("=" * 80)
        report.append("ML MODEL MONITORING DETAILED REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {monitoring_results.get('timestamp', 'N/A')}")
        report.append(f"Number of Samples: {monitoring_results.get('n_samples', 'N/A')}")
        report.append("")
        
        # Data Drift Details
        if monitoring_results.get('data_drift'):
            from drift_detection import DataDriftDetector
            detector = DataDriftDetector(
                os.path.join(self.artifacts_dir, "baseline_distributions.json"),
                self.config
            )
            drift_summary = detector.get_drift_summary(monitoring_results['data_drift'])
            report.append(drift_summary)
            report.append("")
        
        # Concept Drift Details
        if monitoring_results.get('concept_drift'):
            from drift_detection import ConceptDriftDetector
            detector = ConceptDriftDetector(
                os.path.join(self.artifacts_dir, "baseline_metrics.json"),
                self.config
            )
            concept_summary = detector.get_concept_drift_summary(monitoring_results['concept_drift'])
            report.append(concept_summary)
            report.append("")
        
        # Performance Details
        if monitoring_results.get('performance'):
            perf = monitoring_results['performance']
            report.append("PERFORMANCE METRICS")
            report.append("-" * 80)
            for key, value in perf.items():
                if key not in ['confusion_matrix', 'vs_baseline']:
                    report.append(f"{key}: {value}")
            report.append("")
        
        # Alerts Details
        if monitoring_results.get('alerts'):
            report.append("ALERTS DETAILS")
            report.append("-" * 80)
            for alert in monitoring_results['alerts']:
                report.append(f"Type: {alert.get('type')}")
                report.append(f"Severity: {alert.get('severity')}")
                report.append(f"Message: {alert.get('message')}")
                report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.dashboards_dir, f'detailed_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Detailed report saved to {report_path}")
        return report_text

