"""
Main Monitoring Pipeline
Orchestrates all monitoring components for production use

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Optional

from utils import (
    setup_logging, load_config, create_directories, 
    load_model, load_artifact, save_artifact
)
from model_training import BaselineModelTrainer
from drift_detection import DataDriftDetector, ConceptDriftDetector
from performance_monitoring import PerformanceMonitor
from alerts import AlertManager, RetrainingManager

logger = logging.getLogger(__name__)

class MonitoringPipeline:
    """
    Main pipeline for ML model monitoring and drift detection.
    
    This ties everything together - loads the model, runs drift detection,
    monitors performance, and generates alerts.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        create_directories(self.config)
        
        # Initialize components
        self.data_drift_detector = None
        self.concept_drift_detector = None
        self.performance_monitor = None
        self.alert_manager = AlertManager(self.config)
        self.retraining_manager = RetrainingManager(self.config)
        
        # Load model and preprocessors - these get loaded in initialize()
        self.model = None
        self.preprocessors = None
        
    def initialize(self):
        """Initialize monitoring components with baseline artifacts"""
        logger.info("Initializing monitoring pipeline...")
        
        artifacts_dir = self.config['paths']['artifacts_dir']
        models_dir = self.config['paths']['models_dir']
        
        # Load baseline distributions - need these to compare against
        baseline_distributions_path = os.path.join(artifacts_dir, "baseline_distributions.json")
        if os.path.exists(baseline_distributions_path):
            self.data_drift_detector = DataDriftDetector(baseline_distributions_path, self.config)
            logger.info("Data drift detector initialized")
        else:
            logger.warning("Baseline distributions not found. Run model training first.")
        
        # Load baseline metrics
        baseline_metrics_path = os.path.join(artifacts_dir, "baseline_metrics.json")
        if os.path.exists(baseline_metrics_path):
            self.concept_drift_detector = ConceptDriftDetector(baseline_metrics_path, self.config)
            self.performance_monitor = PerformanceMonitor(baseline_metrics_path, self.config)
            logger.info("Concept drift detector and performance monitor initialized")
        else:
            logger.warning("Baseline metrics not found. Run model training first.")
        
        # Load model
        model_path = os.path.join(models_dir, "baseline_model.pkl")
        preprocessor_path = os.path.join(models_dir, "preprocessors.pkl")
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            self.model = load_model(model_path)
            self.preprocessors = load_model(preprocessor_path)
            logger.info("Model and preprocessors loaded")
        else:
            logger.warning("Model not found. Run model training first.")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess incoming data using saved preprocessors"""
        if self.preprocessors is None:
            raise ValueError("Preprocessors not loaded. Initialize pipeline first.")
        
        df_processed = df.copy()
        label_encoders = self.preprocessors['label_encoders']
        scaler = self.preprocessors['scaler']
        feature_names = self.preprocessors['feature_names']
        
        # Encode target if present
        target_col = self.config['model']['target_column']
        if target_col in df_processed.columns:
            if 'target' in label_encoders:
                df_processed[target_col] = label_encoders['target'].transform(df_processed[target_col])
        
        # Encode categorical features
        exclude_cols = ['customerID', target_col]
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        categorical_cols = df_processed[feature_cols].select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in label_encoders:
                df_processed[col] = df_processed[col].astype(str).map(
                    lambda x: label_encoders[col].transform([x])[0] 
                    if x in label_encoders[col].classes_ 
                    else -1
                )
        
        # Extract and scale features
        X = df_processed[feature_cols].select_dtypes(include=[np.number])
        X_scaled = scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_names)
    
    def monitor_batch(self, data: pd.DataFrame, ground_truth: Optional[pd.Series] = None) -> Dict:
        """
        Monitor a batch of incoming data
        
        Args:
            data: Incoming data batch
            ground_truth: Optional ground truth labels for performance evaluation
        
        Returns:
            Dictionary with all monitoring results
        """
        logger.info(f"Monitoring batch of {len(data)} samples...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(data),
            'data_drift': None,
            'concept_drift': None,
            'performance': None,
            'alerts': [],
            'retraining_decision': None
        }
        
        # Preprocess data
        try:
            X_processed = self.preprocess_data(data)
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return results
        
        # 1. Data Drift Detection
        if self.data_drift_detector:
            try:
                drift_results = self.data_drift_detector.detect_drift(X_processed)
                results['data_drift'] = drift_results
                
                # Check for alerts
                drift_alerts = self.alert_manager.check_data_drift_alerts(drift_results)
                results['alerts'].extend(drift_alerts)
                
                logger.info(f"Data drift detection completed. Drift detected: {drift_results['overall_drift_detected']}")
            except Exception as e:
                logger.error(f"Error in data drift detection: {e}")
        
        # 2. Model Predictions
        if self.model:
            try:
                predictions = self.model.predict(X_processed)
                prediction_probas = self.model.predict_proba(X_processed)[:, 1]
                results['predictions'] = predictions.tolist()
                results['prediction_probas'] = prediction_probas.tolist()
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
        
        # 3. Performance Monitoring (if ground truth available)
        if ground_truth is not None and self.performance_monitor:
            try:
                # Encode ground truth if needed
                if self.preprocessors and 'target' in self.preprocessors['label_encoders']:
                    le_target = self.preprocessors['label_encoders']['target']
                    y_true = le_target.transform(ground_truth)
                else:
                    y_true = ground_truth.values
                
                # Evaluate performance
                metrics = self.performance_monitor.evaluate_performance(
                    y_true, predictions, prediction_probas
                )
                results['performance'] = metrics
                
                # Check performance thresholds
                violations = self.performance_monitor.check_performance_thresholds(metrics)
                performance_alerts = self.alert_manager.check_performance_alerts(violations)
                results['alerts'].extend(performance_alerts)
                
                # Detect gradual degradation
                degradation = self.performance_monitor.detect_gradual_degradation()
                results['degradation_analysis'] = degradation
                
                # Detect silent failures
                silent_failures = self.performance_monitor.detect_silent_failures(metrics)
                results['silent_failure_analysis'] = silent_failures
                
                # 4. Concept Drift Detection
                if self.concept_drift_detector:
                    concept_drift_results = self.concept_drift_detector.detect_concept_drift(
                        metrics, predictions, y_true
                    )
                    results['concept_drift'] = concept_drift_results
                    
                    # Check for alerts
                    concept_alerts = self.alert_manager.check_concept_drift_alerts(concept_drift_results)
                    results['alerts'].extend(concept_alerts)
                
                logger.info(f"Performance monitoring completed. AUC: {metrics.get('auc', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
        
        # 5. Retraining Decision
        if self.retraining_manager:
            drift_detected = results['data_drift'] and results['data_drift'].get('overall_drift_detected', False)
            concept_drift_detected = results['concept_drift'] and results['concept_drift'].get('concept_drift_detected', False)
            performance_degraded = results.get('performance') and any(
                v.get('degraded', False) for v in results.get('performance', {}).get('vs_baseline', {}).values()
            )
            
            retraining_decision = self.retraining_manager.should_retrain(
                drift_alerts=results['alerts'],
                concept_drift_detected=concept_drift_detected,
                performance_degraded=performance_degraded,
                available_samples=len(data)
            )
            results['retraining_decision'] = retraining_decision
        
        # Save results
        self._save_monitoring_results(results)
        
        return results
    
    def _save_monitoring_results(self, results: Dict):
        """Save monitoring results to disk"""
        artifacts_dir = self.config['paths']['artifacts_dir']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results
        results_path = os.path.join(artifacts_dir, f"monitoring_results_{timestamp}.json")
        save_artifact(results, results_path)
        
        # Update alert and retraining histories
        self.alert_manager.save_alerts(artifacts_dir)
        if results.get('retraining_decision'):
            self.retraining_manager.log_retraining_decision(results['retraining_decision'])
            self.retraining_manager.save_retraining_history(artifacts_dir)
        
        if results.get('performance'):
            self.performance_monitor.save_performance_history(artifacts_dir)

