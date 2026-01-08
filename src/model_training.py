"""
Baseline Model Training Module
Trains an initial ML model and captures baseline metrics and feature distributions

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix
)
import pickle
import os
from pathlib import Path
import logging

from utils import save_model, save_artifact, load_config, create_directories

logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    """Train and save baseline model with feature distributions
    
    This class handles the initial model training and captures everything
    we need for later drift detection.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.label_encoders = {}  # Store encoders for categorical features
        self.scaler = StandardScaler()
        self.feature_names = []
        self.baseline_metrics = {}
        self.baseline_distributions = {}  # Need this for drift comparison later
        
    def load_data(self, data_path):
        """Load and preprocess data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Handle missing values - TotalCharges has some empty strings
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)  # Fill with 0 for new customers
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Target distribution:\n{df[self.config['model']['target_column']].value_counts()}")
        
        return df
    
    def preprocess_features(self, df, is_training=True):
        """Preprocess features for model training
        
        Handles encoding of categorical variables and scaling. 
        The is_training flag determines if we fit or just transform.
        """
        df_processed = df.copy()
        
        # Encode target variable (Yes/No -> 1/0)
        target_col = self.config['model']['target_column']
        if is_training:
            le_target = LabelEncoder()
            df_processed[target_col] = le_target.fit_transform(df_processed[target_col])
            self.label_encoders['target'] = le_target
        else:
            # For inference, use the saved encoder
            if 'target' in self.label_encoders:
                df_processed[target_col] = self.label_encoders['target'].transform(df_processed[target_col])
        
        # Select features
        exclude_cols = ['customerID', target_col]
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        
        # Encode categorical features
        categorical_cols = df_processed[feature_cols].select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df_processed[col] = df_processed[col].astype(str).map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Extract features
        X = df_processed[feature_cols].select_dtypes(include=[np.number])
        y = df_processed[target_col] if target_col in df_processed.columns else None
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_names), y
    
    def train(self, data_path):
        """Train baseline model"""
        logger.info("Starting baseline model training...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess
        X, y = self.preprocess_features(df, is_training=True)
        
        # Split data
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model - using Random Forest, works well for this use case
        logger.info("Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,  # 100 trees should be enough
            max_depth=10,  # Prevent overfitting
            random_state=random_state,
            n_jobs=-1  # Use all cores
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.baseline_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info("Baseline Model Metrics:")
        for metric, value in self.baseline_metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")
        
        # Capture baseline feature distributions
        self.baseline_distributions = self._capture_feature_distributions(X_train)
        
        logger.info("Baseline model training completed successfully!")
        
        return self.model, self.baseline_metrics, self.baseline_distributions
    
    def _capture_feature_distributions(self, X):
        """Capture baseline feature distributions for drift detection"""
        distributions = {}
        
        for col in X.columns:
            values = X[col].values
            distributions[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentiles': {
                    'p25': float(np.percentile(values, 25)),
                    'p50': float(np.percentile(values, 50)),
                    'p75': float(np.percentile(values, 75)),
                    'p90': float(np.percentile(values, 90)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99))
                },
                'histogram': np.histogram(values, bins=50)[0].tolist(),
                'histogram_edges': np.histogram(values, bins=50)[1].tolist()
            }
        
        return distributions
    
    def save_baseline(self, models_dir, artifacts_dir):
        """Save model and baseline artifacts"""
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, "baseline_model.pkl")
        save_model(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessors
        preprocessor_path = os.path.join(models_dir, "preprocessors.pkl")
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        save_model(preprocessors, preprocessor_path)
        logger.info(f"Preprocessors saved to {preprocessor_path}")
        
        # Save baseline metrics
        metrics_path = os.path.join(artifacts_dir, "baseline_metrics.json")
        save_artifact(self.baseline_metrics, metrics_path)
        logger.info(f"Baseline metrics saved to {metrics_path}")
        
        # Save baseline distributions
        distributions_path = os.path.join(artifacts_dir, "baseline_distributions.json")
        save_artifact(self.baseline_distributions, distributions_path)
        logger.info(f"Baseline distributions saved to {distributions_path}")
        
        return {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'metrics_path': metrics_path,
            'distributions_path': distributions_path
        }

def main():
    """Main training script"""
    from utils import setup_logging, load_config, create_directories
    
    logger = setup_logging()
    config = load_config()
    create_directories(config)
    
    data_path = os.path.join(config['paths']['data_dir'], "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    trainer = BaselineModelTrainer(config)
    trainer.train(data_path)
    trainer.save_baseline(
        config['paths']['models_dir'],
        config['paths']['artifacts_dir']
    )
    
    logger.info("Baseline model setup complete!")

if __name__ == "__main__":
    main()

