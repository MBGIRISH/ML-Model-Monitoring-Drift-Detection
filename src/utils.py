"""
Utility functions for ML Model Monitoring & Drift Detection Platform

Author: M B GIRISH
Date: January 2026
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
import json
from datetime import datetime
from pathlib import Path
import logging

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"monitoring_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_model(model, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_artifact(data, artifact_path):
    """Save artifacts (baseline distributions, metrics, etc.)"""
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    if artifact_path.endswith('.json'):
        with open(artifact_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif artifact_path.endswith('.pkl'):
        with open(artifact_path, 'wb') as f:
            pickle.dump(data, f)

def load_artifact(artifact_path):
    """Load artifacts"""
    if artifact_path.endswith('.json'):
        with open(artifact_path, 'r') as f:
            return json.load(f)
    elif artifact_path.endswith('.pkl'):
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)

def create_directories(config):
    """Create necessary directories"""
    paths = config['paths']
    for path_key, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)

