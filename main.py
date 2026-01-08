"""
Main Execution Script
Runs the complete ML Model Monitoring & Drift Detection pipeline

Author: M B GIRISH
Date: January 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import setup_logging, load_config, create_directories
from model_training import BaselineModelTrainer
from monitoring_pipeline import MonitoringPipeline
from dashboard import MonitoringDashboard

def simulate_time_based_data_split(df, n_splits=5):
    """
    Simulate time-based data splits for monitoring.
    
    In a real production setup, this would be actual streaming data or
    batches coming in over time. For demo purposes, we're just splitting
    the existing data.
    """
    # Sort by tenure - longer tenure could represent "older" data
    df_sorted = df.sort_values('tenure')
    
    # Split into batches
    split_size = len(df_sorted) // n_splits
    splits = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else len(df_sorted)
        split = df_sorted.iloc[start_idx:end_idx].copy()
        splits.append(split)
    
    return splits

def introduce_drift(df, drift_type='moderate'):
    """
    Introduce artificial drift to simulate real-world scenarios.
    
    In reality, drift happens naturally - prices change, customer behavior
    shifts, etc. This function just simulates that for testing.
    """
    df_drifted = df.copy()
    
    if drift_type == 'moderate':
        # Simulate price increases - happens all the time
        df_drifted['MonthlyCharges'] = df_drifted['MonthlyCharges'] * np.random.uniform(1.1, 1.3, len(df_drifted))
        
        # Some customers switch to month-to-month (more flexible)
        mask = np.random.random(len(df_drifted)) < 0.2
        df_drifted.loc[mask, 'Contract'] = 'Month-to-month'
    
    elif drift_type == 'severe':
        # Big price jump and shorter tenures (maybe economic downturn?)
        df_drifted['MonthlyCharges'] = df_drifted['MonthlyCharges'] * np.random.uniform(1.3, 1.5, len(df_drifted))
        df_drifted['tenure'] = df_drifted['tenure'] * np.random.uniform(0.7, 0.9, len(df_drifted))
        
        # More customers on month-to-month and using electronic payments
        mask = np.random.random(len(df_drifted)) < 0.4
        df_drifted.loc[mask, 'Contract'] = 'Month-to-month'
        df_drifted.loc[mask, 'PaymentMethod'] = 'Electronic check'
    
    return df_drifted

def main():
    """Main execution function"""
    print("=" * 80)
    print("ML MODEL MONITORING & DRIFT DETECTION PLATFORM")
    print("=" * 80)
    print()
    
    # Setup
    logger = setup_logging()
    config = load_config()
    create_directories(config)
    
    data_path = os.path.join(config['paths']['data_dir'], "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Step 1: Train Baseline Model
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING BASELINE MODEL")
    print("=" * 80)
    
    if not os.path.exists(os.path.join(config['paths']['models_dir'], "baseline_model.pkl")):
        trainer = BaselineModelTrainer(config)
        trainer.train(data_path)
        trainer.save_baseline(
            config['paths']['models_dir'],
            config['paths']['artifacts_dir']
        )
        print("✓ Baseline model trained and saved")
    else:
        print("✓ Baseline model already exists, skipping training")
    
    # Step 2: Initialize Monitoring Pipeline
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING MONITORING PIPELINE")
    print("=" * 80)
    
    pipeline = MonitoringPipeline()
    pipeline.initialize()
    print("✓ Monitoring pipeline initialized")
    
    # Step 3: Load Data and Simulate Monitoring
    print("\n" + "=" * 80)
    print("STEP 3: SIMULATING PRODUCTION MONITORING")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Split data into time-based batches
    data_splits = simulate_time_based_data_split(df, n_splits=5)
    
    # Initialize dashboard
    dashboard = MonitoringDashboard(config, config['paths']['artifacts_dir'])
    
    all_results = []
    
    # Monitor each batch
    for i, batch in enumerate(data_splits):
        print(f"\n--- Monitoring Batch {i+1}/{len(data_splits)} ---")
        print(f"Batch size: {len(batch)}")
        
        # Introduce drift in later batches to simulate real-world scenario
        if i >= 2:
            drift_type = 'moderate' if i == 2 else 'severe'
            batch = introduce_drift(batch, drift_type=drift_type)
            print(f"  → Introduced {drift_type} drift")
        
        # Extract features and target
        feature_cols = [col for col in batch.columns 
                       if col not in ['customerID', config['model']['target_column']]]
        X_batch = batch[feature_cols + ['customerID']]
        y_batch = batch[config['model']['target_column']]
        
        # Monitor batch
        results = pipeline.monitor_batch(X_batch, ground_truth=y_batch)
        all_results.append(results)
        
        # Print summary
        if results.get('data_drift'):
            print(f"  Data Drift: {'DETECTED' if results['data_drift']['overall_drift_detected'] else 'NONE'}")
        if results.get('concept_drift'):
            print(f"  Concept Drift: {'DETECTED' if results['concept_drift']['concept_drift_detected'] else 'NONE'}")
        if results.get('performance'):
            print(f"  Performance - AUC: {results['performance'].get('auc', 'N/A'):.4f}, "
                  f"Accuracy: {results['performance'].get('accuracy', 'N/A'):.4f}")
        print(f"  Alerts: {len(results.get('alerts', []))}")
    
    # Step 4: Generate Dashboards and Reports
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING DASHBOARDS AND REPORTS")
    print("=" * 80)
    
    # Plot feature drift for latest batch
    if all_results[-1].get('data_drift'):
        dashboard.plot_feature_drift(all_results[-1]['data_drift'])
        print("✓ Feature drift plots generated")
    
    # Plot performance trends
    performance_history = []
    for result in all_results:
        if result.get('performance'):
            performance_history.append(result['performance'])
    
    if performance_history:
        dashboard.plot_performance_trends(performance_history)
        print("✓ Performance trends plots generated")
    
    # Generate reports
    latest_results = all_results[-1]
    dashboard.create_executive_summary(latest_results)
    dashboard.create_detailed_report(latest_results)
    print("✓ Reports generated")
    
    # Step 5: Summary
    print("\n" + "=" * 80)
    print("MONITORING SUMMARY")
    print("=" * 80)
    
    total_alerts = sum(len(r.get('alerts', [])) for r in all_results)
    drift_detected_count = sum(1 for r in all_results if r.get('data_drift', {}).get('overall_drift_detected', False))
    concept_drift_count = sum(1 for r in all_results if r.get('concept_drift', {}).get('concept_drift_detected', False))
    
    print(f"Total Batches Monitored: {len(all_results)}")
    print(f"Total Alerts Generated: {total_alerts}")
    print(f"Batches with Data Drift: {drift_detected_count}")
    print(f"Batches with Concept Drift: {concept_drift_count}")
    print(f"\nAll artifacts saved to: {config['paths']['artifacts_dir']}")
    print(f"Dashboards saved to: {config['paths']['dashboards_dir']}")
    print(f"Models saved to: {config['paths']['models_dir']}")
    
    print("\n" + "=" * 80)
    print("MONITORING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

