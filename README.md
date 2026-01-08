# ML Model Monitoring & Data Drift Detection Platform

**Author**: M B GIRISH  
**Date**: January 2026

A production-ready platform for monitoring machine learning models in production. 
It detects data drift and concept drift, tracks performance degradation over time, 
and triggers alerts when retraining is needed.

## 📋 Table of Contents

- [Overview](#overview)
- [Business Context](#business-context)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Components](#components)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Understanding Drift](#understanding-drift)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This platform tries to answer the question: **"What happens after the model is deployed?"**

In my experience, models work great initially but then start degrading. This system 
helps catch those issues early.

Machine learning models degrade over time due to:
- **Data Drift**: Changes in feature distributions (the input data changes)
- **Concept Drift**: Changes in the relationship between features and target (P(Y|X) changes)
- **Performance Decay**: Gradual degradation in model accuracy

This platform continuously monitors these aspects and provides actionable insights for ML engineers and business teams.

## 💼 Business Context

**Use Case**: Customer Churn Prediction

A machine learning model has been deployed to predict customer churn for a telecommunications company. Over time:
- Customer demographics may shift
- Service offerings may change
- Economic conditions may affect behavior
- The model's predictive power may degrade

The system monitors:
1. **Data Quality**: Are incoming features similar to training data?
2. **Model Performance**: Is the model still accurate?
3. **Drift Detection**: Has the data or concept changed?
4. **Alerting**: When should we retrain?

## ✨ Features

### Core Capabilities

1. **Baseline Model Setup**
   - Train initial ML model (Random Forest Classifier)
   - Capture baseline metrics (AUC, Accuracy, Precision, Recall, F1)
   - Save feature distributions for drift comparison

2. **Data Drift Detection**
   - **Population Stability Index (PSI)**: Measures distribution shifts
   - **KL-Divergence**: Quantifies distribution differences
   - **Statistical Tests**: Kolmogorov-Smirnov, Mann-Whitney U
   - Feature-level drift detection and severity classification

3. **Concept Drift Detection**
   - Performance decay analysis
   - Error distribution changes
   - Rolling window evaluation
   - Distinguishes between data drift and concept drift

4. **Performance Monitoring**
   - Real-time metric tracking (AUC, Accuracy, Precision, Recall)
   - Gradual degradation detection
   - Silent failure identification
   - Threshold violation alerts

5. **Alerting & Retraining Logic**
   - Configurable thresholds for drift and performance
   - Multi-level alerting (HIGH, MEDIUM, LOW)
   - Automatic retraining decision logic
   - Model comparison and deployment recommendations

6. **Dashboards & Reporting**
   - Static visualizations (matplotlib/seaborn)
   - Interactive Streamlit dashboard
   - Executive summaries
   - Detailed technical reports

## 📁 Project Structure

```
ML Model Monitoring & Drift Detection/
├── config/
│   └── config.yaml              # Configuration file with thresholds
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Utility functions
│   ├── model_training.py        # Baseline model training
│   ├── drift_detection.py      # Data & concept drift detection
│   ├── performance_monitoring.py  # Performance tracking
│   ├── alerts.py                # Alerting & retraining logic
│   ├── monitoring_pipeline.py   # Main orchestration pipeline
│   ├── dashboard.py             # Static visualization & reporting
│   └── streamlit_dashboard.py   # Interactive web dashboard
├── models/                      # Saved models (generated)
├── artifacts/                   # Baseline metrics, distributions (generated)
├── dashboards/                  # Generated plots and reports (generated)
├── logs/                        # Log files (generated)
├── main.py                      # Main execution script
├── run_dashboard.py             # Streamlit dashboard launcher
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python3 -c "import pandas, numpy, sklearn, scipy, matplotlib, seaborn, yaml, streamlit, plotly; print('✓ All dependencies installed!')"
```

## 🏃 Quick Start

### Step 1: Run the Monitoring Pipeline

```bash
python3 main.py
```

This will:
1. Train a baseline model (if not already trained)
2. Initialize the monitoring pipeline
3. Simulate production monitoring with time-based data splits
4. Detect drift and performance issues
5. Generate dashboards and reports

### Step 2: Launch Interactive Dashboard

```bash
# Option 1: Direct Streamlit command
streamlit run src/streamlit_dashboard.py

# Option 2: Using the launcher script
python3 run_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### Expected Output

```
================================================================================
ML MODEL MONITORING & DRIFT DETECTION PLATFORM
================================================================================

================================================================================
STEP 1: TRAINING BASELINE MODEL
================================================================================
✓ Baseline model trained and saved

================================================================================
STEP 2: INITIALIZING MONITORING PIPELINE
================================================================================
✓ Monitoring pipeline initialized

================================================================================
STEP 3: SIMULATING PRODUCTION MONITORING
================================================================================
--- Monitoring Batch 1/5 ---
  Data Drift: DETECTED
  Concept Drift: NONE
  Performance - AUC: 0.8975, Accuracy: 0.8232
  Alerts: 20

================================================================================
STEP 4: GENERATING DASHBOARDS AND REPORTS
================================================================================
✓ Feature drift plots generated
✓ Performance trends plots generated
✓ Reports generated
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize thresholds and behavior:

```yaml
# Data Drift Thresholds
data_drift:
  psi_threshold: 0.2          # PSI > 0.2 indicates significant drift
  kl_divergence_threshold: 0.5
  feature_drift_threshold: 0.15
  window_size: 1000

# Concept Drift Thresholds
concept_drift:
  performance_decay_threshold: 0.05  # 5% drop in performance
  auc_drop_threshold: 0.03
  error_rate_increase: 0.1
  rolling_window_size: 500

# Performance Thresholds
performance:
  min_auc: 0.70
  min_accuracy: 0.75
  alert_on_degradation: true
  evaluation_window: 100

# Alerting Configuration
alerts:
  enabled: true
  drift_alert_threshold: 0.2
  performance_alert_threshold: 0.05
  log_level: "INFO"

# Retraining Configuration
retraining:
  auto_retrain: false
  min_samples_for_retrain: 1000
  retrain_on_drift: true
  retrain_on_performance_drop: true
  model_comparison_metric: "auc"
```

### Threshold Guidelines

**PSI Thresholds**:
- `< 0.1`: No significant change
- `0.1 - 0.2`: Moderate change (monitor)
- `≥ 0.2`: Significant drift (action required)

**Performance Thresholds**:
- Based on business requirements
- Consider cost of false positives/negatives
- Account for model uncertainty

## 🔧 Components

### 1. Baseline Model Training (`src/model_training.py`)

Trains and saves the initial production model with baseline metrics and feature distributions.

**Usage**:
```python
from src.model_training import BaselineModelTrainer
from src.utils import load_config

config = load_config()
trainer = BaselineModelTrainer(config)
trainer.train("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
trainer.save_baseline("models/", "artifacts/")
```

### 2. Data Drift Detection (`src/drift_detection.py`)

Detects changes in feature distributions using PSI, KL-Divergence, and statistical tests.

**Usage**:
```python
from src.drift_detection import DataDriftDetector

detector = DataDriftDetector("artifacts/baseline_distributions.json", config)
drift_results = detector.detect_drift(current_data)
```

### 3. Concept Drift Detection (`src/drift_detection.py`)

Detects changes in feature-target relationships by analyzing performance degradation.

**Usage**:
```python
from src.drift_detection import ConceptDriftDetector

detector = ConceptDriftDetector("artifacts/baseline_metrics.json", config)
concept_drift = detector.detect_concept_drift(current_metrics, predictions, actual)
```

### 4. Performance Monitoring (`src/performance_monitoring.py`)

Tracks model performance over time and identifies degradation patterns.

**Usage**:
```python
from src.performance_monitoring import PerformanceMonitor

monitor = PerformanceMonitor("artifacts/baseline_metrics.json", config)
metrics = monitor.evaluate_performance(y_true, y_pred, y_pred_proba)
violations = monitor.check_performance_thresholds(metrics)
```

### 5. Alerting System (`src/alerts.py`)

Triggers alerts based on drift and performance issues with severity classification.

**Alert Types**:
- `DATA_DRIFT`: High PSI in features
- `CONCEPT_DRIFT`: Performance degradation
- `PERFORMANCE_THRESHOLD_VIOLATION`: Metrics below thresholds
- `ERROR_RATE_INCREASE`: Significant error rate increase

**Usage**:
```python
from src.alerts import AlertManager

alert_manager = AlertManager(config)
alerts = alert_manager.check_data_drift_alerts(drift_results)
```

### 6. Retraining Logic (`src/alerts.py`)

Decides when to retrain the model and compares new models with baseline.

**Usage**:
```python
from src.alerts import RetrainingManager

retrain_manager = RetrainingManager(config)
decision = retrain_manager.should_retrain(
    drift_alerts, concept_drift_detected, performance_degraded, n_samples
)
```

### 7. Monitoring Pipeline (`src/monitoring_pipeline.py`)

Orchestrates all monitoring components for end-to-end workflow.

**Usage**:
```python
from src.monitoring_pipeline import MonitoringPipeline

pipeline = MonitoringPipeline()
pipeline.initialize()
results = pipeline.monitor_batch(data, ground_truth)
```

### 8. Static Dashboard (`src/dashboard.py`)

Generates static visualizations and reports.

**Usage**:
```python
from src.dashboard import MonitoringDashboard

dashboard = MonitoringDashboard(config, "artifacts/")
dashboard.plot_feature_drift(drift_results)
dashboard.plot_performance_trends(performance_history)
dashboard.create_executive_summary(results)
```

## 📊 Streamlit Dashboard

### Launching the Dashboard

```bash
streamlit run src/streamlit_dashboard.py
```

### Dashboard Features

#### Overview Tab
- **Model Health Metrics**: Current AUC, Accuracy, Precision, Recall, F1-Score
- **Baseline Comparison**: Side-by-side comparison with baseline metrics
- **Drift Status**: Quick view of data drift and concept drift detection
- **Confusion Matrix**: Latest model predictions visualization

#### Performance Tab
- **Interactive Performance Charts**: 
  - AUC, Accuracy, Precision, Recall trends over time
  - Baseline reference lines
  - Hover tooltips for detailed values
- **Performance History Table**: Complete performance metrics history

#### Drift Detection Tab
- **Feature Drift Visualization**: 
  - PSI scores for all features
  - Color-coded by drift detection status
  - Threshold indicators
- **Top Drifted Features**: Table showing features with highest drift
- **Concept Drift Analysis**: 
  - Performance degradation indicators
  - Error rate analysis
  - Severity classification

#### Alerts Tab
- **Alert Summary**: 
  - High, Medium, Low severity breakdown
  - Color-coded alert cards
- **Alert Statistics**: 
  - Pie chart: Alerts by type
  - Bar chart: Alerts by severity
- **Complete Alert History**: Expandable table with all alerts

#### Reports Tab
- **Retraining Recommendations**: 
  - Should retrain decision
  - Priority level
  - Conditions met
- **Executive Summary**: 
  - Model health overview
  - Key metrics
  - Downloadable report

### Refreshing Data

The dashboard caches data for performance. To refresh:
1. Click the **"Reload Data"** button in the sidebar
2. Or restart the Streamlit app (Ctrl+C and run again)

### Accessing Remotely

To access the dashboard from another machine:

```bash
streamlit run src/streamlit_dashboard.py --server.address 0.0.0.0
```

Then access via: `http://<your-ip>:8501`

**Note**: Only do this on trusted networks for security.

## 📊 Understanding Drift

### Data Drift vs Concept Drift

**Data Drift (Covariate Shift)**:
- **Definition**: Changes in the distribution of input features P(X)
- **Example**: Customer age distribution shifts, monthly charges increase
- **Detection**: PSI, KL-Divergence, statistical tests
- **Impact**: Model may still work if P(Y|X) is stable

**Concept Drift**:
- **Definition**: Changes in the relationship between features and target P(Y|X)
- **Example**: Same features, but churn behavior changes (e.g., economic downturn)
- **Detection**: Performance degradation, error rate increases
- **Impact**: Model predictions become less accurate

### When to Retrain?

The platform recommends retraining when:
1. **High Data Drift**: PSI > 0.2 in multiple features
2. **Concept Drift**: AUC drops by > 5%
3. **Performance Degradation**: Metrics below thresholds
4. **Sufficient Data**: At least 1000 new samples available

## 📈 Outputs

### Generated Files

1. **Models** (`models/`):
   - `baseline_model.pkl`: Trained model
   - `preprocessors.pkl`: Feature encoders and scaler

2. **Artifacts** (`artifacts/`):
   - `baseline_metrics.json`: Baseline performance metrics
   - `baseline_distributions.json`: Feature distributions
   - `monitoring_results_*.json`: Per-batch monitoring results
   - `performance_history.json`: Performance over time
   - `alerts_history.json`: All generated alerts
   - `retraining_history.json`: Retraining decisions

3. **Dashboards** (`dashboards/`):
   - `feature_drift_*.png`: Feature drift visualizations
   - `performance_trends_*.png`: Performance over time charts
   - `executive_summary_*.txt`: Business-friendly summary
   - `detailed_report_*.txt`: Technical detailed report

4. **Logs** (`logs/`):
   - `monitoring_YYYYMMDD.log`: Daily log files

## 🚀 Production Deployment

### Production Considerations

1. **Data Storage**: 
   - Replace JSON files with database (PostgreSQL, MongoDB)
   - Use time-series databases for performance metrics
   - Implement data retention policies

2. **Scheduled Monitoring**:
   - Set up cron jobs or scheduled tasks
   - Monitor batches at regular intervals (hourly, daily)
   - Automate retraining pipeline

3. **Alerting Integration**:
   - Connect to Slack, email, or PagerDuty
   - Implement escalation policies
   - Create alert routing rules

4. **Authentication & Security**:
   - Add authentication to Streamlit dashboard
   - Use HTTPS in production
   - Implement role-based access control

5. **Model Versioning**:
   - Use MLflow or similar for model versioning
   - Track model lineage
   - Implement A/B testing for new models

6. **Scalability**:
   - Use distributed computing for large datasets
   - Implement sampling for drift detection
   - Cache baseline distributions

### Docker Deployment

Example `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "src/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ml-monitoring .
docker run -p 8501:8501 ml-monitoring
```

### Cloud Deployment

**Streamlit Cloud**:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

**AWS/GCP/Azure**:
1. Containerize application
2. Deploy to container service (ECS, Cloud Run, Container Instances)
3. Set up load balancer and auto-scaling

## 🐛 Troubleshooting

### Dashboard shows "No data available"

**Solution**: Run `python3 main.py` first to generate monitoring data

### Import errors

**Solution**: 
- Ensure virtual environment is activated
- Check you're in the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Port already in use

**Solution**: 
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
streamlit run src/streamlit_dashboard.py --server.port 8502
```

### Charts not displaying

**Solution**: 
- Check browser console for JavaScript errors
- Ensure Plotly is installed: `pip install plotly`
- Clear browser cache
- Check browser compatibility

### Performance issues

**Solution**:
- Use data sampling for large datasets
- Enable caching in Streamlit
- Consider using a database instead of JSON files
- Optimize drift detection calculations

## 🛠️ Customization

### Adding New Metrics

Edit `src/performance_monitoring.py`:
```python
def evaluate_performance(self, y_true, y_pred, y_pred_proba):
    metrics = {
        # ... existing metrics
        'custom_metric': calculate_custom_metric(y_true, y_pred)
    }
    return metrics
```

### Adding New Drift Detection Methods

Edit `src/drift_detection.py`:
```python
def calculate_custom_drift_metric(self, expected, actual):
    # Your custom drift calculation
    return drift_score
```

### Custom Alerting

Edit `src/alerts.py`:
```python
def check_custom_alerts(self, monitoring_results):
    # Your custom alert logic
    return alerts
```

## 📝 Best Practices

1. **Threshold Selection**:
   - Set thresholds based on business impact
   - Review alerts regularly to reduce false positives
   - Document threshold decisions

2. **Monitoring Schedule**:
   - Monitor at regular intervals (daily recommended)
   - Adjust frequency based on data volume
   - Set up automated monitoring

3. **Retraining Strategy**:
   - Maintain a retraining schedule (e.g., monthly)
   - Document all retraining decisions
   - Compare models before deployment

4. **Data Management**:
   - Implement data retention policies
   - Archive old monitoring results
   - Keep baseline distributions for comparison

5. **Documentation**:
   - Document model changes
   - Track drift events and responses
   - Maintain runbooks for common issues

## 📄 License

This project is provided as-is. Feel free to use it for your own projects.

## 🙏 Acknowledgments

- Dataset: Telco Customer Churn (publicly available dataset)
- Methods: PSI and KL-Divergence are standard drift detection techniques used in the industry
- Thanks to the ML community for sharing best practices on model monitoring

---

**Author**: M B GIRISH | **Date**: January 2026  
**Built for Production ML Teams** | *Answering "What happens after deployment?"*
