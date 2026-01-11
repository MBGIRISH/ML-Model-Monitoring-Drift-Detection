# ML Model Monitoring & Data Drift Detection Platform

**Author**: M B GIRISH  
**Date**: January 2026

## Problem Statement

Machine learning models degrade in production environments due to evolving data distributions and changing relationships between features and targets. This degradation occurs silently, leading to poor predictions and business impact before teams realize the model has failed.

Traditional ML workflows focus on training and deployment but lack systematic monitoring mechanisms. Organizations need automated systems to:
- Detect when input data distributions shift (data drift)
- Identify when feature-target relationships change (concept drift)
- Track performance degradation over time
- Trigger alerts and retraining workflows when thresholds are exceeded

Success is measured by early detection of model degradation, reduced false positive alerts, and timely retraining decisions that maintain model performance above acceptable thresholds.

## Objective

Develop a production-ready monitoring system that continuously tracks deployed ML models, detects drift using statistical methods, monitors performance metrics, and provides actionable alerts for retraining decisions. The system should be configurable, scalable, and provide both programmatic APIs and interactive dashboards for different stakeholder needs.

Key constraints:
- Must work with classification models initially
- Should handle batch monitoring scenarios
- Configuration-driven thresholds for different business contexts
- Minimal latency for drift detection calculations

## Dataset

**Dataset**: Telco Customer Churn Dataset (IBM Watson Analytics)
- **Type**: Structured tabular data
- **Size**: 7,043 customer records, 21 features
- **Source**: Publicly available telecommunications customer dataset

**Key Features**:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Service information: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account details: tenure, Contract, PaperlessBilling, PaymentMethod
- Billing: MonthlyCharges, TotalCharges
- Target: Churn (binary classification)

**Data Preprocessing**:
- Converted TotalCharges from string to numeric (handled empty strings)
- Filled missing TotalCharges with 0 (new customers)
- Encoded categorical variables using LabelEncoder
- Standardized numerical features using StandardScaler
- Stratified train-test split (80-20) to maintain class distribution

## Approach

The solution implements a multi-layered monitoring architecture:

**Baseline Establishment**:
- Train initial production model (Random Forest Classifier)
- Capture baseline performance metrics (AUC, Accuracy, Precision, Recall, F1)
- Store feature distributions for statistical comparison

**Drift Detection Strategy**:
- **Data Drift**: Compare current feature distributions against baseline using Population Stability Index (PSI) and KL-Divergence
- **Concept Drift**: Monitor performance degradation and error distribution changes over rolling windows
- Statistical tests (Kolmogorov-Smirnov, Mann-Whitney U) for distribution comparisons

**Monitoring Pipeline**:
- Batch-based monitoring with configurable window sizes
- Real-time performance metric calculation
- Threshold-based alerting system with severity classification
- Automatic retraining recommendation logic

**Training Strategy**:
- Single train-test split for baseline establishment
- Time-based simulation for production monitoring scenarios
- Model comparison framework for retraining decisions

## Model & Techniques Used

**Machine Learning Model**:
- Random Forest Classifier (scikit-learn)
  - 100 estimators
  - Max depth 10 (prevent overfitting)
  - Handles mixed feature types effectively

**Statistical Techniques**:
- **Population Stability Index (PSI)**: Industry standard for detecting distribution shifts
- **KL-Divergence**: Information-theoretic measure of distribution differences
- **Kolmogorov-Smirnov Test**: Non-parametric test for distribution equality
- **Mann-Whitney U Test**: Non-parametric test for comparing distributions

**Libraries & Frameworks**:
- Python 3.8+
- scikit-learn: Model training and evaluation
- pandas, numpy: Data manipulation and numerical computations
- scipy: Statistical tests and calculations
- matplotlib, seaborn: Static visualizations
- streamlit, plotly: Interactive dashboard development
- pyyaml: Configuration management

## Evaluation Metrics

**Primary Metrics**:
- **AUC-ROC**: Measures model's ability to distinguish between classes (chosen for class imbalance)
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall

**Drift Detection Metrics**:
- **PSI Score**: Threshold of 0.2 indicates significant drift
- **KL-Divergence**: Threshold of 0.5 for distribution differences
- **Performance Decay**: 5% drop in AUC triggers concept drift alert

**Why These Metrics**:
- AUC-ROC is robust to class imbalance (churn rates typically 15-30%)
- PSI is widely adopted in financial services and ML operations
- Performance-based concept drift detection directly ties to business impact

**Validation Strategy**:
- Baseline metrics calculated on held-out test set
- Monitoring uses simulated production batches
- Thresholds validated through iterative tuning

## Results

**Baseline Model Performance**:
- AUC: 0.8349
- Accuracy: 0.7956
- Precision: 0.6453
- Recall: 0.5107
- F1-Score: 0.5701

**Monitoring Capabilities**:
- Successfully detects data drift in all 19 features with PSI scores ranging from 2.35 to 7.05
- Concept drift detection triggers when AUC drops below 0.79 (5% degradation threshold)
- Alert system generates severity-classified notifications (HIGH, MEDIUM, LOW)
- Retraining recommendations provided based on configurable conditions

**Key Insights**:
- Feature-level drift detection allows identification of specific problematic features
- Performance metrics can remain stable despite data drift (data drift detected, no concept drift)
- Rolling window evaluation captures gradual degradation patterns
- Threshold configuration is critical for reducing false positive alerts

**Limitations**:
- Current implementation uses synthetic baseline distributions for PSI calculation (normal approximation)
- Monitoring assumes batch processing; real-time streaming requires additional architecture
- Alert system is file-based; production deployments need database integration
- Model comparison logic is simplified; production systems should use more sophisticated A/B testing

## Business / Real-World Impact

**Use Cases**:
- **Production ML Monitoring**: Continuously monitor deployed models in production environments
- **Regulatory Compliance**: Financial services and healthcare industries require model monitoring for regulatory compliance
- **Cost Optimization**: Early detection prevents costly bad predictions and business decisions
- **Model Lifecycle Management**: Systematic approach to model refresh and retraining decisions

**Beneficiaries**:
- **ML Engineers**: Automated monitoring reduces manual monitoring overhead
- **Data Science Teams**: Actionable alerts enable proactive model maintenance
- **Business Stakeholders**: Executive dashboards provide model health visibility
- **Operations Teams**: Clear retraining recommendations support operational decisions

**Decision Support**:
- Retraining decisions based on quantitative drift metrics rather than time-based schedules
- Resource allocation prioritization (retrain models with highest drift/performance degradation)
- Risk assessment through drift severity classification
- Performance SLA monitoring through threshold-based alerting

## Project Structure

```
ML-Model-Monitoring-Drift-Detection/
├── config/
│   └── config.yaml              # Configuration file with thresholds and settings
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Utility functions for config, file I/O
│   ├── model_training.py        # Baseline model training and distribution capture
│   ├── drift_detection.py       # Data drift (PSI, KL-Divergence) and concept drift detection
│   ├── performance_monitoring.py # Performance tracking and degradation detection
│   ├── alerts.py                # Alerting system and retraining logic
│   ├── monitoring_pipeline.py   # Main orchestration pipeline
│   ├── dashboard.py             # Static visualization and reporting
│   └── streamlit_dashboard.py   # Interactive web dashboard
├── models/                      # Saved models (generated)
│   ├── baseline_model.pkl
│   └── preprocessors.pkl
├── artifacts/                   # Monitoring artifacts (generated)
│   ├── baseline_metrics.json
│   ├── baseline_distributions.json
│   ├── performance_history.json
│   └── alerts_history.json
├── dashboards/                  # Generated visualizations (generated)
│   ├── feature_drift_*.png
│   └── performance_trends_*.png
├── logs/                        # Application logs (generated)
├── main.py                      # Main execution script
├── run_dashboard.py             # Streamlit dashboard launcher
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## How to Run This Project

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/MBGIRISH/ML-Model-Monitoring-Drift-Detection.git
cd ML-Model-Monitoring-Drift-Detection
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the monitoring pipeline**:
```bash
python3 main.py
```

This will:
- Train the baseline model (if not already trained)
- Initialize monitoring components
- Simulate production monitoring with time-based data splits
- Generate monitoring results, alerts, and reports

5. **Launch interactive dashboard** (optional):
```bash
streamlit run src/streamlit_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Configuration

Edit `config/config.yaml` to customize thresholds:
- Data drift thresholds (PSI, KL-Divergence)
- Performance thresholds (min AUC, accuracy)
- Alert sensitivity levels
- Retraining decision criteria

## Dashboard Screenshots

### Overview Dashboard
The overview tab provides a comprehensive view of model health, including current performance metrics compared to baseline, drift status indicators, and a confusion matrix visualization.

![Overview Dashboard](dashboards/feature_drift_20260108_113126.png)

### Performance Monitoring
Performance trends are tracked over time with interactive charts showing AUC, Accuracy, Precision, and Recall. Baseline reference lines enable quick comparison of current performance against established benchmarks.

![Performance Dashboard](dashboards/performance_trends_20260108_113126.png)

### Drift Detection Analysis
Feature-level drift detection visualizes PSI scores for all features, with color-coding to indicate drift severity. Features are sorted by drift magnitude, making it easy to identify the most problematic features.

![Drift Detection Dashboard](dashboards/feature_drift_20260108_112158.png)

## Future Improvements

**Model Enhancements**:
- Implement incremental drift detection for streaming data scenarios
- Add support for regression models and multi-class classification
- Integrate model-agnostic drift detection methods (e.g., MMD, Wasserstein distance)
- Implement adaptive threshold adjustment based on historical patterns

**Data Improvements**:
- Replace synthetic baseline distribution generation with stored empirical distributions
- Add support for high-dimensional feature spaces with dimensionality reduction
- Implement data quality checks (missing values, outliers, schema validation)
- Support for time-series feature drift detection

**Deployment & Scaling**:
- Database integration (PostgreSQL, MongoDB) for artifact storage
- Real-time monitoring pipeline using Apache Kafka or similar streaming platforms
- Containerization with Docker for easy deployment
- Integration with MLflow for model versioning and experiment tracking
- Cloud deployment templates (AWS, GCP, Azure)

**Alerting Enhancements**:
- Integration with external alerting systems (PagerDuty, Slack, email)
- Alert fatigue reduction through intelligent grouping and prioritization
- Custom alert rules for domain-specific scenarios
- Alert escalation policies based on drift persistence

## Key Learnings

**Technical Learnings**:
- PSI calculation requires careful binning strategy; bin count significantly affects sensitivity
- Synthetic baseline distribution generation (normal approximation) works for initial implementation but empirical distributions are more accurate
- Performance monitoring requires sufficient sample sizes per evaluation window to reduce variance
- Concept drift detection through performance metrics is more reliable than feature-based methods when ground truth labels are available

**Data Science Learnings**:
- Threshold selection is domain-specific and requires iteration based on business impact
- False positive alerts can be costly; conservative thresholds with high precision are preferable
- Model monitoring is as important as model development; monitoring infrastructure should be built alongside models
- Visualization is critical for stakeholder communication; both technical and executive dashboards serve different needs
- Retraining decisions should consider multiple factors (drift severity, performance degradation, data availability, business context)

**Production Considerations**:
- Configuration management enables quick threshold adjustments without code changes
- Modular architecture allows independent updates to monitoring components
- Logging and artifact storage are essential for audit trails and debugging
- Interactive dashboards reduce the need for custom reporting scripts

## References

**Datasets**:
- Telco Customer Churn Dataset: IBM Watson Analytics (publicly available)

**Papers & Articles**:
- Population Stability Index (PSI) methodology widely used in financial services for model monitoring
- Concept drift detection techniques from "Learning Under Concept Drift: An Overview" (Gama et al.)
- Statistical methods for distribution comparison (Kolmogorov-Smirnov, Mann-Whitney U tests)

**Tools & Libraries**:
- scikit-learn: Machine learning model implementation
- scipy: Statistical functions and tests
- streamlit: Interactive dashboard framework
- plotly: Interactive visualization library

**Industry Practices**:
- Model monitoring best practices from ML Operations (MLOps) communities
- Drift detection thresholds based on industry standards (PSI > 0.2 indicates significant drift)
- Production ML monitoring patterns from cloud ML platforms (AWS SageMaker, Google Vertex AI, Azure ML)

---

**Repository**: [https://github.com/MBGIRISH/ML-Model-Monitoring-Drift-Detection](https://github.com/MBGIRISH/ML-Model-Monitoring-Drift-Detection)
