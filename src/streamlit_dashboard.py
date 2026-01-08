"""
Streamlit Dashboard for ML Model Monitoring & Drift Detection
Interactive web-based dashboard for real-time monitoring

Author: M B GIRISH
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from src module
try:
    from src.utils import load_config, load_artifact
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import load_config, load_artifact

# Page configuration
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_monitoring_data(artifacts_dir):
    """Load monitoring data with caching"""
    data = {}
    
    # Load baseline metrics
    baseline_metrics_path = os.path.join(artifacts_dir, "baseline_metrics.json")
    if os.path.exists(baseline_metrics_path):
        data['baseline_metrics'] = load_artifact(baseline_metrics_path)
    
    # Load baseline distributions
    baseline_dist_path = os.path.join(artifacts_dir, "baseline_distributions.json")
    if os.path.exists(baseline_dist_path):
        data['baseline_distributions'] = load_artifact(baseline_dist_path)
    
    # Load performance history
    perf_history_path = os.path.join(artifacts_dir, "performance_history.json")
    if os.path.exists(perf_history_path):
        data['performance_history'] = load_artifact(perf_history_path)
    
    # Load alerts history
    alerts_path = os.path.join(artifacts_dir, "alerts_history.json")
    if os.path.exists(alerts_path):
        data['alerts'] = load_artifact(alerts_path)
    
    # Load latest monitoring results
    artifacts_files = [f for f in os.listdir(artifacts_dir) if f.startswith('monitoring_results_')]
    if artifacts_files:
        latest_file = sorted(artifacts_files)[-1]
        latest_path = os.path.join(artifacts_dir, latest_file)
        data['latest_results'] = load_artifact(latest_path)
    
    return data

def plot_performance_metrics(performance_history, baseline_metrics):
    """Plot performance metrics over time"""
    if not performance_history:
        st.warning("No performance history available")
        return
    
    df = pd.DataFrame(performance_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AUC Over Time', 'Accuracy Over Time', 
                       'Precision Over Time', 'Recall Over Time'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    metrics = ['auc', 'accuracy', 'precision', 'recall']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, pos in zip(metrics, positions):
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=pos[0], col=pos[1]
            )
            
            # Add baseline line
            if metric in baseline_metrics:
                fig.add_hline(
                    y=baseline_metrics[metric],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Baseline: {baseline_metrics[metric]:.4f}",
                    row=pos[0], col=pos[1]
                )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Performance Metrics Over Time",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_drift(drift_results):
    """Plot feature drift scores"""
    if not drift_results or 'features' not in drift_results:
        st.warning("No drift data available")
        return
    
    features_data = []
    for feature, results in drift_results['features'].items():
        features_data.append({
            'feature': feature,
            'psi': results['psi'],
            'kl_divergence': results['kl_divergence'],
            'drift_detected': results['drift_detected'],
            'severity': results.get('drift_severity', 'LOW')
        })
    
    df = pd.DataFrame(features_data)
    df = df.sort_values('psi', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Color by drift detection
    colors = ['#f44336' if d else '#4caf50' for d in df['drift_detected']]
    
    fig.add_trace(go.Bar(
        x=df['feature'],
        y=df['psi'],
        marker_color=colors,
        text=df['psi'].round(4),
        textposition='outside',
        name='PSI Score'
    ))
    
    # Add threshold line
    config = load_config()
    threshold = config['data_drift']['psi_threshold']
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Feature Drift Detection (PSI Scores)",
        xaxis_title="Features",
        yaxis_title="PSI Score",
        height=500,
        xaxis={'tickangle': -45},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top drifted features
    st.subheader("Top Features with Drift")
    drifted_features = df[df['drift_detected']].head(10)
    st.dataframe(
        drifted_features[['feature', 'psi', 'kl_divergence', 'severity']],
        use_container_width=True
    )

def plot_confusion_matrix(confusion_matrix_data):
    """Plot confusion matrix"""
    cm = np.array(confusion_matrix_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        height=400,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_alerts(alerts):
    """Display alerts in a formatted way"""
    if not alerts:
        st.success("✅ No alerts - Model is healthy!")
        return
    
    # Group alerts by severity
    high_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
    medium_alerts = [a for a in alerts if a.get('severity') == 'MEDIUM']
    low_alerts = [a for a in alerts if a.get('severity') == 'LOW']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Severity", len(high_alerts), delta=None)
        if high_alerts:
            for alert in high_alerts[:3]:
                st.markdown(f'<div class="alert-high">🔴 {alert.get("message", "N/A")}</div>', 
                           unsafe_allow_html=True)
    
    with col2:
        st.metric("Medium Severity", len(medium_alerts), delta=None)
        if medium_alerts:
            for alert in medium_alerts[:3]:
                st.markdown(f'<div class="alert-medium">🟠 {alert.get("message", "N/A")}</div>', 
                           unsafe_allow_html=True)
    
    with col3:
        st.metric("Low Severity", len(low_alerts), delta=None)
        if low_alerts:
            for alert in low_alerts[:3]:
                st.markdown(f'<div class="alert-low">🟢 {alert.get("message", "N/A")}</div>', 
                           unsafe_allow_html=True)
    
    # Show all alerts in expander
    with st.expander("View All Alerts"):
        alerts_df = pd.DataFrame(alerts)
        if not alerts_df.empty:
            st.dataframe(alerts_df[['timestamp', 'type', 'severity', 'message']], 
                        use_container_width=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 ML Model Monitoring & Drift Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load configuration
    try:
        config = load_config()
        artifacts_dir = config['paths']['artifacts_dir']
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("**Model**: Churn Classifier\n\n**Type**: Classification")
        
        st.header("📈 Monitoring Status")
        
        # Load data
        with st.spinner("Loading monitoring data..."):
            data = load_monitoring_data(artifacts_dir)
        
        if 'baseline_metrics' in data:
            st.success("✅ Baseline model loaded")
        else:
            st.warning("⚠️ Baseline model not found. Run training first.")
        
        if 'performance_history' in data:
            st.success(f"✅ {len(data['performance_history'])} performance evaluations")
        else:
            st.info("ℹ️ No performance history yet")
        
        if 'alerts' in data:
            st.info(f"📢 {len(data['alerts'])} total alerts")
        
        st.header("🔄 Refresh")
        if st.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    if 'baseline_metrics' not in data:
        st.error("Please train the baseline model first by running: `python main.py`")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Performance", 
        "🔍 Drift Detection", 
        "🚨 Alerts", 
        "📋 Reports"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Model Health Overview")
        
        baseline_metrics = data['baseline_metrics']
        
        # Current metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if 'latest_results' in data and 'performance' in data['latest_results']:
            current_perf = data['latest_results']['performance']
            
            with col1:
                current_auc = current_perf.get('auc', 0)
                baseline_auc = baseline_metrics.get('auc', 0)
                delta_auc = current_auc - baseline_auc
                st.metric("AUC", f"{current_auc:.4f}", f"{delta_auc:+.4f}")
            
            with col2:
                current_acc = current_perf.get('accuracy', 0)
                baseline_acc = baseline_metrics.get('accuracy', 0)
                delta_acc = current_acc - baseline_acc
                st.metric("Accuracy", f"{current_acc:.4f}", f"{delta_acc:+.4f}")
            
            with col3:
                st.metric("Precision", f"{current_perf.get('precision', 0):.4f}")
            
            with col4:
                st.metric("Recall", f"{current_perf.get('recall', 0):.4f}")
            
            with col5:
                st.metric("F1-Score", f"{current_perf.get('f1_score', 0):.4f}")
        else:
            st.info("No recent performance data. Run monitoring to see current metrics.")
        
        # Baseline metrics
        st.subheader("Baseline Model Metrics")
        baseline_col1, baseline_col2, baseline_col3, baseline_col4, baseline_col5 = st.columns(5)
        
        with baseline_col1:
            st.metric("Baseline AUC", f"{baseline_metrics.get('auc', 0):.4f}")
        with baseline_col2:
            st.metric("Baseline Accuracy", f"{baseline_metrics.get('accuracy', 0):.4f}")
        with baseline_col3:
            st.metric("Baseline Precision", f"{baseline_metrics.get('precision', 0):.4f}")
        with baseline_col4:
            st.metric("Baseline Recall", f"{baseline_metrics.get('recall', 0):.4f}")
        with baseline_col5:
            st.metric("Baseline F1", f"{baseline_metrics.get('f1_score', 0):.4f}")
        
        # Drift status
        st.subheader("Drift Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'latest_results' in data and 'data_drift' in data['latest_results']:
                data_drift = data['latest_results']['data_drift']
                drift_detected = data_drift.get('overall_drift_detected', False)
                if drift_detected:
                    st.error(f"⚠️ Data Drift Detected")
                    st.write(f"Features with drift: {data_drift['drift_summary']['features_with_drift']}")
                else:
                    st.success("✅ No Data Drift")
            else:
                st.info("No drift data available")
        
        with col2:
            if 'latest_results' in data and 'concept_drift' in data['latest_results']:
                concept_drift = data['latest_results']['concept_drift']
                concept_drift_detected = concept_drift.get('concept_drift_detected', False)
                if concept_drift_detected:
                    st.error(f"⚠️ Concept Drift Detected")
                    st.write(f"Severity: {concept_drift.get('drift_severity', 'N/A')}")
                else:
                    st.success("✅ No Concept Drift")
            else:
                st.info("No concept drift data available")
        
        # Confusion Matrix
        if 'latest_results' in data and 'performance' in data['latest_results']:
            st.subheader("Latest Confusion Matrix")
            cm = data['latest_results']['performance'].get('confusion_matrix')
            if cm:
                plot_confusion_matrix(cm)
    
    # Tab 2: Performance
    with tab2:
        st.header("Performance Monitoring")
        
        if 'performance_history' in data and data['performance_history']:
            plot_performance_metrics(data['performance_history'], baseline_metrics)
            
            # Performance table
            st.subheader("Performance History")
            perf_df = pd.DataFrame(data['performance_history'])
            st.dataframe(perf_df[['timestamp', 'accuracy', 'auc', 'precision', 'recall', 'f1_score']], 
                        use_container_width=True)
        else:
            st.info("No performance history available. Run monitoring to generate performance data.")
    
    # Tab 3: Drift Detection
    with tab3:
        st.header("Data & Concept Drift Detection")
        
        if 'latest_results' in data and 'data_drift' in data['latest_results']:
            st.subheader("Feature Drift Analysis")
            plot_feature_drift(data['latest_results']['data_drift'])
            
            # Drift details
            st.subheader("Drift Details by Feature")
            drift_features = data['latest_results']['data_drift']['features']
            drift_df = pd.DataFrame([
                {
                    'feature': k,
                    'psi': v['psi'],
                    'kl_divergence': v['kl_divergence'],
                    'drift_detected': v['drift_detected'],
                    'severity': v.get('drift_severity', 'LOW'),
                    'baseline_mean': v['baseline_stats']['mean'],
                    'current_mean': v['current_stats']['mean']
                }
                for k, v in drift_features.items()
            ])
            st.dataframe(drift_df, use_container_width=True)
        else:
            st.info("No drift detection data available")
        
        # Concept drift
        if 'latest_results' in data and 'concept_drift' in data['latest_results']:
            st.subheader("Concept Drift Analysis")
            concept_drift = data['latest_results']['concept_drift']
            
            if concept_drift.get('concept_drift_detected'):
                st.error("⚠️ Concept Drift Detected")
                st.write(f"**Severity**: {concept_drift.get('drift_severity', 'N/A')}")
                
                # Performance changes
                if 'performance_changes' in concept_drift:
                    st.write("**Performance Changes:**")
                    perf_changes = concept_drift['performance_changes']
                    for metric, change in perf_changes.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**{metric.upper()}**")
                        with col2:
                            st.write(f"Baseline: {change['baseline']:.4f}")
                        with col3:
                            change_pct = change.get('change_percentage', 0)
                            if change.get('degraded'):
                                st.error(f"Current: {change['current']:.4f} ({change_pct:.2f}%)")
                            else:
                                st.success(f"Current: {change['current']:.4f} ({change_pct:+.2f}%)")
            else:
                st.success("✅ No Concept Drift Detected")
    
    # Tab 4: Alerts
    with tab4:
        st.header("Alert Management")
        
        if 'alerts' in data and data['alerts']:
            display_alerts(data['alerts'])
            
            # Alert statistics
            st.subheader("Alert Statistics")
            alerts_df = pd.DataFrame(data['alerts'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Alerts by type
                alert_types = alerts_df['type'].value_counts()
                fig = px.pie(
                    values=alert_types.values,
                    names=alert_types.index,
                    title="Alerts by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Alerts by severity
                alert_severity = alerts_df['severity'].value_counts()
                fig = px.bar(
                    x=alert_severity.index,
                    y=alert_severity.values,
                    title="Alerts by Severity",
                    labels={'x': 'Severity', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alerts generated yet")
    
    # Tab 5: Reports
    with tab5:
        st.header("Monitoring Reports")
        
        # Retraining recommendations
        if 'latest_results' in data and 'retraining_decision' in data['latest_results']:
            st.subheader("Retraining Recommendation")
            retrain_decision = data['latest_results']['retraining_decision']
            
            col1, col2 = st.columns(2)
            
            with col1:
                should_retrain = retrain_decision.get('should_retrain', False)
                if should_retrain:
                    st.error("🔄 **RETRAINING RECOMMENDED**")
                else:
                    st.success("✅ **No Retraining Needed**")
            
            with col2:
                st.write(f"**Priority**: {retrain_decision.get('priority', 'N/A')}")
                st.write(f"**Reason**: {retrain_decision.get('reason', 'N/A')}")
            
            if retrain_decision.get('conditions_met'):
                st.write("**Conditions Met:**")
                for condition in retrain_decision['conditions_met']:
                    st.write(f"- {condition}")
        
        # Executive summary
        st.subheader("Executive Summary")
        
        summary_text = f"""
        **Model**: Churn Classifier
        
        **Baseline Performance**:
        - AUC: {baseline_metrics.get('auc', 0):.4f}
        - Accuracy: {baseline_metrics.get('accuracy', 0):.4f}
        
        **Current Status**:
        """
        
        if 'latest_results' in data:
            if 'performance' in data['latest_results']:
                current_perf = data['latest_results']['performance']
                summary_text += f"""
        - Current AUC: {current_perf.get('auc', 0):.4f}
        - Current Accuracy: {current_perf.get('accuracy', 0):.4f}
        """
            
            if 'data_drift' in data['latest_results']:
                data_drift = data['latest_results']['data_drift']
                summary_text += f"""
        - Data Drift: {'Detected' if data_drift.get('overall_drift_detected') else 'None'}
        - Features with Drift: {data_drift['drift_summary']['features_with_drift']}
        """
            
            if 'concept_drift' in data['latest_results']:
                concept_drift = data['latest_results']['concept_drift']
                summary_text += f"""
        - Concept Drift: {'Detected' if concept_drift.get('concept_drift_detected') else 'None'}
        """
        
        if 'alerts' in data:
            summary_text += f"""
        - Total Alerts: {len(data['alerts'])}
        """
        
        st.markdown(summary_text)
        
        # Download report
        st.download_button(
            label="📥 Download Executive Summary",
            data=summary_text,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

