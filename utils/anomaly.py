"""
Aadhaar Anomaly Detection Utilities

This module provides methods for detecting unusual patterns in
aggregated Aadhaar data. All outputs are labeled as operational
signals, not fraud indicators.

Methods: Z-Score, IQR, Isolation Forest
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.ensemble import IsolationForest


def detect_zscore_anomalies(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect anomalies using Z-Score method.
    
    Points with |z-score| > threshold are flagged as anomalies.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        threshold: Z-score threshold (default 3.0)
        group_by: Optional column to compute z-scores within groups
        
    Returns:
        DataFrame with anomaly flags and z-scores
    """
    df = df.copy()
    
    if group_by and group_by in df.columns:
        # Compute z-scores within each group
        grouped = df.groupby(group_by)[column]
        df['zscore'] = grouped.transform(lambda x: (x - x.mean()) / x.std())
    else:
        # Global z-scores
        mean = df[column].mean()
        std = df[column].std()
        df['zscore'] = (df[column] - mean) / std
    
    df['is_anomaly_zscore'] = df['zscore'].abs() > threshold
    
    return df


def detect_iqr_anomalies(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect anomalies using IQR (Interquartile Range) method.
    
    Points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        multiplier: IQR multiplier (default 1.5)
        group_by: Optional column to compute IQR within groups
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    def compute_iqr_bounds(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return (series < lower) | (series > upper)
    
    if group_by and group_by in df.columns:
        df['is_anomaly_iqr'] = df.groupby(group_by)[column].transform(compute_iqr_bounds)
    else:
        df['is_anomaly_iqr'] = compute_iqr_bounds(df[column])
    
    return df


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest algorithm.
    
    This method can detect multivariate anomalies.
    
    Args:
        df: Input DataFrame
        columns: Columns to use for anomaly detection
        contamination: Expected proportion of anomalies (default 5%)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with anomaly flags and anomaly scores
    """
    df = df.copy()
    
    # Select features for isolation forest
    features = df[columns].fillna(0)
    
    # Fit isolation forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    # -1 for anomalies, 1 for normal
    predictions = iso_forest.fit_predict(features)
    df['is_anomaly_iforest'] = predictions == -1
    df['anomaly_score'] = -iso_forest.score_samples(features)
    
    return df


def classify_severity(
    zscore: float,
    is_iqr_anomaly: bool,
    is_iforest_anomaly: bool = False
) -> str:
    """
    Classify anomaly severity based on detection methods.
    
    Categories:
    - Critical: Flagged by all methods or z-score > 4
    - High: Flagged by 2+ methods or z-score > 3
    - Medium: Flagged by 1 method
    - Low: Borderline (z-score between 2 and 3)
    
    Args:
        zscore: Absolute z-score value
        is_iqr_anomaly: Whether flagged by IQR method
        is_iforest_anomaly: Whether flagged by Isolation Forest
        
    Returns:
        Severity level string
    """
    flags = sum([abs(zscore) > 3, is_iqr_anomaly, is_iforest_anomaly])
    
    if abs(zscore) > 4 or flags == 3:
        return 'Critical'
    elif abs(zscore) > 3 or flags >= 2:
        return 'High'
    elif flags == 1:
        return 'Medium'
    elif abs(zscore) > 2:
        return 'Low'
    else:
        return 'Normal'


def generate_anomaly_report(
    df: pd.DataFrame,
    metrics: List[str],
    date_column: str = 'date',
    state_column: Optional[str] = 'state',
    district_column: Optional[str] = None,
    zscore_threshold: float = 3.0,
    use_isolation_forest: bool = True
) -> pd.DataFrame:
    """
    Generate a comprehensive anomaly detection report.
    
    Output includes: Date, State, District (if available), Metric, 
    Value, Z-Score, Severity, Detection Methods.
    
    ⚠️ Results labeled as "Operational Signals" - NOT fraud indicators.
    
    Args:
        df: Input DataFrame with computed indicators
        metrics: List of metric columns to analyze
        date_column: Date column name
        state_column: State column name (optional)
        district_column: District column name (optional)
        zscore_threshold: Z-score threshold for flagging
        use_isolation_forest: Whether to use Isolation Forest
        
    Returns:
        Anomaly report DataFrame
    """
    anomaly_records = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Create working copy
        working_df = df.copy()
        
        # Apply detection methods
        group_col = state_column if state_column in df.columns else None
        
        working_df = detect_zscore_anomalies(
            working_df, metric, zscore_threshold, group_col
        )
        working_df = detect_iqr_anomalies(working_df, metric, 1.5, group_col)
        
        # Isolation Forest (multivariate)
        if use_isolation_forest and len(metrics) > 1:
            available_metrics = [m for m in metrics if m in working_df.columns]
            if len(available_metrics) >= 2:
                working_df = detect_isolation_forest_anomalies(
                    working_df, available_metrics
                )
                iforest_col = 'is_anomaly_iforest'
            else:
                working_df['is_anomaly_iforest'] = False
                iforest_col = 'is_anomaly_iforest'
        else:
            working_df['is_anomaly_iforest'] = False
            iforest_col = 'is_anomaly_iforest'
        
        # Filter anomalies
        anomaly_mask = (
            working_df['is_anomaly_zscore'] | 
            working_df['is_anomaly_iqr'] | 
            working_df.get(iforest_col, False)
        )
        
        anomalies = working_df[anomaly_mask].copy()
        
        for _, row in anomalies.iterrows():
            # Determine severity
            severity = classify_severity(
                row.get('zscore', 0),
                row.get('is_anomaly_iqr', False),
                row.get('is_anomaly_iforest', False)
            )
            
            # Build detection methods list
            methods = []
            if row.get('is_anomaly_zscore', False):
                methods.append('Z-Score')
            if row.get('is_anomaly_iqr', False):
                methods.append('IQR')
            if row.get('is_anomaly_iforest', False):
                methods.append('IsolationForest')
            
            record = {
                'date': row[date_column],
                'metric': metric,
                'value': row[metric],
                'zscore': abs(row.get('zscore', 0)),
                'severity': severity,
                'detection_methods': ', '.join(methods),
                'signal_type': 'Operational Signal'
            }
            
            if state_column and state_column in row.index:
                record['state'] = row[state_column]
            
            if district_column and district_column in row.index:
                record['district'] = row[district_column]
            
            anomaly_records.append(record)
    
    report = pd.DataFrame(anomaly_records)
    
    if not report.empty:
        # Sort by severity and date
        severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        report['severity_rank'] = report['severity'].map(severity_order)
        report = report.sort_values(['severity_rank', 'date']).drop(columns=['severity_rank'])
        report = report.reset_index(drop=True)
    
    return report


def get_anomaly_summary(anomaly_report: pd.DataFrame) -> dict:
    """
    Generate summary statistics from anomaly report.
    
    Args:
        anomaly_report: Output from generate_anomaly_report
        
    Returns:
        Dictionary with summary statistics
    """
    if anomaly_report.empty:
        return {
            'total_anomalies': 0,
            'by_severity': {},
            'by_metric': {},
            'by_state': {}
        }
    
    summary = {
        'total_anomalies': len(anomaly_report),
        'by_severity': anomaly_report['severity'].value_counts().to_dict(),
        'by_metric': anomaly_report['metric'].value_counts().to_dict()
    }
    
    if 'state' in anomaly_report.columns:
        summary['by_state'] = anomaly_report['state'].value_counts().to_dict()
    
    return summary


def get_administrative_suggestions(anomaly_report: pd.DataFrame) -> List[str]:
    """
    Generate administrative action suggestions based on anomaly patterns.
    
    These are advisory recommendations for planning purposes.
    
    Args:
        anomaly_report: Output from generate_anomaly_report
        
    Returns:
        List of suggested actions
    """
    suggestions = []
    
    if anomaly_report.empty:
        suggestions.append("✓ No significant anomalies detected. Operations appear normal.")
        return suggestions
    
    # Analyze patterns
    summary = get_anomaly_summary(anomaly_report)
    
    # Critical severity suggestions
    critical_count = summary['by_severity'].get('Critical', 0)
    if critical_count > 0:
        suggestions.append(
            f"⚠️ {critical_count} critical operational signals detected. "
            "Recommend immediate review of affected locations/periods."
        )
    
    # High workload suggestions
    update_metrics = ['Service_Workload', 'Total_Updates', 'Total_Biometric_Updates']
    for metric in update_metrics:
        if metric in summary['by_metric']:
            friendly_name = metric.replace('_', ' ').replace('Service Workload', 'service workload').replace('Total', 'total')
            suggestions.append(
                f"📊 Unusual {friendly_name} patterns detected. "
                "Consider reviewing staffing at enrolment centres."
            )
    
    # Geographic concentration
    if 'by_state' in summary:
        top_states = list(summary['by_state'].keys())[:3]
        if top_states:
            suggestions.append(
                f"📍 Anomalies concentrated in: {', '.join(top_states)}. "
                "Recommend targeted operational review."
            )
    
    # Seasonal patterns
    if 'date' in anomaly_report.columns:
        anomaly_report['month'] = pd.to_datetime(anomaly_report['date']).dt.month
        peak_months = anomaly_report['month'].value_counts().head(2).index.tolist()
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        if peak_months:
            peak_str = ', '.join([month_names.get(m, str(m)) for m in peak_months])
            suggestions.append(
                f"📅 Higher anomaly concentration in {peak_str}. "
                "Plan for increased monitoring during these periods."
            )
    
    return suggestions


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
    
    from utils.preprocessing import load_and_preprocess_all_data, merge_all_datasets
    from utils.indicators import compute_all_indicators
    
    DATA_PATH = r"d:\COURSES\AADHAAR HACKATHON"
    
    enrolment, demographic, biometric = load_and_preprocess_all_data(
        DATA_PATH, aggregate_level='national'
    )
    
    merged = merge_all_datasets(enrolment, demographic, biometric)
    df = compute_all_indicators(merged)
    
    # Generate anomaly report
    metrics = ['Total_Enrolment', 'Total_Updates', 'Update_Pressure_Index']
    report = generate_anomaly_report(df, metrics)
    
    print("\nAnomaly Report:")
    print(report.head(10))
    
    print("\nSummary:")
    print(get_anomaly_summary(report))
    
    print("\nSuggested Actions:")
    for suggestion in get_administrative_suggestions(report):
        print(f"  {suggestion}")
