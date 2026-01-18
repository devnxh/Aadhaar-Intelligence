"""
Aadhaar Indicators & Feature Engineering

This module computes derived metrics for analyzing Aadhaar
enrolment and update patterns at an aggregated level.

All indicators are planning-level metrics for administrative decision support.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_total_enrolment(df: pd.DataFrame) -> pd.Series:
    """
    Compute total enrolment across all age groups.
    
    Total_Enrolment = age_0_5 + age_5_17 + age_18_greater
    
    Args:
        df: DataFrame with age group columns
        
    Returns:
        Series with total enrolment values
    """
    return df['age_0_5'] + df['age_5_17'] + df['age_18_greater']


def compute_total_demographic_updates(df: pd.DataFrame) -> pd.Series:
    """
    Compute total demographic updates across age groups.
    
    Total_Demographic_Updates = demo_age_5_17 + demo_age_17_
    
    Args:
        df: DataFrame with demographic update columns
        
    Returns:
        Series with total demographic update values
    """
    return df['demo_age_5_17'] + df['demo_age_17_']


def compute_total_biometric_updates(df: pd.DataFrame) -> pd.Series:
    """
    Compute total biometric updates across age groups.
    
    Total_Biometric_Updates = bio_age_5_17 + bio_age_17_
    
    Args:
        df: DataFrame with biometric update columns
        
    Returns:
        Series with total biometric update values
    """
    return df['bio_age_5_17'] + df['bio_age_17_']


def compute_total_updates(df: pd.DataFrame) -> pd.Series:
    """
    Compute total updates (demographic + biometric).
    
    Total_Updates = Total_Demographic_Updates + Total_Biometric_Updates
    
    Args:
        df: DataFrame with Total_Demographic_Updates and Total_Biometric_Updates
        
    Returns:
        Series with total update values
    """
    demo = df.get('Total_Demographic_Updates', 0)
    bio = df.get('Total_Biometric_Updates', 0)
    
    if isinstance(demo, int):
        demo = compute_total_demographic_updates(df)
    if isinstance(bio, int):
        bio = compute_total_biometric_updates(df)
    
    return demo + bio


def compute_service_workload(df: pd.DataFrame) -> pd.Series:
    """
    Compute Service Workload (formerly Update Pressure Index).
    
    Service_Workload = Total_Updates / Total_Enrolment
    
    This shows how many updates are happening compared to new registrations.
    Higher values mean centres are busier with updates than new cards.
    
    Args:
        df: DataFrame with Total_Updates and Total_Enrolment columns
        
    Returns:
        Series with service workload values
    """
    total_updates = df.get('Total_Updates')
    total_enrolment = df.get('Total_Enrolment')
    
    if total_updates is None:
        total_updates = compute_total_updates(df)
    if total_enrolment is None:
        total_enrolment = compute_total_enrolment(df)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            total_enrolment > 0,
            total_updates / total_enrolment,
            0
        )
    
    return pd.Series(result, index=df.index)


def compute_youth_update_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute Youth Update Rate (formerly Age Transition Rate).
    
    Youth_Update_Rate = bio_age_17_ / age_5_17
    
    This shows how many youth (5-17) are getting biometric updates
    compared to new youth registrations.
    
    Args:
        df: DataFrame with bio_age_17_ and age_5_17 columns
        
    Returns:
        Series with youth update rate values
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            df['age_5_17'] > 0,
            df['bio_age_17_'] / df['age_5_17'],
            0
        )
    
    return pd.Series(result, index=df.index)


def compute_enrolment_growth_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute month-over-month enrolment growth rate.
    
    Growth_Rate = (Current - Previous) / Previous * 100
    
    Args:
        df: DataFrame sorted by date with Total_Enrolment
        
    Returns:
        Series with growth rate percentages
    """
    total_enrolment = df.get('Total_Enrolment')
    if total_enrolment is None:
        total_enrolment = compute_total_enrolment(df)
    
    return total_enrolment.pct_change() * 100


def compute_update_growth_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute month-over-month update growth rate.
    
    Args:
        df: DataFrame sorted by date with Total_Updates
        
    Returns:
        Series with growth rate percentages
    """
    total_updates = df.get('Total_Updates')
    if total_updates is None:
        total_updates = compute_total_updates(df)
    
    return total_updates.pct_change() * 100


def compute_biometric_to_demographic_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute ratio of biometric updates to demographic updates.
    
    Ratio = Total_Biometric_Updates / Total_Demographic_Updates
    
    Higher values indicate more biometric maintenance activity.
    
    Args:
        df: DataFrame with update columns
        
    Returns:
        Series with ratio values
    """
    bio = df.get('Total_Biometric_Updates')
    demo = df.get('Total_Demographic_Updates')
    
    if bio is None:
        bio = compute_total_biometric_updates(df)
    if demo is None:
        demo = compute_total_demographic_updates(df)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(demo > 0, bio / demo, 0)
    
    return pd.Series(result, index=df.index)


def compute_child_enrolment_share(df: pd.DataFrame) -> pd.Series:
    """
    Compute share of child enrolments (0-17) in total enrolments.
    
    Child_Share = (age_0_5 + age_5_17) / Total_Enrolment * 100
    
    Args:
        df: DataFrame with age group columns
        
    Returns:
        Series with child enrolment share percentages
    """
    child_enrolment = df['age_0_5'] + df['age_5_17']
    total = df.get('Total_Enrolment')
    
    if total is None:
        total = compute_total_enrolment(df)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(total > 0, child_enrolment / total * 100, 0)
    
    return pd.Series(result, index=df.index)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived indicators and add them to the DataFrame.
    
    This is the main function to call for feature engineering.
    
    Args:
        df: Merged DataFrame with enrolment, demographic, and biometric columns
        
    Returns:
        DataFrame with all computed indicators added
    """
    df = df.copy()
    
    # Core totals
    if 'age_0_5' in df.columns:
        df['Total_Enrolment'] = compute_total_enrolment(df)
    
    if 'demo_age_5_17' in df.columns:
        df['Total_Demographic_Updates'] = compute_total_demographic_updates(df)
    
    if 'bio_age_5_17' in df.columns:
        df['Total_Biometric_Updates'] = compute_total_biometric_updates(df)
    
    # Combined updates
    if 'Total_Demographic_Updates' in df.columns and 'Total_Biometric_Updates' in df.columns:
        df['Total_Updates'] = df['Total_Demographic_Updates'] + df['Total_Biometric_Updates']
    
    # Service Workload (formerly Update Pressure Index)
    if 'Total_Updates' in df.columns and 'Total_Enrolment' in df.columns:
        df['Service_Workload'] = compute_service_workload(df)
        # Keep old name for backward compatibility
        df['Update_Pressure_Index'] = df['Service_Workload']
    
    if 'bio_age_17_' in df.columns and 'age_5_17' in df.columns:
        df['Youth_Update_Rate'] = compute_youth_update_rate(df)
        # Keep old name for backward compatibility
        df['Age_Transition_Rate'] = df['Youth_Update_Rate']
    
    # Growth rates (requires sorted data)
    if 'Total_Enrolment' in df.columns:
        df['Enrolment_Growth_Rate'] = compute_enrolment_growth_rate(df)
    
    if 'Total_Updates' in df.columns:
        df['Update_Growth_Rate'] = compute_update_growth_rate(df)
    
    # Additional insights
    if 'Total_Biometric_Updates' in df.columns and 'Total_Demographic_Updates' in df.columns:
        df['Bio_Demo_Ratio'] = compute_biometric_to_demographic_ratio(df)
    
    if 'age_0_5' in df.columns:
        df['Child_Share'] = compute_child_enrolment_share(df)
        # Keep old name for backward compatibility  
        df['Child_Enrolment_Share'] = df['Child_Share']
    
    return df


def categorize_workload(workload: pd.Series) -> pd.Series:
    """
    Categorize service workload into levels for dashboard display.
    
    Categories:
    - Low: < 1.0 (fewer updates than new registrations)
    - Normal: 1.0 - 2.0 (balanced workload)
    - High: 2.0 - 5.0 (more updates than registrations)
    - Very High: > 5.0 (significantly more updates)
    
    Args:
        workload: Series of Service_Workload values
        
    Returns:
        Series with categorical workload levels
    """
    conditions = [
        workload < 1.0,
        (workload >= 1.0) & (workload < 2.0),
        (workload >= 2.0) & (workload < 5.0),
        workload >= 5.0
    ]
    choices = ['Low', 'Normal', 'High', 'Very High']
    
    return pd.Series(
        np.select(conditions, choices, default='Unknown'),
        index=workload.index
    )


# Keep old function for backward compatibility
def categorize_update_pressure(pressure_index: pd.Series) -> pd.Series:
    return categorize_workload(pressure_index)


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for dashboard KPIs.
    
    Args:
        df: DataFrame with computed indicators
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {}
    
    if 'Total_Enrolment' in df.columns:
        stats['total_enrolment'] = df['Total_Enrolment'].sum()
        stats['avg_monthly_enrolment'] = df['Total_Enrolment'].mean()
    
    if 'Total_Updates' in df.columns:
        stats['total_updates'] = df['Total_Updates'].sum()
        stats['avg_monthly_updates'] = df['Total_Updates'].mean()
    
    if 'Service_Workload' in df.columns:
        stats['avg_service_workload'] = df['Service_Workload'].mean()
        stats['max_service_workload'] = df['Service_Workload'].max()
        # Keep old keys for backward compatibility
        stats['avg_update_pressure'] = stats['avg_service_workload']
        stats['max_update_pressure'] = stats['max_service_workload']
    
    if 'Total_Demographic_Updates' in df.columns:
        stats['total_demographic_updates'] = df['Total_Demographic_Updates'].sum()
    
    if 'Total_Biometric_Updates' in df.columns:
        stats['total_biometric_updates'] = df['Total_Biometric_Updates'].sum()
    
    return stats


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
    
    from utils.preprocessing import load_and_preprocess_all_data, merge_all_datasets
    
    DATA_PATH = r"d:\COURSES\AADHAAR HACKATHON"
    
    enrolment, demographic, biometric = load_and_preprocess_all_data(
        DATA_PATH, aggregate_level='national'
    )
    
    merged = merge_all_datasets(enrolment, demographic, biometric)
    df_with_indicators = compute_all_indicators(merged)
    
    print("\nDataFrame with indicators:")
    print(df_with_indicators.head())
    
    print("\nSummary Statistics:")
    print(get_summary_statistics(df_with_indicators))
