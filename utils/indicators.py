"""
Indicator calculation utilities for Aadhaar Intelligence Platform.

This module provides functions for calculating derived indicators
and metrics from Aadhaar enrollment and update data.
"""

import pandas as pd
import numpy as np
from typing import Union, List


def calculate_total_updates(df):
    """
    Calculate total updates by summing demographic and biometric updates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified dataframe with demographic and biometric update columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with total_updates column added
    """
    df = df.copy()
    
    # Identify demographic update columns
    demo_cols = [col for col in df.columns if 'demo_age' in col]
    bio_cols = [col for col in df.columns if 'bio_age' in col]
    
    # Calculate totals
    if demo_cols:
        df['total_demo_updates'] = df[demo_cols].sum(axis=1)
    else:
        df['total_demo_updates'] = 0
    
    if bio_cols:
        df['total_bio_updates'] = df[bio_cols].sum(axis=1)
    else:
        df['total_bio_updates'] = 0
    
    df['total_updates'] = df['total_demo_updates'] + df['total_bio_updates']
    
    print(f"✓ Calculated total updates")
    print(f"  - Average demographic updates: {df['total_demo_updates'].mean():.2f}")
    print(f"  - Average biometric updates: {df['total_bio_updates'].mean():.2f}")
    print(f"  - Average total updates: {df['total_updates'].mean():.2f}")
    
    return df


def calculate_total_enrolment(df):
    """
    Calculate total enrolment across all age groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with age group enrolment columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with total_enrolment column added
    """
    df = df.copy()
    
    # Identify age group columns
    age_cols = [col for col in df.columns if 'age_' in col and 'demo' not in col and 'bio' not in col]
    
    if age_cols:
        df['total_enrolment'] = df[age_cols].sum(axis=1)
    else:
        df['total_enrolment'] = 0
    
    print(f"✓ Calculated total enrolment")
    print(f"  - Average total enrolment: {df['total_enrolment'].mean():.2f}")
    
    return df


def calculate_update_pressure_index(df):
    """
    Calculate Update Pressure Index (UPI) = Total Updates / Total Enrolment
    
    Higher values indicate more update pressure relative to enrolment base.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with total_updates and total_enrolment columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with update_pressure_index column added
    """
    df = df.copy()
    
    # Ensure we have the required columns
    if 'total_updates' not in df.columns:
        df = calculate_total_updates(df)
    
    if 'total_enrolment' not in df.columns:
        df = calculate_total_enrolment(df)
    
    # Calculate UPI with safe division (avoid division by zero)
    df['update_pressure_index'] = df.apply(
        lambda row: row['total_updates'] / row['total_enrolment'] if row['total_enrolment'] > 0 else 0,
        axis=1
    )
    
    print(f"✓ Calculated Update Pressure Index")
    print(f"  - Mean UPI: {df['update_pressure_index'].mean():.4f}")
    print(f"  - Median UPI: {df['update_pressure_index'].median():.4f}")
    print(f"  - Max UPI: {df['update_pressure_index'].max():.4f}")
    
    return df



def calculate_age_transition_rate(df):
    """
    Calculate Age Transition Rate = bio_age_17_ / age_5_17.
    
    This indicator tracks the transition of individuals from child to adult category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with relevant age columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with age_transition_rate column added
    """
    df = df.copy()
    
    # Check if required columns exist
    if 'bio_age_17_' in df.columns and 'age_5_17' in df.columns:
        df['age_transition_rate'] = np.where(
            df['age_5_17'] > 0,
            df['bio_age_17_'] / df['age_5_17'],
            0
        )
        
        print(f"✓ Calculated Age Transition Rate")
        print(f"  - Mean transition rate: {df['age_transition_rate'].mean():.4f}")
    else:
        print("⚠ Age transition rate: Required columns not found")
        df['age_transition_rate'] = 0
    
    return df


def calculate_update_ratio(df):
    """
    Calculate demographic vs biometric update ratio.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with update columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with demo_bio_ratio column added
    """
    df = df.copy()
    
    if 'total_demo_updates' not in df.columns:
        df = calculate_total_updates(df)
    
    # Calculate ratio
    df['demo_bio_ratio'] = np.where(
        df['total_bio_updates'] > 0,
        df['total_demo_updates'] / df['total_bio_updates'],
        np.nan
    )
    
    print(f"✓ Calculated Demographic/Biometric Update Ratio")
    print(f"  - Mean ratio: {df['demo_bio_ratio'].mean():.4f}")
    
    return df


def calculate_growth_rates(df, groupby_cols=['state', 'district'], period='M'):
    """
    Calculate growth rates for enrolment and updates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time series data
    groupby_cols : list
        Columns to group by for growth calculation
    period : str
        Period for resampling ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with growth rate columns added
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if 'date' not in df.columns:
        print("⚠ Growth rates: Date column not found")
        return df
    
    df = df.sort_values(['state', 'district', 'date'])
    
    # Calculate period-over-period growth for each group
    for col in ['total_enrolment', 'total_updates']:
        if col in df.columns:
            df[f'{col}_growth'] = df.groupby(groupby_cols)[col].pct_change() * 100
    
    print(f"✓ Calculated growth rates")
    
    return df


def calculate_coverage_metrics(df, population_data=None):
    """
    Calculate coverage metrics if population data is available.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with enrolment data
    population_data : pd.DataFrame, optional
        Population data by state/district
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with coverage metrics
    """
    df = df.copy()
    
    if population_data is not None:
        # Merge with population data
        df = df.merge(
            population_data,
            on=['state', 'district'],
            how='left'
        )
        
        # Calculate coverage percentage
        if 'population' in df.columns:
            df['coverage_rate'] = (df['total_enrolment'] / df['population']) * 100
            print(f"✓ Calculated coverage rates")
            print(f"  - Mean coverage: {df['coverage_rate'].mean():.2f}%")
    else:
        print("⚠ Coverage metrics: Population data not provided")
    
    return df


def calculate_all_indicators(df, include_growth=True):
    """
    Calculate all derived indicators in one go.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    include_growth : bool
        Whether to calculate growth rates
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all indicators
    """
    print("=" * 60)
    print("CALCULATING DERIVED INDICATORS")
    print("=" * 60)
    
    df = df.copy()
    
    # Calculate base metrics
    df = calculate_total_enrolment(df)
    df = calculate_total_updates(df)
    
    # Calculate indicators
    df = calculate_update_pressure_index(df)
    df = calculate_age_transition_rate(df)
    df = calculate_update_ratio(df)
    
    # Calculate growth rates if requested
    if include_growth and 'date' in df.columns:
        df = calculate_growth_rates(df)
    
    print("\n" + "=" * 60)
    print("INDICATOR CALCULATION COMPLETE")
    print("=" * 60)
    
    return df


def get_top_districts(df, metric='update_pressure_index', n=10, ascending=False):
    """
    Get top N districts by a specific metric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with calculated metrics
    metric : str
        Metric to rank by
    n : int
        Number of top districts to return
    ascending : bool
        If True, return bottom N instead
        
    Returns:
    --------
    pd.DataFrame
        Top N districts
    """
    if metric not in df.columns:
        print(f"Error: Metric '{metric}' not found in dataframe")
        return None
    
    # Group by district and calculate mean
    district_metrics = df.groupby(['state', 'district'])[metric].mean().reset_index()
    
    # Sort and get top N
    top_districts = district_metrics.sort_values(metric, ascending=ascending).head(n)
    
    return top_districts


def create_indicator_summary(df):
    """
    Create a summary table of all indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with calculated indicators
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    indicator_cols = [
        'total_enrolment', 'total_updates', 'total_demo_updates', 
        'total_bio_updates', 'update_pressure_index', 'age_transition_rate',
        'demo_bio_ratio'
    ]
    
    # Filter to existing columns
    existing_indicators = [col for col in indicator_cols if col in df.columns]
    
    # Create summary
    summary = df[existing_indicators].describe().T
    summary['median'] = df[existing_indicators].median()
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Indicators module loaded successfully")
    print("Available functions:")
    print("  - calculate_total_updates()")
    print("  - calculate_update_pressure_index()")
    print("  - calculate_age_transition_rate()")
    print("  - calculate_all_indicators()")
    print("  - get_top_districts()")
    print("  - create_indicator_summary()")
