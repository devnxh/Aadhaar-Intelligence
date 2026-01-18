"""
Data preprocessing utilities for Aadhaar Intelligence Platform.

This module provides functions for loading, cleaning, standardizing,
and aggregating Aadhaar enrollment and update datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_datasets(data_dir='data'):
    """
    Load all three Aadhaar datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
        
    Returns:
    --------
    tuple
        (enrolment_df, demographic_df, biometric_df)
    """
    data_path = Path(data_dir)
    
    try:
        enrolment_df = pd.read_csv(data_path / 'enrolment.csv')
        demographic_df = pd.read_csv(data_path / 'demographic_updates.csv')
        biometric_df = pd.read_csv(data_path / 'biometric_updates.csv')
        
        print("✓ Successfully loaded all datasets")
        print(f"  - Enrolment records: {len(enrolment_df):,}")
        print(f"  - Demographic update records: {len(demographic_df):,}")
        print(f"  - Biometric update records: {len(biometric_df):,}")
        
        return enrolment_df, demographic_df, biometric_df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in '{data_dir}/' directory")
        print(f"Please ensure the following files exist:")
        print("  - enrolment.csv")
        print("  - demographic_updates.csv")
        print("  - biometric_updates.csv")
        raise e


def standardize_columns(df, dataset_type):
    """
    Standardize column names and data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_type : str
        Type of dataset ('enrolment', 'demographic', 'biometric')
        
    Returns:
    --------
    pd.DataFrame
        Standardized dataframe
    """
    df = df.copy()
    
    # Standardize column names (lowercase, strip spaces)
    df.columns = df.columns.str.lower().str.strip()
    
    # Standardize date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Standardize geographic columns
    for col in ['state', 'district']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    
    # Ensure numeric columns are numeric
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        if col not in ['date', 'state', 'district']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✓ Standardized {dataset_type} dataset columns")
    
    return df


def handle_missing_values(df, dataset_type):
    """
    Handle missing values using logical strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_type : str
        Type of dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    
    missing_before = df.isnull().sum().sum()
    
    # Fill numeric columns with 0 (representing no activity)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # For geographic columns, keep as is or drop rows if critical
    if df['state'].isnull().any() or df['district'].isnull().any():
        print(f"  Warning: Found {df['state'].isnull().sum()} missing states")
        print(f"  Warning: Found {df['district'].isnull().sum()} missing districts")
        df = df.dropna(subset=['state', 'district'])
    
    # Drop rows with missing dates
    if 'date' in df.columns:
        df = df.dropna(subset=['date'])
    
    missing_after = df.isnull().sum().sum()
    
    print(f"✓ Handled missing values in {dataset_type} dataset")
    print(f"  - Missing values before: {missing_before}")
    print(f"  - Missing values after: {missing_after}")
    
    return df


def aggregate_data(df, group_by=['date', 'state', 'district']):
    """
    Aggregate data by specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_by : list
        Columns to group by
        
    Returns:
    --------
    pd.DataFrame
        Aggregated dataframe
    """
    # Identify numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove grouping columns from numeric columns
    numeric_cols = [col for col in numeric_cols if col not in group_by]
    
    if not numeric_cols:
        return df
    
    # Aggregate by summing numeric columns
    df_agg = df.groupby(group_by)[numeric_cols].sum().reset_index()
    
    print(f"✓ Aggregated data by {group_by}")
    print(f"  - Records before aggregation: {len(df):,}")
    print(f"  - Records after aggregation: {len(df_agg):,}")
    
    return df_agg


def create_unified_table(enrolment_df, demographic_df, biometric_df):
    """
    Create a unified analytical table from all datasets.
    
    Parameters:
    -----------
    enrolment_df : pd.DataFrame
        Enrolment dataset
    demographic_df : pd.DataFrame
        Demographic update dataset
    biometric_df : pd.DataFrame
        Biometric update dataset
        
    Returns:
    --------
    pd.DataFrame
        Unified dataframe
    """
    # Merge datasets on common keys
    merge_keys = ['date', 'state', 'district', 'pincode']
    
    # Start with enrolment as base
    unified_df = enrolment_df.copy()
    
    # Merge demographic updates
    unified_df = unified_df.merge(
        demographic_df,
        on=merge_keys,
        how='outer',
        suffixes=('', '_demo')
    )
    
    # Merge biometric updates
    unified_df = unified_df.merge(
        biometric_df,
        on=merge_keys,
        how='outer',
        suffixes=('', '_bio')
    )
    
    # Fill NaN values with 0 for numeric columns
    numeric_cols = unified_df.select_dtypes(include=[np.number]).columns
    unified_df[numeric_cols] = unified_df[numeric_cols].fillna(0)
    
    print("✓ Created unified analytical table")
    print(f"  - Total records: {len(unified_df):,}")
    print(f"  - Columns: {len(unified_df.columns)}")
    print(f"  - Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")
    
    return unified_df


def get_data_summary(df, name="Dataset"):
    """
    Generate a comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    name : str
        Name of the dataset
        
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'date_range': None,
        'states': None,
        'districts': None,
        'missing_values': df.isnull().sum().sum()
    }
    
    if 'date' in df.columns:
        summary['date_range'] = (df['date'].min(), df['date'].max())
    
    if 'state' in df.columns:
        summary['states'] = df['state'].nunique()
    
    if 'district' in df.columns:
        summary['districts'] = df['district'].nunique()
    
    return summary


def save_cleaned_data(df, filename, output_dir='data/processed'):
    """
    Save cleaned dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    filename : str
        Output filename
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    
    print(f"✓ Saved cleaned data to {filepath}")


def preprocess_all_datasets(data_dir='data'):
    """
    Complete preprocessing pipeline for all datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
        
    Returns:
    --------
    tuple
        (enrolment_clean, demographic_clean, biometric_clean, unified_df)
    """
    print("=" * 60)
    print("AADHAAR DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load datasets
    print("\n[1/5] Loading datasets...")
    enrolment_df, demographic_df, biometric_df = load_datasets(data_dir)
    
    # Standardize columns
    print("\n[2/5] Standardizing columns...")
    enrolment_df = standardize_columns(enrolment_df, 'enrolment')
    demographic_df = standardize_columns(demographic_df, 'demographic')
    biometric_df = standardize_columns(biometric_df, 'biometric')
    
    # Handle missing values
    print("\n[3/5] Handling missing values...")
    enrolment_df = handle_missing_values(enrolment_df, 'enrolment')
    demographic_df = handle_missing_values(demographic_df, 'demographic')
    biometric_df = handle_missing_values(biometric_df, 'biometric')
    
    # Aggregate data
    print("\n[4/5] Aggregating data...")
    enrolment_df = aggregate_data(enrolment_df)
    demographic_df = aggregate_data(demographic_df)
    biometric_df = aggregate_data(biometric_df)
    
    # Create unified table
    print("\n[5/5] Creating unified analytical table...")
    unified_df = create_unified_table(enrolment_df, demographic_df, biometric_df)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return enrolment_df, demographic_df, biometric_df, unified_df


if __name__ == "__main__":
    # Example usage
    enrolment, demographic, biometric, unified = preprocess_all_datasets()
    
    # Print summaries
    print("\nDataset Summaries:")
    for df, name in [(enrolment, "Enrolment"), 
                      (demographic, "Demographic"), 
                      (biometric, "Biometric"),
                      (unified, "Unified")]:
        summary = get_data_summary(df, name)
        print(f"\n{name}:")
        for key, value in summary.items():
            if key != 'name':
                print(f"  {key}: {value}")
