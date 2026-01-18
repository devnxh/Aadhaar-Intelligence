"""
Aadhaar Data Preprocessing Utilities

This module provides functions for loading, cleaning, and aggregating
Aadhaar enrolment, demographic, and biometric update datasets.

All data processing maintains aggregated, anonymised outputs only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csv_files_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a specified folder.
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        Concatenated DataFrame of all CSVs
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_enrolment_data(data_path: str) -> pd.DataFrame:
    """
    Load Aadhaar enrolment data from the enrolment folder.
    
    Expected columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
    
    Args:
        data_path: Base path to api_data folders
        
    Returns:
        DataFrame with enrolment data
    """
    folder = Path(data_path) / "api_data_aadhar_enrolment"
    df = load_csv_files_from_folder(folder)
    
    # Validate expected columns
    expected_cols = ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater']
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in enrolment data: {missing}")
    
    return df


def load_demographic_data(data_path: str) -> pd.DataFrame:
    """
    Load Aadhaar demographic update data.
    
    Expected columns: date, state, district, pincode, demo_age_5_17, demo_age_17_
    
    Args:
        data_path: Base path to api_data folders
        
    Returns:
        DataFrame with demographic update data
    """
    folder = Path(data_path) / "api_data_aadhar_demographic"
    df = load_csv_files_from_folder(folder)
    
    expected_cols = ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_']
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in demographic data: {missing}")
    
    return df


def load_biometric_data(data_path: str) -> pd.DataFrame:
    """
    Load Aadhaar biometric update data.
    
    Expected columns: date, state, district, pincode, bio_age_5_17, bio_age_17_
    
    Args:
        data_path: Base path to api_data folders
        
    Returns:
        DataFrame with biometric update data
    """
    folder = Path(data_path) / "api_data_aadhar_biometric"
    df = load_csv_files_from_folder(folder)
    
    expected_cols = ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in biometric data: {missing}")
    
    return df


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def parse_dates(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Parse date column to datetime format.
    
    Handles multiple date formats (DD-MM-YYYY, YYYY-MM-DD, etc.)
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with parsed dates
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df[date_column].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} rows with invalid dates removed")
        df = df.dropna(subset=[date_column])
    
    return df


def clean_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns: convert to numeric, fill NaN with 0, ensure non-negative.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of column names to clean
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)  # Ensure non-negative
    
    return df


def clean_enrolment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean enrolment dataset."""
    df = parse_dates(df)
    numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    df = clean_numeric_columns(df, numeric_cols)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def clean_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean demographic update dataset."""
    df = parse_dates(df)
    numeric_cols = ['demo_age_5_17', 'demo_age_17_']
    df = clean_numeric_columns(df, numeric_cols)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def clean_biometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean biometric update dataset."""
    df = parse_dates(df)
    numeric_cols = ['bio_age_5_17', 'bio_age_17_']
    df = clean_numeric_columns(df, numeric_cols)
    df = df.sort_values('date').reset_index(drop=True)
    return df


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_monthly(
    df: pd.DataFrame,
    date_column: str = 'date',
    group_columns: Optional[List[str]] = None,
    sum_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate data to monthly frequency.
    
    Monthly aggregation reduces volatility and aligns with administrative planning cycles.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        group_columns: Additional columns to group by (e.g., ['state'])
        sum_columns: Columns to sum
        
    Returns:
        Monthly aggregated DataFrame
    """
    df = df.copy()
    
    # Create year-month column
    df['year_month'] = df[date_column].dt.to_period('M')
    
    # Define grouping
    if group_columns:
        group_by = ['year_month'] + group_columns
    else:
        group_by = ['year_month']
    
    # Define sum columns if not provided
    if sum_columns is None:
        sum_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Aggregate
    agg_dict = {col: 'sum' for col in sum_columns if col in df.columns}
    monthly_df = df.groupby(group_by, as_index=False).agg(agg_dict)
    
    # Convert period back to timestamp (first day of month)
    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
    monthly_df = monthly_df.drop(columns=['year_month'])
    
    # Reorder columns
    cols = ['date'] + [c for c in monthly_df.columns if c != 'date']
    monthly_df = monthly_df[cols]
    
    return monthly_df


def create_national_aggregate(
    df: pd.DataFrame,
    date_column: str = 'date',
    sum_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create national-level aggregates (all data combined by date).
    
    Args:
        df: Input DataFrame (already monthly aggregated)
        date_column: Date column name
        sum_columns: Columns to sum
        
    Returns:
        National aggregate DataFrame
    """
    if sum_columns is None:
        sum_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    agg_dict = {col: 'sum' for col in sum_columns if col in df.columns}
    national_df = df.groupby(date_column, as_index=False).agg(agg_dict)
    national_df['level'] = 'National'
    
    return national_df


def create_state_aggregate(
    df: pd.DataFrame,
    date_column: str = 'date',
    state_column: str = 'state',
    sum_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create state-level aggregates.
    
    Args:
        df: Input DataFrame (already monthly aggregated)
        date_column: Date column name
        state_column: State column name
        sum_columns: Columns to sum
        
    Returns:
        State aggregate DataFrame
    """
    if sum_columns is None:
        sum_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    agg_dict = {col: 'sum' for col in sum_columns if col in df.columns}
    state_df = df.groupby([date_column, state_column], as_index=False).agg(agg_dict)
    state_df['level'] = 'State'
    
    return state_df


def fill_missing_months(
    df: pd.DataFrame,
    date_column: str = 'date',
    group_column: Optional[str] = None,
    fill_value: float = 0
) -> pd.DataFrame:
    """
    Fill missing months with specified value (default 0).
    
    Args:
        df: Input DataFrame
        date_column: Date column name
        group_column: If provided, fill missing months for each group
        fill_value: Value to fill missing entries
        
    Returns:
        DataFrame with missing months filled
    """
    df = df.copy()
    
    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    all_months = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    if group_column and group_column in df.columns:
        # Create complete index for each group
        groups = df[group_column].unique()
        complete_idx = pd.MultiIndex.from_product(
            [all_months, groups],
            names=[date_column, group_column]
        )
        complete_df = pd.DataFrame(index=complete_idx).reset_index()
        
        # Merge with original data
        df = complete_df.merge(df, on=[date_column, group_column], how='left')
    else:
        # Single time series
        complete_df = pd.DataFrame({date_column: all_months})
        df = complete_df.merge(df, on=date_column, how='left')
    
    # Fill numeric columns with specified value
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_value)
    
    return df


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def load_and_preprocess_all_data(
    data_path: str,
    aggregate_level: str = 'national'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline for all three datasets.
    
    Args:
        data_path: Base path to api_data folders
        aggregate_level: 'national', 'state', or 'both'
        
    Returns:
        Tuple of (enrolment_df, demographic_df, biometric_df)
    """
    print("Loading and preprocessing Aadhaar data...")
    
    # Load raw data
    print("  Loading enrolment data...")
    enrolment_raw = load_enrolment_data(data_path)
    print(f"    Loaded {len(enrolment_raw):,} records")
    
    print("  Loading demographic data...")
    demographic_raw = load_demographic_data(data_path)
    print(f"    Loaded {len(demographic_raw):,} records")
    
    print("  Loading biometric data...")
    biometric_raw = load_biometric_data(data_path)
    print(f"    Loaded {len(biometric_raw):,} records")
    
    # Clean data
    print("  Cleaning data...")
    enrolment_clean = clean_enrolment_data(enrolment_raw)
    demographic_clean = clean_demographic_data(demographic_raw)
    biometric_clean = clean_biometric_data(biometric_raw)
    
    # Aggregate to monthly
    print("  Aggregating to monthly frequency...")
    
    if aggregate_level == 'national':
        enrolment_monthly = aggregate_to_monthly(
            enrolment_clean,
            sum_columns=['age_0_5', 'age_5_17', 'age_18_greater']
        )
        demographic_monthly = aggregate_to_monthly(
            demographic_clean,
            sum_columns=['demo_age_5_17', 'demo_age_17_']
        )
        biometric_monthly = aggregate_to_monthly(
            biometric_clean,
            sum_columns=['bio_age_5_17', 'bio_age_17_']
        )
    elif aggregate_level == 'state':
        enrolment_monthly = aggregate_to_monthly(
            enrolment_clean,
            group_columns=['state'],
            sum_columns=['age_0_5', 'age_5_17', 'age_18_greater']
        )
        demographic_monthly = aggregate_to_monthly(
            demographic_clean,
            group_columns=['state'],
            sum_columns=['demo_age_5_17', 'demo_age_17_']
        )
        biometric_monthly = aggregate_to_monthly(
            biometric_clean,
            group_columns=['state'],
            sum_columns=['bio_age_5_17', 'bio_age_17_']
        )
    
    # Fill missing months
    print("  Filling missing months with zeros...")
    if aggregate_level == 'state':
        enrolment_monthly = fill_missing_months(enrolment_monthly, group_column='state')
        demographic_monthly = fill_missing_months(demographic_monthly, group_column='state')
        biometric_monthly = fill_missing_months(biometric_monthly, group_column='state')
    else:
        enrolment_monthly = fill_missing_months(enrolment_monthly)
        demographic_monthly = fill_missing_months(demographic_monthly)
        biometric_monthly = fill_missing_months(biometric_monthly)
    
    print("  Preprocessing complete!")
    print(f"    Enrolment: {len(enrolment_monthly)} monthly records")
    print(f"    Demographic: {len(demographic_monthly)} monthly records")
    print(f"    Biometric: {len(biometric_monthly)} monthly records")
    
    return enrolment_monthly, demographic_monthly, biometric_monthly


def merge_all_datasets(
    enrolment_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    on_columns: List[str] = ['date']
) -> pd.DataFrame:
    """
    Merge all three datasets into a unified DataFrame.
    
    Args:
        enrolment_df: Enrolment data
        demographic_df: Demographic update data
        biometric_df: Biometric update data
        on_columns: Columns to merge on
        
    Returns:
        Merged DataFrame
    """
    merged = enrolment_df.merge(demographic_df, on=on_columns, how='outer')
    merged = merged.merge(biometric_df, on=on_columns, how='outer')
    
    # Fill any NaN from outer join with 0
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    return merged.sort_values('date').reset_index(drop=True)


if __name__ == "__main__":
    # Test the preprocessing
    DATA_PATH = r"d:\COURSES\AADHAAR HACKATHON"
    
    enrolment, demographic, biometric = load_and_preprocess_all_data(
        DATA_PATH, aggregate_level='national'
    )
    
    print("\nEnrolment sample:")
    print(enrolment.head())
    
    print("\nDemographic sample:")
    print(demographic.head())
    
    print("\nBiometric sample:")
    print(biometric.head())
