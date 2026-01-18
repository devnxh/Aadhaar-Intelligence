"""
Sample Data Generator for Aadhaar Intelligence Platform

This script generates synthetic sample data for testing and demonstration purposes.
The data mimics the structure of real Aadhaar datasets but contains no real information.

Usage:
    python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_STATES = 10
NUM_DISTRICTS_PER_STATE = 5
NUM_DAYS = 365
START_DATE = datetime(2023, 1, 1)

# Indian state names (sample)
STATES = [
    'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Gujarat',
    'Rajasthan', 'West Bengal', 'Madhya Pradesh', 'Telangana', 'Bihar'
]

# Sample district names
DISTRICT_PREFIXES = ['North', 'South', 'East', 'West', 'Central']
DISTRICT_SUFFIXES = ['District', 'Zone', 'Region', 'Area', 'Division']


def generate_districts(state):
    """Generate district names for a state"""
    districts = []
    for i in range(NUM_DISTRICTS_PER_STATE):
        prefix = random.choice(DISTRICT_PREFIXES)
        suffix = random.choice(DISTRICT_SUFFIXES)
        districts.append(f"{prefix} {state} {suffix}")
    return districts


def generate_enrolment_data():
    """Generate synthetic enrolment dataset"""
    print("Generating enrolment data...")
    
    data = []
    
    for state in STATES[:NUM_STATES]:
        districts = generate_districts(state)
        
        for district in districts:
            # Generate 2-3 pincodes per district
            num_pincodes = random.randint(2, 3)
            
            for _ in range(num_pincodes):
                pincode = random.randint(100000, 999999)
                
                # Generate data for each day
                for day in range(NUM_DAYS):
                    date = START_DATE + timedelta(days=day)
                    
                    # Generate age group data with realistic patterns
                    # Add seasonal variation and trends
                    trend = day / NUM_DAYS * 0.3
                    seasonal = np.sin(2 * np.pi * day / 365) * 0.2
                    
                    base_multiplier = 1 + trend + seasonal
                    
                    age_0_5 = max(0, int(np.random.poisson(50) * base_multiplier))
                    age_5_17 = max(0, int(np.random.poisson(80) * base_multiplier))
                    age_18_greater = max(0, int(np.random.poisson(150) * base_multiplier))
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'state': state,
                        'district': district,
                        'pincode': pincode,
                        'age_0_5': age_0_5,
                        'age_5_17': age_5_17,
                        'age_18_greater': age_18_greater
                    })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} enrolment records")
    return df


def generate_demographic_update_data():
    """Generate synthetic demographic update dataset"""
    print("Generating demographic update data...")
    
    data = []
    
    for state in STATES[:NUM_STATES]:
        districts = generate_districts(state)
        
        for district in districts:
            num_pincodes = random.randint(2, 3)
            
            for _ in range(num_pincodes):
                pincode = random.randint(100000, 999999)
                
                for day in range(NUM_DAYS):
                    date = START_DATE + timedelta(days=day)
                    
                    # Updates are generally lower than enrolments
                    trend = day / NUM_DAYS * 0.2
                    seasonal = np.sin(2 * np.pi * day / 365) * 0.15
                    
                    base_multiplier = 1 + trend + seasonal
                    
                    demo_age_5_17 = max(0, int(np.random.poisson(20) * base_multiplier))
                    demo_age_17_ = max(0, int(np.random.poisson(40) * base_multiplier))
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'state': state,
                        'district': district,
                        'pincode': pincode,
                        'demo_age_5_17': demo_age_5_17,
                        'demo_age_17_': demo_age_17_
                    })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} demographic update records")
    return df


def generate_biometric_update_data():
    """Generate synthetic biometric update dataset"""
    print("Generating biometric update data...")
    
    data = []
    
    for state in STATES[:NUM_STATES]:
        districts = generate_districts(state)
        
        for district in districts:
            num_pincodes = random.randint(2, 3)
            
            for _ in range(num_pincodes):
                pincode = random.randint(100000, 999999)
                
                for day in range(NUM_DAYS):
                    date = START_DATE + timedelta(days=day)
                    
                    # Biometric updates similar to demographic
                    trend = day / NUM_DAYS * 0.25
                    seasonal = np.sin(2 * np.pi * day / 365) * 0.1
                    
                    base_multiplier = 1 + trend + seasonal
                    
                    bio_age_5_17 = max(0, int(np.random.poisson(15) * base_multiplier))
                    bio_age_17_ = max(0, int(np.random.poisson(35) * base_multiplier))
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'state': state,
                        'district': district,
                        'pincode': pincode,
                        'bio_age_5_17': bio_age_5_17,
                        'bio_age_17_': bio_age_17_
                    })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} biometric update records")
    return df


def add_anomalies(df, col_names, anomaly_rate=0.001):
    """Add some anomalous values to make anomaly detection interesting"""
    num_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        for col in col_names:
            if col in df.columns:
                # Make value 5-10x larger than normal
                df.loc[idx, col] = df.loc[idx, col] * np.random.uniform(5, 10)
    
    print(f"✓ Added {num_anomalies} anomalies")
    return df


def main():
    """Main function to generate all sample datasets"""
    print("=" * 60)
    print("AADHAAR SAMPLE DATA GENERATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  States: {NUM_STATES}")
    print(f"  Districts per state: {NUM_DISTRICTS_PER_STATE}")
    print(f"  Days of data: {NUM_DAYS}")
    print(f"  Start date: {START_DATE.strftime('%Y-%m-%d')}")
    print()
    
    # Generate datasets
    enrolment_df = generate_enrolment_data()
    demographic_df = generate_demographic_update_data()
    biometric_df = generate_biometric_update_data()
    
    # Add some anomalies
    print("\nAdding anomalies...")
    enrolment_df = add_anomalies(enrolment_df, ['age_0_5', 'age_5_17', 'age_18_greater'])
    demographic_df = add_anomalies(demographic_df, ['demo_age_5_17', 'demo_age_17_'])
    biometric_df = add_anomalies(biometric_df, ['bio_age_5_17', 'bio_age_17_'])
    
    # Save to CSV
    print("\nSaving datasets...")
    
    # Create data directory if it doesn't exist
    from pathlib import Path
    data_dir = Path('../data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    enrolment_df.to_csv(data_dir / 'enrolment.csv', index=False)
    demographic_df.to_csv(data_dir / 'demographic_updates.csv', index=False)
    biometric_df.to_csv(data_dir / 'biometric_updates.csv', index=False)
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  ✓ data/enrolment.csv")
    print("  ✓ data/demographic_updates.csv")
    print("  ✓ data/biometric_updates.csv")
    print("\nYou can now run the notebooks and dashboard!")


if __name__ == "__main__":
    main()
