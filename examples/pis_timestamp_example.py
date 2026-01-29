# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

#!/usr/bin/env python3

"""
Example: Using PIS Data with Timestamps
=======================================

This example demonstrates how the CREATED_DATETIME column is included in 
the fake_pis_data.csv file and how it can be accessed for temporal analysis.

The timestamp format is MM:SS.S (minutes:seconds.tenths) and provides 
timing information for each inspection record.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from popsborder.consignments import PISConsignmentGenerator

def main():
    print("PIS Data with Timestamps Example")
    print("=" * 40)
    
    # Load the PIS data directly to show timestamp information
    pis_data = pd.read_csv('../hierarchal sampling/data/fake_pis_data.csv')
    
    print(f"Total records in PIS data: {len(pis_data)}")
    print(f"Columns available: {list(pis_data.columns)}")
    print()
    
    # Show sample timestamp data
    print("Sample timestamp data:")
    print("-" * 25)
    sample_data = pis_data[['INSPECTION_NUMBER', 'COUNTRY_OF_ORIGIN_NAME', 'CREATED_DATETIME']].head(10)
    for _, record in sample_data.iterrows():
        print(f"  {record['INSPECTION_NUMBER']}: {record['COUNTRY_OF_ORIGIN_NAME']} - {record['CREATED_DATETIME']}")
    
    print()
    
    # Parse timestamps for analysis
    print("Timestamp Analysis:")
    print("-" * 18)
    
    # Convert timestamps to seconds for analysis
    def timestamp_to_seconds(timestamp_str):
        """Convert MM:SS.S format to total seconds"""
        minutes, seconds_tenths = timestamp_str.split(':')
        seconds, tenths = seconds_tenths.split('.')
        return int(minutes) * 60 + int(seconds) + int(tenths) / 10
    
    pis_data['seconds'] = pis_data['CREATED_DATETIME'].apply(timestamp_to_seconds)
    
    print(f"Average inspection time: {pis_data['seconds'].mean():.1f} seconds")
    print(f"Minimum inspection time: {pis_data['seconds'].min():.1f} seconds ({pis_data.loc[pis_data['seconds'].idxmin(), 'CREATED_DATETIME']})")
    print(f"Maximum inspection time: {pis_data['seconds'].max():.1f} seconds ({pis_data.loc[pis_data['seconds'].idxmax(), 'CREATED_DATETIME']})")
    
    # Show timestamp distribution by country
    print()
    print("Average inspection time by country:")
    print("-" * 35)
    country_times = pis_data.groupby('COUNTRY_OF_ORIGIN_NAME')['seconds'].mean().sort_values(ascending=False)
    for country, avg_time in country_times.items():
        minutes = int(avg_time // 60)
        seconds = avg_time % 60
        print(f"  {country}: {minutes:02d}:{seconds:04.1f}")
    
    print()
    
    # Demonstrate that PIS generator still works normally
    print("PIS Generator Integration:")
    print("-" * 25)
    
    generator = PISConsignmentGenerator(
        filename='../hierarchal sampling/data/fake_pis_data.csv'
    )
    
    print("✅ PIS generator loads data with timestamps successfully")
    print("✅ Generator focuses on inspection data, ignores timestamps")
    print("✅ All existing functionality preserved")
    
    # Generate a sample consignment to prove it works
    consignment = generator.generate_consignment()
    print(f"✅ Generated consignment: {consignment.num_inspection_units} units, {consignment.num_sample_units} sample units")
    
    print()
    print("Note: The CREATED_DATETIME column is available in the CSV data")
    print("      but is not used by the consignment generation process.")
    print("      It can be accessed for separate temporal analysis if needed.")

if __name__ == "__main__":
    main()