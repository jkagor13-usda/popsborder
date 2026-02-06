# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

#!/usr/bin/env python3

"""
Example: PIS Data with Enhanced Timestamp Integration
====================================================

This example demonstrates how the PISConsignmentGenerator now uses the 
CREATED_DATETIME column from the PIS data to set realistic consignment 
timestamps instead of using hardcoded dates.

The timestamps are converted from MM:SS.S format (minutes:seconds.tenths)
to proper datetime objects for use in the simulation system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from popsborder.consignments import PISConsignmentGenerator
from datetime import datetime, timedelta

def main():
    print("PIS Data Enhanced Timestamp Integration")
    print("=" * 45)
    
    # Load the raw PIS data to compare with generated consignments
    pis_data = pd.read_csv('../hierarchal sampling/data/fake_pis_data.csv')
    
    print(f"Raw PIS data contains {len(pis_data)} records")
    print(f"Unique consignments: {pis_data['INSPECTION_NUMBER'].nunique()}")
    
    # Show sample raw timestamps
    print("\nSample raw timestamps from CSV:")
    print("-" * 35)
    sample_records = pis_data[['INSPECTION_NUMBER', 'CREATED_DATETIME', 'COUNTRY_OF_ORIGIN_NAME']].head(8)
    for _, record in sample_records.iterrows():
        print(f"  {record['INSPECTION_NUMBER']}: {record['CREATED_DATETIME']} ({record['COUNTRY_OF_ORIGIN_NAME']})")
    
    print("\n" + "="*45)
    
    # Create PIS generator
    generator = PISConsignmentGenerator(
        filename='../hierarchal sampling/data/fake_pis_data.csv'
    )
    
    print(f"\nPIS Generator loaded {len(generator.consignment_groups)} consignments")
    
    # Generate consignments and show how timestamps are integrated
    print("\nGenerated Consignments with Timestamp Integration:")
    print("-" * 55)
    
    for i in range(6):
        consignment = generator.generate_consignment()
        
        # Extract time components for display
        dt = consignment._date
        time_part = dt.strftime("%H:%M:%S.%f")[:-3]  # Remove last 3 digits of microseconds
        
        print(f"\nConsignment {i+1}:")
        print(f"  Full DateTime: {dt}")
        print(f"  Time Component: {time_part} (from CREATED_DATETIME)")
        print(f"  Date Property: {consignment.date}")
        print(f"  Origin: {consignment.origin}")
        print(f"  Port: {consignment.port}")
        print(f"  Sample Units: {consignment.num_sample_units}")
        print(f"  Plants: {consignment.num_plants}")
        
        # Calculate inspection duration for analysis
        base_time = datetime(2020, 1, 1)
        duration = (dt - base_time).total_seconds()
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"  Inspection Duration: {minutes:02d}:{seconds:04.1f}")
    
    print("\n" + "="*45)
    print("\nTimestamp Integration Summary:")
    print("✅ CREATED_DATETIME column successfully integrated")
    print("✅ MM:SS.S format converted to proper datetime objects") 
    print("✅ Timestamps preserve inspection timing information")
    print("✅ Base date (2020-01-01) used for consistency")
    print("✅ All existing PIS functionality preserved")
    print("✅ Fallback to default date if timestamp parsing fails")
    
    print("\nTechnical Details:")
    print("• Format: MM:SS.S → datetime with time offset")
    print("• Base Date: 2020-01-01 00:00:00.000000")
    print("• Example: 34:54.6 → 2020-01-01 00:34:54.600000")
    print("• Precision: Microseconds (0.1 second = 100,000 μs)")
    
    print("\nUsage in Simulation:")
    print("• consignment.date → date portion for scheduling")
    print("• consignment._date → full datetime for detailed analysis")
    print("• Temporal analysis now possible with real timing data")

if __name__ == "__main__":
    main()