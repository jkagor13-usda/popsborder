# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

#!/usr/bin/env python3

"""
Example: Using PIS Data for Consignment Generation
==================================================

This example demonstrates how to use the fake_pis_data.csv file to generate
consignments with multiple material types per consignment, based on real
Plant Inspection System (PIS) records.

Key Features:
- Each INSPECTION_NUMBER represents a consignment
- Multiple rows with same INSPECTION_NUMBER become inspection units with different material types
- Actual quantities from PIS data are preserved (TOTAL_SAMPLING_UNITS, TOTAL_PLANT_QUANTITY)
- Hierarchical structure supports plant-level contamination modeling
- Includes CREATED_DATETIME timestamps for temporal analysis (MM:SS.S format)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from popsborder.consignments import PISConsignmentGenerator

def main():
    print("PIS Data Consignment Generation Example")
    print("=" * 50)
    
    # Create PIS consignment generator - fully data-driven!
    generator = PISConsignmentGenerator(
        filename='../hierarchal sampling/data/fake_pis_data.csv'
    )
    
    print(f"Total consignments available: {len(generator.consignment_groups)}")
    print()
    
    # Generate and display first few consignments
    for i in range(5):
        consignment = generator.generate_consignment()
        
        print(f"Consignment {i+1}:")
        print(f"  Origin: {consignment.origin}")
        print(f"  Port: {consignment.port}")
        print(f"  Pathway: {consignment.pathway}")
        print(f"  Material Types: {consignment.material_type}")
        print(f"  Inspection Units: {consignment.num_inspection_units}")
        print(f"  Total Sample Units: {consignment.num_sample_units}")
        print(f"  Total Plants: {consignment.num_plants}")
        
        if consignment.num_inspection_units > 1:
            print("  ** MULTI-MATERIAL CONSIGNMENT **")
            
        print("  Inspection Unit Details:")
        for j, iu in enumerate(consignment.inspection_units):
            print(f"    Unit {j+1}: {iu.material_type}")
            print(f"            Sample Units: {iu.num_sample_units}")
            print(f"            Plant Objects: {len(iu.included_unit_objects)}")
        
        print()
        
        # Stop if we found a multi-material consignment for detailed inspection
        if consignment.num_inspection_units > 1:
            print("Detailed Multi-Material Consignment Analysis:")
            print("-" * 45)
            
            total_sample_units_check = 0
            for j, iu in enumerate(consignment.inspection_units):
                print(f"  Inspection Unit {j+1} ({iu.material_type}):")
                print(f"    Sample Units Array Shape: {iu.included_units.shape}")
                print(f"    Sample Unit Objects: {len(iu.included_unit_objects)}")
                
                # Check first few sample unit objects
                for k, suo in enumerate(iu.included_unit_objects[:3]):
                    print(f"      SampleUnit {k+1}: {suo.num_plants} plants")
                    
                total_sample_units_check += iu.num_sample_units
                
            print(f"  Total verification: {total_sample_units_check} == {consignment.num_sample_units}")
            break
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()

