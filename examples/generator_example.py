#!/usr/bin/env python3

"""
Example: Synthetic Consignment Data Generation
==============================================

This example demonstrates how to use the SyntheticConsignmentDataGenerator
to create realistic synthetic consignment data for testing and simulation.

The generator supports multiple sampling methods and can work with or without
input data files for training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from popsborder.generator import SyntheticConsignmentDataGenerator

def main():
    print("Synthetic Consignment Data Generation Example")
    print("=" * 50)
    
    # Example 1: Basic synthetic data generation
    print("\n1. Basic Synthetic Data Generation:")
    print("-" * 35)
    
    generator = SyntheticConsignmentDataGenerator()
    
    # Generate a single record
    single_record = generator.generate_consignment_record()
    print("Single consignment record:")
    for key, value in single_record.items():
        print(f"  {key}: {value}")
    
    # Generate a small dataset
    print(f"\nGenerating dataset of 10 records...")
    dataset = generator.generate_dataset(10, use_input_data=False)
    print(f"Generated dataset shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")
    
    # Generate statistics
    stats = generator.generate_statistics(dataset)
    print(f"\nDataset Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Numeric columns: {len(stats['numeric_columns'])}")
    print(f"  Categorical columns: {len(stats['categorical_columns'])}")
    
    # Example 2: Advanced sampling with input data
    print(f"\n2. Advanced Sampling with PIS Data:")
    print("-" * 35)
    
    try:
        # Initialize with PIS data
        advanced_generator = SyntheticConsignmentDataGenerator(
            input_data_file='../hierarchal sampling/data/fake_pis_data.csv'
        )
        
        if advanced_generator.input_data is not None:
            print(f"Loaded input data with {len(advanced_generator.input_data)} records")
        else:
            print("No input data loaded")
        
        # Test different sampling methods
        sampling_methods = ['naive', 'sequential', 'gmm', 'gaussian_copula']
        
        for method in sampling_methods:
            print(f"\nTesting {method} sampling:")
            synthetic_data = advanced_generator.generate_from_input_data(
                n_samples=5,
                sampling_method=method
            )
            print(f"  Generated shape: {synthetic_data.shape}")
            
            # Show sample of generated data
            if not synthetic_data.empty:
                print("  Sample record:")
                first_record = synthetic_data.iloc[0]
                for col in ['COUNTRY_OF_ORIGIN_NAME', 'PATHWAY', 'PROPAGATIVE_MATERIAL_TYPE', 'CREATED_DATETIME']:
                    if col in first_record:
                        print(f"    {col}: {first_record[col]}")
        
        # Calculate quality metrics
        print(f"\n3. Quality Assessment:")
        print("-" * 20)
        
        if advanced_generator.input_data is not None:
            original_sample = advanced_generator.input_data.head(50)
        else:
            original_sample = None
        synthetic_sample = advanced_generator.generate_from_input_data(
            n_samples=50, 
            sampling_method='gaussian_copula'
        )
        
        if original_sample is not None:
            metrics = advanced_generator.calculate_quality_metrics(
                original_sample, 
                synthetic_sample
            )
        else:
            metrics = {}
        
        print("Quality metrics (lower is better):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        # Save synthetic data
        print(f"\n4. Saving Synthetic Data:")
        print("-" * 25)
        
        output_file = "synthetic_consignments_example.csv"
        advanced_generator.save_to_csv(synthetic_sample, output_file)
        print(f"Synthetic data saved to: {output_file}")
        
    except Exception as e:
        print(f"Advanced sampling example failed: {e}")
        print("This is normal if the PIS data file is not available.")
    
    print(f"\nExample completed successfully!")
    print("The generator is now consistent with popsborder package style.")

if __name__ == "__main__":
    main()