# Slippage Data Directory

This directory contains all the essential data files needed for the PIS slippage analysis system.

## Files:

### Configuration Files:
- **config.yml** - Main configuration file containing simulation parameters, contamination settings, and sampling rules
- **compliance_table.csv** - Lookup table for compliance rates by country and material type

### Scenario Files:
- **pis_contaminate_scenarios.csv** - Defines contamination scenarios for testing various outbreak conditions

### Data Files:
- **fake_pis_data.csv** - Synthetic PIS consignment data for testing
- **synthetic_data.csv** - Generated synthetic consignment data
- **contaminate_examples.csv** - Example contamination patterns
- **sampling_examples.csv** - Example sampling configurations

## Usage:

The slippage_integrated.py script automatically looks for these files in this directory. All data files were moved from the original `hierarchal sampling/data` folder to provide a cleaner, more organized structure.

## File Sources:

These files were copied from `hierarchal sampling/data` and are essential for:
- Standard slippage analysis
- Scenario-based contamination testing  
- Hierarchical population analysis
- System verification and validation

## Notes:

- The default data directory is now `data_input` instead of `hierarchal sampling/data`
- All file paths in the main script have been updated accordingly
- This consolidation makes the analysis system more portable and organized
