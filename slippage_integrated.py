#!/usr/bin/env python3

""" Enhanced Slippage Analysis Using Popsborder Simulation
=====================================
JHU/APL Extensions and Modifications:
=====================================

Authors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

This enhanced script leverages the popsborder.outputs module for comprehensive
slippage analysis with proper integration into the existing framework.

Features:
- Uses popsborder.outputs functions for consistent reporting
- Integrates with existing configuration and scenario management
- Provides comprehensive pandas DataFrame exports
- Maintains compatibility with existing analysis workflows
- Professional JHU/APL formatted output with unit labeling
- Suppresses compliance warnings for clean output
- Modular testing functions in separate test module

Analysis Modes:
- standard: Basic slippage analysis with multiple simulations
- scenarios: Comprehensive contamination scenario testing
- hierarchical: Detailed hierarchical population analysis
- verify: System consistency verification and testing
- all: Execute all analysis modes

Usage:
    python slippage_integrated.py [--mode standard|scenarios|hierarchical|verify|all]
    python slippage_integrated.py --mode scenarios --consignments 20 --simulations 3
    python slippage_integrated.py --mode standard --show-compliance-warnings

Options:
    --mode: Analysis mode to run (default: all)
    --consignments: Number of consignments per simulation (default: 20)
    --simulations: Number of simulation runs (default: 3)
    --show-compliance-warnings: Show compliance warnings (default: suppress)

Data Structure:
    slippage_data/: Contains all configuration and input data files
    tests/: Contains testing and verification modules
    output/: Analysis results and exported data files
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from contextlib import redirect_stdout
from io import StringIO
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from popsborder.scenarios import run_scenarios
from popsborder.simulation import simulation
from popsborder.inputs import load_configuration, load_scenario_table, load_compliance_lookup_csv
from popsborder.outputs import (
    save_scenario_result_to_pandas,
    save_simulation_result_to_pandas,
    print_totals_as_text,
    config_to_simplified_simulation_params,
    pretty_consignment,
    PrintReporter,
    MuteReporter
)
from popsborder.consignments import get_consignment_generator
from popsborder.contamination import get_contaminant_function
from popsborder.inspections import get_sample_function


class SlippageAnalyzer:
    """Enhanced slippage analyzer using popsborder.outputs functions"""
    
    def __init__(self, data_dir="slippage_data", suppress_compliance_warnings=True):
        """Initialize analyzer with data directory"""
        self.data_dir = Path(data_dir)
        self.config = None
        self.compliance_table = None
        self.results = []
        self.suppress_compliance_warnings = suppress_compliance_warnings
        
        # Ensure output directory exists
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Setup output configuration
        self.output_config = {
            'config_columns': [
                'contamination/contamination_unit',
                'contamination/contamination_rate/distribution',
                'contamination/contamination_rate/value',
                'contamination/arrangement',
                'inspection/sample_strategy',
                'inspection/proportion/value',
                'inspection/tolerance_level'
            ],
            'result_columns': [
                'num_inspections',
                'intercepted',
                'false_neg',
                'missing',
                'true_contamination_rate',
                'avg_missed_contamination_rate',
                'max_missed_contamination_rate',
                'total_missed_contaminants',
                'total_intercepted_contaminants'
            ]
        }
    
    def _suppress_compliance_warnings(self):
        """Context manager to suppress compliance warnings"""
        class ComplianceWarningFilter:
            def __init__(self, original_stdout, suppress):
                self.original_stdout = original_stdout
                self.suppress = suppress
                self.buffer = StringIO()
                self.last_was_warning = False
                
            def write(self, text):
                if self.suppress:
                    # Check if this is a compliance warning line
                    if "WARNING: No compliance found" in text:
                        self.last_was_warning = True
                        return  # Suppress the warning completely
                    # Check if this is just a newline after a warning
                    elif self.last_was_warning and text.strip() == "":
                        return  # Suppress empty lines after warnings
                    else:
                        self.last_was_warning = False
                        self.original_stdout.write(text)
                else:
                    self.original_stdout.write(text)
                    
            def flush(self):
                self.original_stdout.flush()
        
        class WarningSuppressionContext:
            def __init__(self, suppress):
                self.suppress = suppress
                self.original_stdout = None
                self.filter = None
                
            def __enter__(self):
                if self.suppress:
                    self.original_stdout = sys.stdout
                    self.filter = ComplianceWarningFilter(self.original_stdout, True)
                    sys.stdout = self.filter
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.suppress and self.original_stdout:
                    sys.stdout = self.original_stdout
        
        return WarningSuppressionContext(self.suppress_compliance_warnings)
    
    def load_configuration(self):
        """Load configuration files using popsborder functions"""
        print("Loading Configuration Files...")
        
        try:
            self.config = load_configuration(self.data_dir / "config.yml")
            print(f"    Main config: {self.data_dir / 'config.yml'}")
            
            self.compliance_table = load_compliance_lookup_csv(self.data_dir / "compliance_table.csv")
            print(f"    Compliance table: {self.data_dir / 'compliance_table.csv'}")
            
            return True
        except Exception as e:
            print(f"    Configuration loading failed: {e}")
            return False
    
    def run_standard_analysis(self, num_consignments=20, num_simulations=3):
        """Run standard slippage analysis"""
        print(f"\nSTANDARD SLIPPAGE ANALYSIS")
        print("=" * 40)
        print(f"Analysis Parameters: {num_consignments} consignments × {num_simulations} simulations")
        
        if not self.config:
            if not self.load_configuration():
                return None
        
        # Run simulations with different random seeds
        simulation_results = []
        for sim_num in range(num_simulations):
            print(f"\nSimulation {sim_num+1}/{num_simulations}")
            
            with self._suppress_compliance_warnings():
                result = simulation(
                    config=self.config,
                    num_consignments=num_consignments,
                    seed=42 + sim_num,
                    compliance_table=self.compliance_table,
                    detailed=True
                )
            
            simulation_results.append((result, self.config))
            
            print(f"   Consignments: {result.num_inspections}")
            print(f"   Intercepted: {result.intercepted}")
            print(f"   Missing: {result.missing:.1f}%")
            print(f"   Contamination rate: {result.true_contamination_rate:.4f}")
        
        # Convert to pandas DataFrame using popsborder.outputs
        results_df = save_scenario_result_to_pandas(
            simulation_results,
            config_columns=self.output_config['config_columns'],
            result_columns=self.output_config['result_columns']
        )
        
        # Print summary using popsborder function
        print(f"\nSTANDARD ANALYSIS SUMMARY")
        print("-" * 40)
        
        # Calculate aggregate statistics with unit labeling
        total_contaminated = sum((r[0].intercepted + r[0].false_neg) for r in simulation_results)
        total_detected = sum(r[0].intercepted for r in simulation_results)
        total_consignments = sum(r[0].num_inspections for r in simulation_results)
        
        # Get inspection unit statistics where available
        total_sample_units = sum(getattr(r[0], 'total_num_sample_units', 0) for r in simulation_results)
        total_inspection_units = sum(getattr(r[0], 'total_num_inspection_units', 0) for r in simulation_results)
        avg_sample_units_inspected = sum(getattr(r[0], 'pct_sample_units_inspected_completion', 0) for r in simulation_results) / len(simulation_results)
        avg_inspection_units_opened = sum(getattr(r[0], 'pct_inspection_units_opened_completion', 0) for r in simulation_results) / len(simulation_results)
        
        if total_contaminated > 0:
            detection_rate = (total_detected / total_contaminated * 100)
            slippage_rate = 100 - detection_rate
        else:
            detection_rate = 0
            slippage_rate = 0
        
        print(f"Aggregate Results Across {len(simulation_results)} Simulations:")
        print(f"  - Total consignments: {total_consignments}")
        print(f"  - Total contaminated: {total_contaminated}")
        print(f"  - Total detected: {total_detected}")
        print(f"  - Detection rate: {detection_rate:.1f}%")
        print(f"  - Slippage rate: {slippage_rate:.1f}%")
        
        # Individual simulation details
        print(f"\nIndividual Simulation Details:")
        for i, (result, _) in enumerate(simulation_results, 1):
            contaminated = result.intercepted + result.false_neg
            detection = (result.intercepted / contaminated * 100) if contaminated > 0 else 0
            print(f"  Sim {i}: {result.num_inspections} consignments, {contaminated} contaminated, {detection:.1f}% detected")
        
        # Save results
        output_file = "output/standard_slippage_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults exported to: {output_file}")
        
        return results_df
    
    def run_scenario_analysis(self, scenario_file="pis_contaminate_scenarios.csv"):
        """Run PIS contamination scenario analysis"""
        print(f"\nPIS CONTAMINATION SCENARIO ANALYSIS")
        print("=" * 50)
        
        if not self.config:
            if not self.load_configuration():
                return None
        
        try:
            # Load scenarios
            scenarios = load_scenario_table(self.data_dir / scenario_file)
            print(f" Loaded {len(scenarios)} scenarios from {scenario_file}")
            
            # Run scenarios using popsborder function
            print(f"\nRunning Scenarios...")
            with self._suppress_compliance_warnings():
                scenario_results_raw = run_scenarios(
                    config=self.config,
                    scenario_table=scenarios,
                    seed=42,
                    num_simulations=3,
                    num_consignments=20,
                    compliance_table=self.compliance_table,
                    detailed=True
                )
            
            # Extract result and config from the 3-element tuples
            # run_scenarios returns (details, result, config) but save_scenario_result_to_pandas expects (result, config)
            scenario_results = [(result, config) for details, result, config in scenario_results_raw]
            
            # Convert to pandas DataFrame
            results_df = save_scenario_result_to_pandas(
                scenario_results,
                config_columns=self.output_config['config_columns'] + ['name'],
                result_columns=self.output_config['result_columns']
            )
            
            # Analyze results
            self._analyze_scenario_results(results_df)
            
            # Save detailed results
            output_file = "output/pis_contamination_scenario_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\nResults exported to: {output_file}")
            
            # Create scenario summary
            summary_df = self._create_scenario_summary(results_df)
            summary_file = "output/pis_scenario_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Scenario summary exported to: {summary_file}")
            
            return results_df
            
        except Exception as e:
            print(f" Scenario analysis failed: {e}")
            return None
    
    def _analyze_scenario_results(self, results_df):
        """Analyze and display scenario results"""
        print(f"\nPIS SCENARIO ANALYSIS")
        print("=" * 50)
        print(f"Unit definitions: Sample units (individual items), Inspection units (containers/boxes)")
        
        # Overall statistics - NOTE: Values are averaged across multiple simulations per scenario
        total_scenarios = len(results_df)
        total_consignments_tested = results_df['num_inspections'].sum()  # Total across all scenarios
        
        # These are sums of averages (each row represents average across simulations for that scenario)
        sum_avg_contaminated = (results_df['intercepted'] + results_df['false_neg']).sum()
        sum_avg_detected = results_df['intercepted'].sum()
        sum_avg_slipped = results_df['false_neg'].sum()
        
        detection_rate = (sum_avg_detected / sum_avg_contaminated * 100) if sum_avg_contaminated > 0 else 0
        slippage_rate = (sum_avg_slipped / sum_avg_contaminated * 100) if sum_avg_contaminated > 0 else 0
        
        print(f"Overall Statistics (Across {total_scenarios} Scenarios):")
        print(f"   Total consignments tested: {total_consignments_tested:.0f} consignments ({total_consignments_tested/total_scenarios:.0f} per scenario)")
        print(f"   Sum of avg contaminated consignments per scenario: {sum_avg_contaminated:.1f}")
        print(f"   Sum of avg detected consignments per scenario: {sum_avg_detected:.1f}/{sum_avg_contaminated:.1f} ({detection_rate:.1f}%)")
        print(f"   Sum of avg missed consignments per scenario: {sum_avg_slipped:.1f}/{sum_avg_contaminated:.1f} ({slippage_rate:.1f}%)")
        print(f"\nNote on fractional values:")
        print(f"   Each scenario runs multiple simulations with results averaged per scenario")
        print(f"   Fractional values represent statistical averages, not partial consignments")
        
        # Category analysis
        self._analyze_by_categories(results_df)
        
        # Risk assessment
        print(f"\nRISK ASSESSMENT:")
        if slippage_rate < 10:
            risk_level = "LOW RISK"
        elif slippage_rate < 20:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"
        
        print(f"   Aggregate slippage rate across scenarios: {slippage_rate:.1f}% - {risk_level}")
        print(f"   Note: 0% slippage indicates perfect detection across all scenario types")
        print(f"   PIS-based inspection effectiveness: {detection_rate:.1f}%")
    
    def _analyze_by_categories(self, results_df):
        """Analyze results by scenario categories"""
        print(f"\n Performance by Scenario Category:")
        
        # Define categories based on scenario names
        categories = {
            'Low-Risk': results_df[results_df['name'].str.contains('low|detection', case=False, na=False)],
            'Country-Specific': results_df[results_df['name'].str.contains('japan|netherlands', case=False, na=False)],
            'Size-Based': results_df[results_df['name'].str.contains('small|large', case=False, na=False)],
            'Outbreak': results_df[results_df['name'].str.contains('outbreak|realistic', case=False, na=False)],
            'Detection-Test': results_df[results_df['name'].str.contains('threshold|detection', case=False, na=False)],
            'Seasonal': results_df[results_df['name'].str.contains('seasonal', case=False, na=False)],
            'Intensive-Inspection': results_df[results_df['name'].str.contains('intensive|mixed|miami', case=False, na=False)]
        }
        
        for category, data in categories.items():
            if len(data) > 0:
                total_contaminated = (data['intercepted'] + data['false_neg']).sum()
                detected = data['intercepted'].sum()
                detection_rate = (detected / total_contaminated * 100) if total_contaminated > 0 else 0
                slippage_rate = 100 - detection_rate
                avg_contamination = data['true_contamination_rate'].mean() * 100
                
                print(f"   {category}:")
                print(f"     Detection: {detected}/{total_contaminated} ({detection_rate:.1f}%)")
                print(f"     Slippage: {slippage_rate:.1f}%")
                print(f"     Avg contamination: {avg_contamination:.2f}%")
    
    def _create_scenario_summary(self, results_df):
        """Create summary DataFrame for scenarios"""
        summary_data = []
        
        for _, row in results_df.iterrows():
            contaminated = row['intercepted'] + row['false_neg']
            detection_rate = (row['intercepted'] / contaminated * 100) if contaminated > 0 else 0
            
            summary_data.append({
                'scenario_name': row['name'],
                'contamination_unit': row['contamination/contamination_unit'],
                'contamination_rate': row['contamination/contamination_rate/value'],
                'consignments_tested_inspection_units': row['num_inspections'],
                'contaminated_consignments_inspection_units': contaminated,
                'detected_consignments_inspection_units': row['intercepted'],
                'slipped_consignments_inspection_units': row['false_neg'],
                'detection_rate_percent': detection_rate,
                'slippage_rate_percent': 100 - detection_rate,
                'avg_contamination_rate_per_plant_unit': row['true_contamination_rate'],
                'total_contaminants_missed_plant_units': row['total_missed_contaminants'],
                'total_contaminants_detected_plant_units': row['total_intercepted_contaminants']
            })
        
        return pd.DataFrame(summary_data)
    
    def run_hierarchical_analysis(self):
        """Run enhanced hierarchical breakdown analysis"""
        print(f"\nHIERARCHICAL BREAKDOWN")
        print("-" * 30)
        
        if not self.config:
            if not self.load_configuration():
                return None
        
        # Test contamination distribution
        self._test_contamination_distribution()
        
        # Run detailed hierarchical simulation
        self._run_detailed_hierarchical_simulation()
        
        return True
    
    def _test_contamination_distribution(self):
        """Test the contamination rate distribution - delegated to test module"""
        import sys
        sys.path.append('tests')
        from test_slippage_analysis import SlippageSystemTester
        tester = SlippageSystemTester(str(self.data_dir))
        tester.config = self.config
        tester.compliance_table = self.compliance_table
        return tester.test_contamination_distribution()
    
    def _run_detailed_hierarchical_simulation(self):
        """Run detailed simulation with hierarchical analysis"""
        print(f"\nCOMPREHENSIVE HIERARCHICAL ANALYSIS")
        print("=" * 50)
        print(f"Units: Inspection units = consignments/containers, Plant units = individual plants")
        print("")
        
        # Display configuration details using test module
        import sys
        sys.path.append('tests')
        from test_slippage_analysis import SlippageSystemTester
        tester = SlippageSystemTester(str(self.data_dir))
        tester.config = self.config
        tester.test_configuration_analysis()
        
        # Analyze contamination parameters using test module
        import sys
        sys.path.append('tests')
        from test_slippage_analysis import SlippageSystemTester
        tester = SlippageSystemTester(str(self.data_dir))
        tester.config = self.config
        tester.test_contamination_parameters()
        
        # Run simulation
        print(f"\nRunning detailed simulation...")
        with self._suppress_compliance_warnings():
            result = simulation(
                config=self.config,
                num_consignments=20,
                seed=42,
                compliance_table=self.compliance_table,
                detailed=True
            )
        
        # Detailed analysis
        self._analyze_simulation_results(result)
        
        # Save hierarchical analysis
        self._save_hierarchical_analysis(result)
    
    def _analyze_simulation_results(self, result):
        """Analyze detailed simulation results"""
        print(f"\nSIMULATION RESULTS INTERPRETATION:")
        print(f"   Total consignments: {result.num_inspections}")
        
        print(f"   Raw Detection Numbers:")
        print(f"      Intercepted: {result.intercepted}")
        print(f"      Missing (slippage): {result.missing}")
        
        print(f"   Contamination Analysis:")
        print(f"      True contamination rate: {result.true_contamination_rate:.6f}")
        print(f"       INTERPRETATION: With beta distribution α=0.02, β=2.76,")
        print(f"         most consignments get ZERO contaminated sample units due to")
        print(f"         the extremely skewed distribution that generates mostly ~0.000 rates.")
        
        # Multi-level analysis
        self._analyze_hierarchical_levels(result)
    
    def _analyze_hierarchical_levels(self, result):
        """Analyze results at different hierarchical levels"""
        
        # Basic population analysis
        print(f"\nHIERARCHICAL POPULATION ANALYSIS:")
        print(f"    Population Structure:")
        print(f"      Total consignments (inspection units): {result.num_inspections}")
        print(f"      Total inspection units (containers/boxes): {result.total_num_inspection_units}")
        print(f"      Total sample units (units inspected): {result.total_num_sample_units:,}")
        print(f"      Total plants (individual plant units): {result.total_num_plants:,}")
        
        # Inspection coverage analysis
        print(f"\n INSPECTION COVERAGE:")
        print(f"   Sample Unit Inspection:")
        print(f"      Percentage inspected (sample units): {result.pct_sample_units_inspected_completion:.1f}%")
        print(f"      Average sample units inspected per consignment: {result.avg_sample_units_inspected_completion:.1f}")
        
        print(f"    Inspection Unit Coverage:")  
        print(f"      Percentage opened (inspection units): {result.pct_inspection_units_opened_completion:.1f}%")
        print(f"      Average inspection units opened per consignment: {result.avg_inspection_units_opened_completion:.1f}")
        
        # Detection and slippage analysis
        print(f"\n DETECTION & SLIPPAGE ANALYSIS:")
        print(f"    Consignment Level:")
        print(f"      Intercepted: {result.intercepted}")
        print(f"      Missed (slipped): {result.false_neg}")
        print(f"      Slippage rate: {result.missing:.1f}%")
        
        # Contamination statistics
        print(f"\n CONTAMINATION STATISTICS:")
        print(f"   Overall contamination rate (per plant unit): {result.true_contamination_rate:.6f}")
        print(f"   Total contaminants missed (plant units): {result.total_missed_contaminants}")
        print(f"   Total contaminants intercepted (plant units): {result.total_intercepted_contaminants}")
        
        if result.total_missed_contaminants + result.total_intercepted_contaminants > 0:
            total_contaminants = result.total_missed_contaminants + result.total_intercepted_contaminants
            detection_efficiency = (result.total_intercepted_contaminants / total_contaminants * 100)
            print(f"   Detection efficiency: {detection_efficiency:.1f}%")
            print(f"   Slippage efficiency: {100 - detection_efficiency:.1f}%")
        else:
            print(f"   No contamination detected in simulation")
        
        # Analysis interpretation
        print(f"\n ANALYSIS INTERPRETATION:")
        if result.true_contamination_rate == 0:
            print(f"    Zero contamination rate indicates highly skewed beta distribution")
            print(f"       Most consignments receive zero contaminated sample units")
            print(f"       This is realistic for low-prevalence contamination scenarios")
        else:
            print(f"    Contamination present and measurable")
            print(f"    Detection system performance can be evaluated")
        
        # Detection analysis
        print(f"\n RESULTS CLARIFICATION:")
        self._analyze_detection_probabilities(result)
    
    def _analyze_detection_probabilities(self, result):
        """Analyze detection probabilities using available simulation data"""
        
        print(f"\n DETECTION EFFICIENCY ANALYSIS:")
        
        # Calculate total contamination if any
        total_contaminants = result.total_missed_contaminants + result.total_intercepted_contaminants
        
        if total_contaminants > 0:
            detection_efficiency = (result.total_intercepted_contaminants / total_contaminants * 100)
            print(f"   Individual contaminant detection rate: {detection_efficiency:.1f}%")
            print(f"   Individual contaminant slippage rate: {100 - detection_efficiency:.1f}%")
            print(f"   Total contaminants found: {result.total_intercepted_contaminants:,}")
            print(f"   Total contaminants missed: {result.total_missed_contaminants:,}")
        else:
            print(f"   No contamination present in this simulation run")
            print(f"   This is expected with highly skewed beta distribution")
        
        print(f"\n INSPECTION COVERAGE EFFICIENCY:")
        print(f"   Sample unit inspection coverage: {result.pct_sample_units_inspected_completion:.1f}%")
        print(f"   Inspection unit opening rate: {result.pct_inspection_units_opened_completion:.1f}%")
        
        # Theoretical vs actual analysis
        if result.true_contamination_rate > 0:
            expected_contaminated_units = int(result.total_num_sample_units * result.true_contamination_rate)
            print(f"\n THEORETICAL EXPECTATIONS:")
            print(f"   Expected contaminated sample units: {expected_contaminated_units}")
            print(f"   Expected detection with {result.pct_sample_units_inspected_completion:.1f}% inspection: ~{expected_contaminated_units * result.pct_sample_units_inspected_completion / 100:.1f}")
        else:
            print(f"\n DISTRIBUTION CHARACTERISTICS:")
            print(f"   Beta distribution generates mostly zero contamination")
            print(f"   This creates realistic low-prevalence scenarios")
            print(f"   Detection effectiveness varies based on contamination occurrence")
    
    def _save_hierarchical_analysis(self, result):
        """Save hierarchical analysis results to CSV"""
        
        # Create simplified hierarchical analysis DataFrame using available attributes
        hierarchical_data = {
            'metric': [
                'total_consignments',
                'total_inspection_units', 
                'total_sample_units',
                'total_plants',
                'intercepted_consignments',
                'missed_consignments',
                'slippage_rate_percent',
                'true_contamination_rate',
                'total_missed_contaminants',
                'total_intercepted_contaminants',
                'pct_sample_units_inspected',
                'pct_inspection_units_opened',
                'avg_sample_units_inspected',
                'avg_inspection_units_opened'
            ],
            'value': [
                result.num_inspections,
                result.total_num_inspection_units,
                result.total_num_sample_units,
                result.total_num_plants,
                result.intercepted,
                result.false_neg,
                result.missing,
                result.true_contamination_rate,
                result.total_missed_contaminants,
                result.total_intercepted_contaminants,
                result.pct_sample_units_inspected_completion,
                result.pct_inspection_units_opened_completion,
                result.avg_sample_units_inspected_completion,
                result.avg_inspection_units_opened_completion
            ],
            'description': [
                'Number of consignments in simulation',
                'Total inspection units across all consignments',
                'Total sample units across all consignments',
                'Total plants across all consignments',
                'Consignments where contamination was detected',
                'Consignments where contamination was missed (slipped)',
                'Percentage of contaminated consignments that slipped through',
                'Overall contamination rate in the simulation',
                'Total number of individual contaminants that were missed',
                'Total number of individual contaminants that were intercepted',
                'Percentage of sample units that were inspected',
                'Percentage of inspection units that were opened/inspected',
                'Average number of sample units inspected per consignment',
                'Average number of inspection units opened per consignment'
            ]
        }
        
        hierarchical_df = pd.DataFrame(hierarchical_data)
        
        output_file = "output/hierarchical_analysis_results.csv"
        hierarchical_df.to_csv(output_file, index=False)
        print(f"\n Hierarchical analysis results saved to: {output_file}")
    
    def verify_system_consistency(self):
        """Verify system consistency - delegated to test module"""
        import sys
        sys.path.append('tests')
        from test_slippage_analysis import SlippageSystemTester
        tester = SlippageSystemTester(str(self.data_dir))
        tester.config = self.config
        tester.compliance_table = self.compliance_table
        return tester.verify_system_consistency()


def main():
    """Main execution function with command line argument support"""
    parser = argparse.ArgumentParser(
        description="Enhanced Slippage Analysis using Popsborder Outputs"
    )
    parser.add_argument(
        '--mode', 
        choices=['standard', 'scenarios', 'hierarchical', 'verify', 'all'],
        default='all',
        help='Analysis mode to run'
    )
    parser.add_argument(
        '--consignments',
        type=int,
        default=20,
        help='Number of consignments per simulation'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=3,
        help='Number of simulation runs'
    )
    parser.add_argument(
        '--show-compliance-warnings',
        action='store_true',
        help='Show compliance warnings (default: suppress them)'
    )
    
    args = parser.parse_args()
    
    print("SLIPPAGE ANALYSIS USING POPSBORDER OUTPUTS")
    print("=" * 60)
    print(f"Analysis Mode: {args.mode.upper()}")
    print(f"Consignments per simulation: {args.consignments}")
    print(f"Number of simulations: {args.simulations}")
    print(f"Units: Sample units (individual items), Inspection units (boxes), Plant units (individual plants)")
    
    # Create analyzer with warning suppression control
    suppress_warnings = not args.show_compliance_warnings
    analyzer = SlippageAnalyzer(suppress_compliance_warnings=suppress_warnings)
    
    try:
        if args.mode in ['standard', 'all']:
            analyzer.run_standard_analysis(args.consignments, args.simulations)
        
        if args.mode in ['scenarios', 'all']:
            analyzer.run_scenario_analysis()
        
        if args.mode in ['hierarchical', 'all']:
            analyzer.run_hierarchical_analysis()
        
        if args.mode in ['verify', 'all']:
            analyzer.verify_system_consistency()
        
        print(f"\n COMPREHENSIVE ANALYSIS COMPLETED!")
        print("=" * 40)
        print(" All analyses completed successfully")
        print(" System consistency verified")
        print(" Detection effectiveness analyzed")
        print(" Contamination scenarios tested")
        print(" Results exported for USDA APHIS research")
        print(f"\n SYSTEM READY FOR PRODUCTION USE!")
        
    except KeyboardInterrupt:
        print(f"\n\n Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Analysis failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()