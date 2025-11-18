from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from gui.slippage_pipeline import (
    SyntheticOptions,
    create_default_paths,
    load_scenario_dataframe,
    run_slippage_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the slippage contamination simulation pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base directory containing config, scenario, and lookup files (defaults to slippage_data/).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of synthetic consignments to generate.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="sequential",
        help="Sampling method for synthetic data generation.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=1,
        help="Number of simulation repetitions for each scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the underlying engine.",
    )
    return parser.parse_args()


def main(
    *,
    data_dir: Optional[Path] = None,
    n_samples: int = 10,
    sampling_method: str = "sequential",
    num_simulations: int = 1,
    seed: int = 42,
):
    paths = create_default_paths(data_dir or Path("slippage_data"))
    scenario_df = load_scenario_dataframe(paths.scenario_table, dtype="object")
    options = SyntheticOptions(n_samples=n_samples, sampling_method=sampling_method)

    print("Running slippage pipeline with:")
    print(f"  data_dir:          {paths.data_dir}")
    print(f"  scenario_table:    {paths.scenario_table}")
    print(f"  config:            {paths.config}")
    print(f"  compliance_lookup: {paths.compliance_lookup}")
    print(f"  PIS data:          {paths.pis_data}")
    print(f"  RBS data:          {paths.rbs_data}")
    print(f"  synthetic seed:    {paths.synthetic_seed}")
    print(f"  synthetic output:  {paths.synthetic_output}")
    print(f"  n_samples:         {options.n_samples}")
    print(f"  sampling_method:   {options.sampling_method}")
    print(f"  num_simulations:   {num_simulations}")
    print(f"  seed:              {seed}")

    result = run_slippage_pipeline(
        paths,
        scenario_df=scenario_df,
        synthetic_options=options,
        seed=seed,
        num_simulations=num_simulations,
    )

    print("\nSynthetic data preview:")
    print(result.synthetic_data.head().to_string())

    print("\nFitted contamination parameters:")
    print(f"  Alpha: {result.contamination_fit.alpha:.6f}")
    print(f"  Beta : {result.contamination_fit.beta:.6f}")
    print(f"  Theta: {result.contamination_fit.theta:.6f}")

    print("\nScenario result preview:")
    print(result.scenario_results.head().to_string())
    print(f"\nConsignments simulated per scenario: {result.num_consignments}")

    output_path = paths.output_dir / "pis_contamination_scenario_results.csv"
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        data_dir=cli_args.data_dir,
        n_samples=cli_args.n_samples,
        sampling_method=cli_args.sampling_method,
        num_simulations=cli_args.num_simulations,
        seed=cli_args.seed,
    )
