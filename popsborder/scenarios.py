# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2021 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
==========================================================================================
Johns Hopkins University Applied Physics Laboratory (JHU/APL) Extensions and Modifications
==========================================================================================

Contributors: Gary Lin, Joseph Agor (JHU/APL)

Modifications:
    Following Functions Modified
    ----------------
    - run_scenarios():
        * Enhanced to support plant-level parameters for hierarchical consignment structure
        * Added compliance table file integration for RBS scenario execution
        * Enhanced to support input consignment data for realistic scenario modeling
        * Updated parameter handling for refactored terminology (boxes -> inspection_units, items -> sample_units)
        * Maintains backward compatibility with existing scenario definition files
    ----------------
"""

# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# this program; if not, see https://www.gnu.org/licenses/gpl-2.0.html


"""Functionality for running multiple scenarios.

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""

from .inputs import update_config
from .simulation import run_simulation
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np

from slippage_model_utils.r_script_wrapper import find_repo_root


def run_scenarios(
    config,
    scenario_table,
    seed: int,
    num_simulations: int,
    num_consignments: int,
    compliance_table=None,
    detailed: bool = False,
    output_root: Optional[Union[str, Path]] = None,
    progress_callback=None,
    use_rep_consignments: bool = False,
):
    """Run multiple simulation scenarios and collect results.

    For each scenario record in ``scenario_table``, this function:

    1. Merges the base ``config`` with scenario-specific overrides.
    2. Creates an output directory for that scenario.
    3. Generates per-replication random seeds (shared across scenarios).
    4. Calls :func:`run_simulation` to perform repeated simulation runs.
    5. Appends a tuple of results and scenario configuration to the result list.

    Args:
        config: Nested configuration dictionary used as the base for all
            scenarios.
        scenario_table: List of dictionaries with scenario-specific overrides,
            using slash-separated keys (as in :func:`update_config`).
        seed: Global seed for random number generation. All scenarios share
            this base seed, but each replication receives its own derived seed.
        num_simulations: Number of simulation replications per scenario.
        num_consignments: Number of consignments per simulation run.
        compliance_table: Optional pre-loaded compliance table (currently
            unused here, passed via config if required).
        detailed: If True, collect and return detailed per-consignment outputs
            in addition to aggregated totals.
        output_root: Optional root directory for scenario output. If None, an
            ``output/pops_border_scenario_data_<timestamp>`` directory under
            the repository root is created.
        progress_callback: Optional callback for reporting progress; it should
            accept progress information as defined by :func:`run_simulation`.
        use_rep_consignments: If True, reuse the same consignments across
            replications for a given scenario (see :func:`run_simulation`).

    Returns:
        List of results, one entry per scenario. If ``detailed`` is False,
        each entry is a tuple ``(scenario_totals, scenario_config)``.
        If ``detailed`` is True, each entry is
        ``(details, scenario_totals, scenario_config)``.
    """
    results = []

    # Define output directory for the simulated data
    if output_root is None:
        run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        run_dir = find_repo_root() / "output" / f"pops_border_scenario_data_{run_ts}"
    else:
        run_dir = Path(output_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Master RNG from global seed
    master_rng = np.random.default_rng(seed)

    # Precompute a seed for each replication index
    # So replication i has the same base seed across all scenarios
    replication_seeds = master_rng.integers(2 ** 63 - 1, size=num_simulations)
    replication_seeds_inspection = master_rng.integers(2 ** 63 - 1, size=num_simulations)

    for record in scenario_table:
        scenario_name = record["name"]
        print(f"Running scenario: {scenario_name}")
        scenario_config = update_config(config, record)

        # Create the output scenario directory
        output_dir = run_dir / str(scenario_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        scenario_rngs = [
            np.random.default_rng(int(replication_seeds[i]))
            for i in range(num_simulations)
        ]

        scenario_rngs_inspections = [
            np.random.default_rng(int(replication_seeds_inspection[i]))
            for i in range(num_simulations)
        ]

        result = run_simulation(
            config=scenario_config,
            num_simulations=num_simulations,
            num_consignments=num_consignments,
            rngs=scenario_rngs,
            rngs_inspections=scenario_rngs_inspections,
            detailed=detailed,
            output_dir=output_dir,
            progress_callback=progress_callback,
            use_rep_consignments=use_rep_consignments
        )
        if detailed:
            # The result is tuple of details ([0]) and simulation totals ([1]).
            results.append((result[0], result[1], scenario_config))
        else:
            # The result is simulation totals.
            results.append((result, scenario_config))
    return results
