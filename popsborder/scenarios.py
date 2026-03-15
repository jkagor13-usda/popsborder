# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2021 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
Modifications:
- 10/3/2025: Modeifications described below (Gary Lin)
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


"""Functionality for running multiple scenarios

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""

from .inputs import update_config
from .simulation import run_simulation
from datetime import datetime
from pathlib import Path
import numpy as np

from slippage_model_utils.r_script_wrapper import find_repo_root

def run_scenarios(
    config, scenario_table, seed, num_simulations, num_consignments, compliance_table=None, 
    detailed=False
):
    """Run scenarios based on the configuration and list of scenarios

    Parameters
    ----------
    config : nested dict
        Basic configuration for each simulation.
    scenario_table : list of dicts
        Configurations specific for each scenario as a list of dictionaries with
        key/subkey/subsubkey keys.
    seed : int
        Seed for a random generator. All scenarios get the same seed, but each
        simulation within a scenario runs with a different seed based on this one.
    num_simulations : int
        Num of simulations for each scenario.
    num_consignments : int
        Number of shipements in each simulation.

    Returns
    -------
    results : list of tuples
        List of results with one tuple for each scenario. One tuple contains simulation
        result and configuration for that scenario.
    """
    results = []

    # Create master RNG once at the top level
    master_rng = np.random.default_rng(seed)

    # Define output directory for the simulated data
    run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = find_repo_root() / "output" / f"pops_border_scenario_data_{run_ts}"
    run_dir = Path(run_dir)
    for record in scenario_table:
        scenario_name = record["name"]
        print(f"Running scenario: {scenario_name}")
        scenario_config = update_config(config, record)

        # Create the output scenario directory
        output_dir = run_dir / str(scenario_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Spawn independent RNG for this scenario
        scenario_rng = master_rng.spawn(1)[0]

        result = run_simulation(
            config=scenario_config,
            num_simulations=num_simulations,
            num_consignments=num_consignments,
            rng=scenario_rng,
            detailed=detailed,
            output_dir=output_dir,
        )
        if detailed:
            # The result is tuple of details ([0]) and simulation totals ([1]).
            results.append((result[0], result[1], scenario_config))
        else:
            # The result is simulation totals.
            results.append((result, scenario_config))
    return results
