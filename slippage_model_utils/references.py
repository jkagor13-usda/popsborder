# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC


"""Reference File to Hold Naming Conventions for Slippage Model

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

Purpose of this file:
-------------------
- sample_rbs():
    * Implements risk-based sampling methodology using compliance-based detection levels
    * Retrieves country/propagative material specific compliance parameters from lookup table
    * Calculates sample size using hypergeometric distribution based on risk assessment

- select_units_to_inspect():
    * Unified function for selecting units to inspect based on selection strategy
    * Supports random, cluster, and convenience selection strategies
    * Handles both inspection_unit and sample_unit selection

- count_contaminated_inspection_units():
    * Counts contaminated inspection units in consignment
    * Supports refactored terminology (inspection_units vs boxes)

- count_contaminated_sample_units():
    * Counts contaminated sample units in consignment
    * Supports refactored terminology (sample_units vs items)

Modified Functions:
----------------
- get_sample_function():
    * Added RBS structured consignment inspection
    * Enhanced to support compliance table parameter passing

- sample_proportion():
    * Added backward compatibility for min_inspection_units (formerly min_boxes)
    * Updated to handle both old and new terminology in configuration

- sample_n():
    * Added backward compatibility for within_inspection_unit_proportion (formerly within_box_proportion)
    * Added backward compatibility for min_inspection_units (formerly min_boxes)

- convert_sample_units_to_inspection_units_fixed_proportion():
    * Added backward compatibility for within_inspection_unit_proportion

- compute_n_clusters_to_inspect():
    * Added backward compatibility for within_inspection_unit_proportion and min_inspection_units

- inspect():
    * Added backward compatibility for within_inspection_unit_proportion
    * Enhanced detailed tracking with sample_unit_in_inspection_unit_to_sample_unit_index()

Backward Compatibility:
----------------------
- Added aliases: count_contaminated_boxes() -> count_contaminated_inspection_units()
- Added aliases: count_contaminated_items() -> count_contaminated_sample_units()
- Configuration parameter mapping: boxes -> inspection_units, items -> sample_units
- Maintained support for legacy configuration keys while enabling new terminology

"""

country_of_origin_names = {
    'Origin', 'origin', 'Origin Location Country Name', 'Origin Location Name',
    'Origin Name', 'Country Name', 'Country',
    'Country of Origin', 'Country of origin', 'Origin Country'
}
pm_type_names = {
    'Propagative Material Type', 'Propagative Material', 'PM Type',
    'PM', 'Material', 'Material Type'
}

producer_names = {
    'PRODUCER_NAME', 'Producer', 'Producer Name',
    'Prod', 'Producer Group', 'PRODUCER'
}

possible_pis_stations = {
    'Atlanta PIS',
    'Beltsville PIS',
    'Carolina PIS',
    'Guam PIS',
    'Honolulu PIS',
    'Houston PIS',
    'JFK PIS',
    'Linden PIS',
    'Los Angeles PIS',
    'Los Indios PIS',
    'Miami PIS',
    'Nogales PIS',
    'Orlando PIS',
    'San Diego PIS',
    'San Francisco PIS',
    'Seattle PIS'
}

