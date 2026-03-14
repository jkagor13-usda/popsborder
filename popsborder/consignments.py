# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)
# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

"""
Modifications:
- 10/3/2025: Modeifications described below (Gary Lin)
    Following New Classes Added
    ------------------
    - PISConsignmentGenerator:
        * Generate consignments based on existing Plant Inspection System (PIS) records
    - SampleUnit:
        * Manages plant-level sampling units within individual sample_units
        * Provides contamination detection at the plant level
        * Integrates with hierarchical inspection structure
        * Supports numpy array-based plant contamination tracking
    ------------------

    Following Classes Modified
    ----------------
    - InspectionUnit (formerly 'Box'):
        * Extended with sample_unit_objects parameter for hierarchical structure
        * Added material_type attribute to support different materials within a consignment
        * Supports multi-level contamination detection (sample_unit + plant levels)
        * Enhanced with SampleUnit object array for hierarchical access
        * Maintains backward compatibility with original array-based functionality

    - Consignment:
        * Added plant-level attributes (num_plants, plants, plants_per_sample_unit)
        * Enhanced with material_type classification for RBS workflows
        * Supports both traditional sample_unit-based and hierarchical plant-based inspection
        * Added backward compatibility attribute mapping (boxes -> inspection_units, items -> sample_units)
        * Enhanced count_contaminated() method for flexible contamination counting
    ----------------

    New Functions Added
    ----------------
    - get_sample_units_per_inspection_unit():
        * Determines sample_units per inspection_unit based on pathway type
        * Supports pathway-specific packaging configurations
        * Handles backward compatibility for items_per_box

    - sample_unit_in_inspection_unit_to_sample_unit_index():
        * Converts local sample_unit index within inspection_unit to global sample_unit index
        * Essential for hierarchical structure navigation and detailed tracking

    - get_inspection_unit_and_sample_unit_index():
        * Converts global sample_unit index to inspection_unit and local sample_unit indices
        * Supports bidirectional navigation in hierarchical structure
    ----------------

    Following Functions Modified
    ----------------
    - get_consignment_generator():
        * Supports RBS generation method selection (parameter-based and record-based)
        * Enhanced backward compatibility for configuration parameter mapping
        * Handles both old terminology (items_per_box) and new terminology (sample_units_per_inspection_unit)
        * Automatically selects PISConsignmentGenerator when file_type="PIS" is specified for RBS method
    - get_items_per_box():
        * Renamed get_plants_per_sample_unit() and updated to match plant terminology
        * Determines number of plants per sample_unit based on pathway type
        * Supports pathway-specific plant quantity configurations
        * Handles backward compatibility for plants_per_item
    ----------------

    Terminology Refactoring
    ----------------
    - Systematically refactored: boxes -> inspection_units, items -> sample_units
    - Updated all class names, method names, and variable names for consistency
    - Added comprehensive backward compatibility for existing configurations
    - Maintained dual access patterns for smooth migration from legacy terminology
    ----------------

    Backward Compatibility
    ----------------
    - Configuration parameter mapping: boxes -> inspection_units, items -> sample_units
    - Legacy attribute access in Consignment class via __getattr__ and __hasattr__
    - Support for both items_per_box and sample_units_per_inspection_unit configuration keys
    - Maintained existing F280 and AQIM consignment generator functionality
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


"""Consignment generation

.. codeauthor:: Vaclav Petras <wenzeslaus gmail com>
.. codeauthor:: Kellyn P. Montgomery <kellynmontgomery gmail com>
.. codeauthor:: Gary Lin <Gary.Lin jhuapl edu>
.. codeauthor:: Joseph Agor <Joseph.Agor jhuapl edu>
"""


import collections
import csv
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from slippage_model_utils.UnitAttributes import RiskUnitConfig


class RiskUnit:
    """Risk Unit

    RiskUnit is a view into array of sample_units, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, sample_units can not
    only be accessed but also modifed through the RiskUnit.
    """

    def __init__(self,
                 sample_units,
                 risk_unit_id=None,
                 sample_unit_ids=None,
                 inspection_unit_ids=None,
                 plant_ids=None,
                 **kwargs):  # Catch all other attributes
        """Store reference to associated sample_units"""
        self.sample_units = sample_units
        self.id = risk_unit_id
        self.sample_unit_ids = sample_unit_ids if sample_unit_ids is not None else []
        self.inspection_unit_ids = (
            inspection_unit_ids if inspection_unit_ids is not None else []
        )
        self.plant_ids = plant_ids if plant_ids is not None else []

        # Set all other attributes dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

    # def __init__(self,
    #              sample_units,
    #              risk_unit_id=None,
    #              sample_unit_ids=None,
    #              inspection_unit_ids=None,
    #              plant_ids=None,
    #              material_type=None,
    #              producer=None,
    #              origin=None,
    #              port=None,
    #              pathway=None):
    #     """Store reference to associated sample_units
    #
    #     :param sample_units: Array-like object of sample_units
    #     :param material_type: Material type for this inspection unit
    #     :param producer: Producer name for this inspection unit
    #     """
    #     self.sample_units = sample_units
    #     self.id = risk_unit_id
    #     self.sample_unit_ids = sample_unit_ids if sample_unit_ids is not None else []
    #     self.inspection_unit_ids = (
    #         inspection_unit_ids if inspection_unit_ids is not None else []
    #     )
    #     self.plant_ids = plant_ids if plant_ids is not None else []
    #     self.material_type = material_type
    #     self.producer = producer
    #     self.origin = origin
    #     self.port = port
    #     self.pathway = pathway

    @property
    def num_sample_units(self):
        """Number of sample_units in the InspectionUnit"""
        if hasattr(self.sample_units, 'shape'):
            return self.sample_units.shape[0]
        else:
            return len(self.sample_units)

    @property
    def n_for_hypergeom(self):
        """Population size N used by hypergeometric sampling (fixed to sample unit count)."""
        return len(self.sample_unit_ids) if self.sample_unit_ids else self.num_sample_units

    def __bool__(self):
        if isinstance(self.sample_units, np.ndarray):
            return bool(np.any(self.sample_units > 0))
        else:
            # Assuming it's a list of SampleUnit objects
            return any(bool(su) for su in self.sample_units)


class InspectionUnit:
    """Inspection unit with fixed hierarchy support.

    Supported shapes:
    - one RiskUnit (which holds SampleUnit data)
    - multiple SampleUnit objects
    - one SampleUnit object represented as a single-element sample_units list
    """

    def __init__(self,
                 included_units=None,
                 material_type=None,
                 producer=None,
                 origin=None,
                 port=None,
                 pathway=None,
                 inspection_unit_id=None,
                 risk_unit_ids=None,
                 sample_unit_ids=None,
                 plant_ids=None,
                 action_status="pending",
                 inspection_print_id = None,
                 is_detected=False):
        """Store references to fixed-hierarchy unit collections."""
        if included_units is None:
            included_units = []

        self.included_units = included_units
        self._included_unit_objects = []
        self.id = inspection_unit_id
        self.inspection_print_id = inspection_print_id

        self.material_type = material_type
        self.producer = producer
        self.origin = origin
        self.port = port
        self.pathway = pathway

        # Fixed hierarchy anchors.
        self.risk_unit_ids = risk_unit_ids if risk_unit_ids is not None else []
        self.sample_unit_ids = sample_unit_ids if sample_unit_ids is not None else []
        self.plant_ids = plant_ids if plant_ids is not None else []
        self.action_status = action_status
        self.is_detected = bool(is_detected)

    @property
    def num_included_units(self):
        """Number of primary units held by this container."""
        if hasattr(self.included_units, "shape"):
            return self.included_units.shape[0]
        return len(self.included_units)

    @property
    def sample_units_per_inspection_unit(self):
        """Backward-compatible size accessor used by legacy logic."""
        if self._included_unit_objects:
            return len(self._included_unit_objects)
        return self.num_included_units

    @property
    def num_sample_units(self):
        """Backward-compatible alias used by contamination logic."""
        return self.sample_units_per_inspection_unit

    @property
    def included_unit_objects(self):
        return self._included_unit_objects

    @included_unit_objects.setter
    def included_unit_objects(self, units):
        self._included_unit_objects = units

    def __bool__(self):
        # Ground-truth infection for an inspection unit should come from its own
        # nested sample/plant data, not the whole risk unit.
        if self._included_unit_objects:
            return any(bool(sample_unit) for sample_unit in self._included_unit_objects)
        if isinstance(self.included_units, np.ndarray):
            return bool(np.any(self.included_units > 0))

        return any(bool(unit) for unit in self.included_units)

    @property
    def is_infected(self):
        return bool(self)

    def reset_detection(self):
        self.is_detected = False

    def mark_detected(self):
        self.is_detected = True


class SampleUnit:
    """Sample unit

    Evaluates to bool when it contains contaminant.

    SampleUnit Container is a view into array of plants, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, plants can not
    only be accessed but also modifed through the InspectionUnit.
    """

    def __init__(self,
                 plants=None,
                 sample_unit_id=None,
                 risk_unit_id=None,
                 inspection_unit_id=None,
                 plant_ids=None,
                 producer=None,
                 origin=None,
                 port=None,
                 pathway=None):
        """Store reference to associated plants

        :param plants: Array-like object of plants
        :param producer: Producer name for this sample unit
        """
        self.plants = plants
        self.id = sample_unit_id
        self.risk_unit_id = risk_unit_id
        self.inspection_unit_id = inspection_unit_id
        self.plant_ids = plant_ids if plant_ids is not None else []
        self.producer = producer
        self.origin = origin
        self.port = port
        self.pathway = pathway

    @property
    def num_plants(self):
        """Number of plants in the Sample Unit"""
        if self.plants is None:
            return 0
        return self.plants.shape[0] if hasattr(self.plants, "shape") else len(self.plants)

    def __bool__(self):
        if self.plants is None:
            return False
        return bool(np.any(self.plants > 0))

    @property
    def is_infected(self):
        return bool(self)


class Consignment(collections.UserDict):
    """A consignment with all its properties and what it contains.

    Access its properties (attributes) is through attribute syntax (new style) or
    using a dictionary-like item access (old style).
    """

    # Inheriting from this library class is its intended use, so disable ancestors msg.
    # pylint: disable=too-many-ancestors
    # This class is meant to hold a lot of attributes.
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        num_sample_units,
        sample_units,
        sample_units_per_inspection_unit,
        num_inspection_units,
        date,
        inspection_units,
        origin,
        port,
        pathway,
        material_type=None,
        num_plants=None,
        plants=None,
        plants_per_sample_unit=None,
        inspection_number=None,
        producer=None,
        risk_units=None,
        sample_unit_objects=None,
        risk_unit_to_inspection_units=None,
        inspection_unit_to_risk_units=None,
        risk_unit_to_sample_units=None,
        sample_unit_to_inspection_unit=None,
        plant_unit_to_inspection_unit=None,
    ):
        """Store reference to associated attributes

        :param flower: string
        :param num_sample_units: integer
        :param material_type: string
        :param sample_units_per_inspection_unit: integer
        :param num_inspection_units: integer
        :param date: Array-like object of dates
        :param inspection_units: Array-like object of inspection_units
        :param origin: string
        :param port: string
        :param pathway: string
        :param num_plants : string (optional)
        :param plants : Array-like object of plants (optional)
        :param plants_per_sample_unit : string (optional)
        """
        super().__init__(
            num_sample_units=num_sample_units,
            sample_units=sample_units,
            sample_units_per_inspection_unit=sample_units_per_inspection_unit,
            num_inspection_units=num_inspection_units,
            date=date,
            inspection_units=inspection_units,
            origin=origin,
            port=port,
            pathway=pathway,
            material_type = material_type,
            num_plants=num_plants,
            plants=plants,
            plants_per_sample_unit=plants_per_sample_unit,
            inspection_number=inspection_number,
            producer=producer,
            risk_units=risk_units,
            sample_unit_objects=sample_unit_objects,
            risk_unit_to_inspection_units=risk_unit_to_inspection_units,
            inspection_unit_to_risk_units=inspection_unit_to_risk_units,
            risk_unit_to_sample_units=risk_unit_to_sample_units,
            sample_unit_to_inspection_unit=sample_unit_to_inspection_unit,
            plant_unit_to_inspection_unit=plant_unit_to_inspection_unit,
        )
        self.num_sample_units = num_sample_units
        self.sample_units = sample_units
        self.sample_units_per_inspection_unit = sample_units_per_inspection_unit
        self.num_inspection_units = num_inspection_units
        self._date = date
        self.inspection_units = inspection_units
        self.origin = origin
        self.port = port
        self.pathway = pathway
        self.material_type = material_type
        self.num_plants = num_plants
        self.plants = plants
        self.plants_per_sample_unit = plants_per_sample_unit
        self.inspection_number = inspection_number
        self.producer = producer
        self.risk_units = risk_units
        self.sample_unit_objects = (
            sample_unit_objects if sample_unit_objects is not None else []
        )
        self.risk_unit_to_inspection_units = (
            risk_unit_to_inspection_units if risk_unit_to_inspection_units is not None else {}
        )
        self.inspection_unit_to_risk_units = (
            inspection_unit_to_risk_units if inspection_unit_to_risk_units is not None else {}
        )
        self.risk_unit_to_sample_units = (
            risk_unit_to_sample_units if risk_unit_to_sample_units is not None else {}
        )
        self.sample_unit_to_inspection_unit = (
            sample_unit_to_inspection_unit if sample_unit_to_inspection_unit is not None else {}
        )
        self.plant_unit_to_inspection_unit = (
            plant_unit_to_inspection_unit if plant_unit_to_inspection_unit is not None else {}
        )
        self._build_hierarchy_indexes()

    def _build_hierarchy_indexes(self):
        """Build hierarchy maps when explicit maps are not provided."""
        if not self.sample_unit_objects:
            self.sample_unit_objects = []
            for inspection_unit in self.inspection_units or []:
                self.sample_unit_objects.extend(getattr(inspection_unit, "included_unit_objects", []))

        if not self.risk_unit_to_inspection_units and self.risk_units:
            for risk_unit in self.risk_units:
                self.risk_unit_to_inspection_units[risk_unit.id] = list(
                    getattr(risk_unit, "inspection_unit_ids", [])
                )

        if not self.inspection_unit_to_risk_units and self.inspection_units:
            for inspection_unit in self.inspection_units:
                if inspection_unit.id is None:
                    continue
                self.inspection_unit_to_risk_units[inspection_unit.id] = list(
                    getattr(inspection_unit, "risk_unit_ids", [])
                )

        if not self.risk_unit_to_sample_units and self.risk_units:
            for risk_unit in self.risk_units:
                self.risk_unit_to_sample_units[risk_unit.id] = list(
                    getattr(risk_unit, "sample_unit_ids", [])
                )

        if not self.sample_unit_to_inspection_unit and self.sample_unit_objects:
            for sample_unit in self.sample_unit_objects:
                if sample_unit.id is not None:
                    self.sample_unit_to_inspection_unit[sample_unit.id] = sample_unit.inspection_unit_id

        if not self.plant_unit_to_inspection_unit and self.inspection_units:
            for inspection_unit in self.inspection_units:
                inspection_unit_id = getattr(inspection_unit, "id", None)
                for plant_id in getattr(inspection_unit, "plant_ids", []):
                    self.plant_unit_to_inspection_unit[plant_id] = inspection_unit_id

    def __hasattr__(self, name):
        return name in self

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return self[name]

    @property
    def date(self):
        """Date of consignment (arrival) as datetime.date object"""
        # For datetime.datetime, returns just date, assumes datetime.date otherwise.
        if hasattr(self._date, "date"):
            return self._date.date()
        return self._date

    @property
    def commodity(self):
        """Convenient (or transitional) alias for flower"""
        return self.flower

    def count_contaminated(self):
        if self.plants is None:
            """Count contaminated sample_units in inspection_unit."""
            return np.count_nonzero(self.sample_units)
        else:
            """Count contaminated plants in inspection_unit."""
            return np.count_nonzero(self.plants)

    def item_in_inspection_unit_to_item_index(self, inspection_unit_index, item_in_inspection_unit_index):
        """Convert item index in a inspection_unit to item index in the consignment"""
        if inspection_unit_index == 0:
            return item_in_inspection_unit_index
        sample_units = 0
        for inspection_unit in self.inspection_units[:inspection_unit_index]:
            sample_units += inspection_unit.sample_units_per_inspection_unit
        return sample_units + item_in_inspection_unit_index
    
    def get_inspection_unit_and_sample_unit_index(self, sample_unit_index):
        inspection_unit_index = sample_unit_index // self.sample_units_per_inspection_unit
        sample_unit_index_in_inspection_unit = sample_unit_index % self.sample_units_per_inspection_unit
        return inspection_unit_index, sample_unit_index_in_inspection_unit
    
    def sample_unit_in_inspection_unit_to_sample_unit_index(self, inspection_unit_index, sample_unit_in_inspection_unit_index):
        """Convert sample_unit index within inspection_unit to global sample_unit index"""
        return inspection_unit_index * self.sample_units_per_inspection_unit + sample_unit_in_inspection_unit_index


class ParameterConsignmentGenerator:
    """Generate a consignments based on configuration parameters"""

    def __init__(self, parameters, sample_units_per_inspection_unit, start_date):
        """Set parameters for consignment generation

        :param parameters: Consignment parameters
        :param ports: List of ports to choose from
        :param sample_units_per_inspection_unit: Configuration driving number of sample_units per inspection_unit
        :param start_date: Date to start consignment dates from
        """
        self.params = parameters
        self.sample_units_per_inspection_unit = sample_units_per_inspection_unit
        self.num_generated = 0
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.date = start_date

    def generate_consignment(self):
        """Generate a new consignment"""
        port = random.choice(self.params["ports"])
        # flowers or commodities
        flower = random.choice(self.params["flowers"])
        origin = random.choice(self.params["origins"])
        num_inspection_units_min = self.params["inspection_units"].get("min", 0)
        num_inspection_units_max = self.params["inspection_units"]["max"]
        pathway = "None"
        sample_units_per_inspection_unit = self.sample_units_per_inspection_unit
        sample_units_per_inspection_unit = get_sample_units_per_inspection_unit(sample_units_per_inspection_unit, pathway)
        num_inspection_units = random.randint(num_inspection_units_min, num_inspection_units_max)
        num_sample_units = sample_units_per_inspection_unit * num_inspection_units
        sample_units = np.zeros(sample_units_per_inspection_unit, dtype=np.int64)
        inspection_units = []
        for i in range(num_inspection_units):
            lower = i * sample_units_per_inspection_unit
            upper = (i + 1) * sample_units_per_inspection_unit
            inspection_units.append(InspectionUnit(sample_units[lower:upper], material_type=flower))
        self.num_generated += 1
        # two consignments every nth day
        if self.num_generated % 3:
            self.date += timedelta(days=1)

        return Consignment(
            material_type=flower,
            num_sample_units=num_sample_units,
            sample_units=sample_units,
            sample_units_per_inspection_unit=sample_units_per_inspection_unit,
            num_inspection_units=num_inspection_units,
            date=self.date,
            inspection_units=inspection_units,
            origin=origin,
            port=port,
            pathway=pathway,
        )

class F280ConsignmentGenerator:
    """Generate a consignments based on existing F280 records"""

    def __init__(self, sample_units_per_inspection_unit, filename, separator=","):
        self.infile = open(filename)
        self.reader = csv.DictReader(self.infile, delimiter=separator)
        self.sample_units_per_inspection_unit = sample_units_per_inspection_unit

    def generate_consignment(self):
        """Generate a new consignment"""
        try:
            record = next(self.reader)
        except StopIteration:
            raise RuntimeError(
                "More consignments requested than number of records in provided F280"
            ) from None

        num_sample_units = int(record["QUANTITY"])
        sample_units = np.zeros(num_sample_units, dtype=np.int64)

        pathway = record["PATHWAY"]
        sample_units_per_inspection_unit = self.sample_units_per_inspection_unit
        sample_units_per_inspection_unit = get_sample_units_per_inspection_unit(sample_units_per_inspection_unit, pathway)

        # rounding up to keep the max per inspection_unit and have enough inspection_units
        num_inspection_units = int(math.ceil(sample_units_per_inspection_unit / float(sample_units_per_inspection_unit)))
        num_inspection_units = max(num_inspection_units, 1)
        inspection_units = []
        for i in range(num_inspection_units):
            lower = i * sample_units_per_inspection_unit
            # slicing does not go over the size even if our last inspection_unit is smaller
            upper = (i + 1) * sample_units_per_inspection_unit
            inspection_units.append(InspectionUnit(sample_units[lower:upper], material_type=record["COMMODITY"]))
        assert sum([inspection_unit.sample_units_per_inspection_unit for inspection_unit in inspection_units]) == sample_units_per_inspection_unit

        date = datetime.strptime(record["REPORT_DT"], "%Y-%m-%d")
        return Consignment(
            material_type=record["COMMODITY"],
            num_sample_units=num_sample_units,
            sample_units=sample_units,
            sample_units_per_inspection_unit=sample_units_per_inspection_unit,
            num_inspection_units=num_inspection_units,
            date=date,
            inspection_units=inspection_units,
            origin=record["ORIGIN_NM"],
            port=record["LOCATION"],
            pathway=pathway,
        )


class PISConsignmentGenerator:
    """Generate consignments based on existing Plant Inspection System (PIS) records
    
    This is a record-based generator under the RBS (Risk-Based Sampling) method that handles
    real Plant Inspection System data with multiple material types per consignment.
    
    The generator is fully data-driven and calculates sample_units_per_inspection_unit and 
    plants_per_sample_unit directly from the TOTAL_SAMPLING_UNITS and TOTAL_PLANT_QUANTITY 
    columns in the PIS data. No configuration parameters are needed for quantities.
    
    Timestamp Integration:
    - Uses CREATED_DATETIME column (MM:SS.S format) for realistic consignment timing
    - Converts timestamps to datetime objects with base date 2020-01-01
    - Falls back to default date if timestamp parsing fails
    - Preserves inspection timing information for temporal analysis
    
    Configuration:
        generation_method: "RBS"
        input_file:
            file_type: "PIS"
            file_name: "path/to/pis_data.csv"
    """

    def __init__(self, filename, separator=",", risk_unit_config: RiskUnitConfig=RiskUnitConfig()):
        """Initialize PIS record-based consignment generator

        :param filename: CSV file containing PIS records with columns:
                        - INSPECTION_NUMBER: Unique consignment identifier
                        - PROPAGATIVE_MATERIAL_TYPE: Material type for each inspection unit
                        - TOTAL_SAMPLING_UNITS: Actual number of sample units
                        - TOTAL_PLANT_QUANTITY: Actual number of plants
                        - COUNTRY_OF_ORIGIN_NAME, INSPECTION_LOCATION_NAME, PATHWAY
                        - CREATED_DATETIME: Timestamp in MM:SS.S format (optional)
                        - PRODUCER: Producer name (optional)
        :param separator: CSV field separator
        :param risk_unit_config: RiskUnitConfig instance for attribute mapping
        """

        self.df = pd.read_csv(filename, sep=separator)
        # Group by inspection number to create consignments
        self.consignment_groups = list(self.df.groupby('INSPECTION_NUMBER'))
        self.current_consignment_index = 0
        self.risk_unit_config = risk_unit_config or RiskUnitConfig()

    def generate_consignment(self):
        """Generate a new consignment from PIS records"""
        if self.current_consignment_index >= len(self.consignment_groups):
            raise RuntimeError(
                "More consignments requested than number of inspection numbers in provided PIS data"
            ) from None

        inspection_number, inspection_units_data = self.consignment_groups[self.current_consignment_index]
        self.current_consignment_index += 1





        """
        Consignment Attributes to be Generated
        num_sample_units,
        sample_units,
        sample_units_per_inspection_unit,
        num_inspection_units,
        date,
        inspection_units,
        origin,
        port,
        pathway,
        material_type=None,
        num_plants=None,
        plants=None,
        plants_per_sample_unit=None,
        inspection_number=None,
        producer=None,
        risk_units=None,
        """

        # Extract common consignment info from first record
        first_record = inspection_units_data.iloc[0]
        pathway = first_record.get("PATHWAY", "None")
        origin = first_record["COUNTRY_OF_ORIGIN_NAME"]
        port = first_record["INSPECTION_LOCATION_NAME"]

        # Build fixed hierarchy:
        # consignment -> risk_units -> sample_units -> plant_units
        total_sample_units = 0
        total_plants = 0
        inspection_units = []
        risk_units = []
        all_sample_unit_objects = []
        sample_unit_by_id = {}
        risk_unit_buckets = {}

        risk_unit_column = None
        for candidate in ["RISK UNIT", "RISK_UNIT", "risk_unit", "RISK_UNIT_NUMBER"]:
            if candidate in inspection_units_data.columns:
                risk_unit_column = candidate
                break
        if risk_unit_column is None:
            raise RuntimeError(
                "PIS input is missing risk unit label column. Expected one of: "
                "'RISK UNIT', 'RISK_UNIT', 'risk_unit', 'RISK_UNIT_NUMBER'."
            )
        
        # Process each inspection unit (row) in this consignment
        for inspection_unit_id, (_, record) in enumerate(inspection_units_data.iterrows()):
            material_type = record["PROPAGATIVE_MATERIAL_TYPE"]
            raw_risk_unit = record.get(risk_unit_column)
            if raw_risk_unit is None or str(raw_risk_unit).strip() == "":
                raise RuntimeError(
                    f"Record for inspection '{inspection_number}' has empty risk unit label "
                    f"in column '{risk_unit_column}'."
                )
            risk_unit_id = str(raw_risk_unit).strip()
            
            # Use the actual quantities from the PIS data
            inspection_unit_sample_units = int(record["SAMPLING_UNITS_FOR_INSPECTION_UNIT"])
            inspection_unit_plants = int(record["QUANTITY"])
            inspection_unit_producer = record.get("producer_group", None)
            inspection_unit_origin = record.get("COUNTRY_OF_ORIGIN_NAME", origin)
            inspection_unit_port = record.get("INSPECTION_LOCATION_NAME", port)
            inspection_unit_pathway = record.get("PATHWAY", pathway)
            inspection_unit_row_id = record.get("Row_ID", pathway)

            # Extract attributes using configuration
            risk_unit_attrs = self.risk_unit_config.get_attributes_from_record(record)

            # Calculate plants per sample_unit for this inspection unit
            if inspection_unit_sample_units > 0:
                base_plants_per_sample_unit = inspection_unit_plants // inspection_unit_sample_units
                remainder_plants = inspection_unit_plants % inspection_unit_sample_units
            else:
                base_plants_per_sample_unit = 1
                remainder_plants = 0
                inspection_unit_sample_units = 1
            
            # Create arrays for this inspection unit
            sample_units_array = np.zeros(inspection_unit_sample_units, dtype=np.int64)
            plants_array = np.zeros(inspection_unit_plants, dtype=np.int64)
            
            # Create SampleUnit objects for hierarchical access
            sample_unit_objects = []
            inspection_sample_unit_ids = []
            inspection_plant_ids = []
            plant_index = 0
            
            for item_index in range(inspection_unit_sample_units):
                # Distribute remainder plants among first few sample units
                plants_in_this_unit = base_plants_per_sample_unit
                if item_index < remainder_plants:
                    plants_in_this_unit += 1
                
                # Ensure we don't exceed total plants
                plants_in_this_unit = min(plants_in_this_unit, inspection_unit_plants - plant_index)
                
                # Create slice for this sample unit
                sample_unit_plants = plants_array[plant_index:plant_index + plants_in_this_unit]
                sample_unit_id = total_sample_units
                total_sample_units += 1

                plant_ids_for_sample_unit = []
                for local_plant_offset in range(plants_in_this_unit):
                    plant_unit_id = total_plants
                    total_plants += 1
                    plant_ids_for_sample_unit.append(plant_unit_id)
                    inspection_plant_ids.append(plant_unit_id)

                sample_unit = SampleUnit(
                    sample_unit_plants,
                    sample_unit_id=sample_unit_id,
                    risk_unit_id=risk_unit_id,
                    inspection_unit_id=inspection_unit_id,
                    plant_ids=plant_ids_for_sample_unit,
                    producer=inspection_unit_producer,
                    origin=inspection_unit_origin,
                    port=inspection_unit_port,
                    pathway=inspection_unit_pathway,
                )
                sample_unit_objects.append(sample_unit)
                all_sample_unit_objects.append(sample_unit)
                sample_unit_by_id[sample_unit_id] = sample_unit
                inspection_sample_unit_ids.append(sample_unit_id)
                plant_index += plants_in_this_unit

            # Create InspectionUnit
            inspection_unit = InspectionUnit(
                sample_units_array,
                material_type=material_type,
                producer=inspection_unit_producer,
                origin=inspection_unit_origin,
                port=inspection_unit_port,
                pathway=inspection_unit_pathway,
                inspection_unit_id=inspection_unit_id,
                risk_unit_ids=[risk_unit_id],
                sample_unit_ids=inspection_sample_unit_ids,
                plant_ids=inspection_plant_ids,
                inspection_print_id=inspection_unit_row_id,
            )

            inspection_unit.included_unit_objects = sample_unit_objects
            inspection_units.append(inspection_unit)

            # Use configured attributes when creating bucket
            if risk_unit_id not in risk_unit_buckets:
                risk_unit_buckets[risk_unit_id] = {
                    "inspection_unit_ids": [],
                    "sample_unit_ids": [],
                    "plant_ids": [],
                    **risk_unit_attrs  # Unpack configured attributes
                }

            risk_bucket = risk_unit_buckets[risk_unit_id]
            risk_bucket["inspection_unit_ids"].append(inspection_unit_id)
            risk_bucket["sample_unit_ids"].extend(inspection_sample_unit_ids)
            risk_bucket["plant_ids"].extend(inspection_plant_ids)

        # Create risk units from accumulated hierarchy links
        for risk_unit_id, bucket in risk_unit_buckets.items():
            # risk_unit = RiskUnit(
            #     np.zeros(len(bucket["sample_unit_ids"]), dtype=np.int64),
            #     risk_unit_id=risk_unit_id,
            #     sample_unit_ids=bucket["sample_unit_ids"],
            #     inspection_unit_ids=bucket["inspection_unit_ids"],
            #     plant_ids=bucket["plant_ids"],
            #     material_type=bucket["material_type"],
            #     producer=bucket["producer"],
            #     origin=bucket["origin"],
            #     port=bucket["port"],
            #     pathway=bucket["pathway"],
            # )
            # Build kwargs dynamically from bucket
            risk_unit_kwargs = {
                "risk_unit_id": risk_unit_id,
                "sample_unit_ids": bucket["sample_unit_ids"],
                "inspection_unit_ids": bucket["inspection_unit_ids"],
                "plant_ids": bucket["plant_ids"],
            }

            # Add all configured attributes from the bucket
            for attr_name in self.risk_unit_config.enabled_attributes:
                if attr_name in bucket:
                    risk_unit_kwargs[attr_name] = bucket[attr_name]

            risk_unit = RiskUnit(
                np.zeros(len(bucket["sample_unit_ids"]), dtype=np.int64),
                **risk_unit_kwargs
            )
            risk_units.append(risk_unit)

        # Create overall arrays for the entire consignment
        consignment_sample_units = np.zeros(total_sample_units, dtype=np.int64)
        consignment_plants = np.zeros(total_plants, dtype=np.int64)
        
        # Calculate average plants per sample_unit for consignment-level tracking
        if total_sample_units > 0:
            avg_plants_per_sample_unit = total_plants // total_sample_units
        else:
            avg_plants_per_sample_unit = 1

        # Use CREATED_DATETIME from PIS data if available, otherwise use default date
        if 'CREATED_DATETIME' in first_record and first_record['CREATED_DATETIME'] is not None and str(first_record['CREATED_DATETIME']).strip():
            try:
                # Convert MM:SS.S timestamp to a datetime object
                # Use base date with the timestamp as time offset
                timestamp = str(first_record['CREATED_DATETIME']).strip()
                minutes, seconds_tenths = timestamp.split(':')
                seconds, tenths = seconds_tenths.split('.')
                
                # Create datetime with base date but using timestamp as time offset
                base_date = datetime(2020, 1, 1)  # Base date for consistency
                # Convert tenths to microseconds: 1 tenth = 100,000 microseconds
                microseconds = int(tenths) * 100000
                time_offset = timedelta(minutes=int(minutes), seconds=int(seconds), microseconds=microseconds)
                date = base_date + time_offset
            except (ValueError, KeyError, AttributeError):
                # Fallback to default date if timestamp parsing fails
                date = datetime.strptime("2020-01-01", "%Y-%m-%d")
        else:
            # Use default date if no CREATED_DATETIME column
            date = datetime.strptime("2020-01-01", "%Y-%m-%d")

        # Calculate average sample_units_per_inspection_unit for compatibility
        if len(inspection_units) > 0:
            avg_sample_units_per_inspection_unit = total_sample_units // len(inspection_units)
        else:
            avg_sample_units_per_inspection_unit = 1

        # Determine consignment-level material_type
        unique_material_types = set(iu.material_type for iu in inspection_units)
        if len(unique_material_types) > 1:
            consignment_material_type = f"Mixed ({len(unique_material_types)} types)"
        else:
            consignment_material_type = next(iter(unique_material_types), None)

        return Consignment(
            num_sample_units=total_sample_units,
            sample_units=consignment_sample_units,
            sample_units_per_inspection_unit=avg_sample_units_per_inspection_unit,  # Average for compatibility
            num_inspection_units=len(inspection_units),
            date=date,
            inspection_units=inspection_units,
            origin=origin,
            port=port,
            pathway=pathway,
            material_type=consignment_material_type,
            num_plants=total_plants,
            plants=consignment_plants,
            plants_per_sample_unit=avg_plants_per_sample_unit,
            inspection_number=inspection_number,
            risk_units=risk_units,
            sample_unit_objects=all_sample_unit_objects,
            risk_unit_to_inspection_units={
                risk_unit.id: list(risk_unit.inspection_unit_ids) for risk_unit in risk_units
            },
            inspection_unit_to_risk_units={
                inspection_unit.id: list(inspection_unit.risk_unit_ids)
                for inspection_unit in inspection_units
                if inspection_unit.id is not None
            },
            risk_unit_to_sample_units={
                risk_unit.id: list(risk_unit.sample_unit_ids) for risk_unit in risk_units
            },
            sample_unit_to_inspection_unit={
                sample_unit.id: sample_unit.inspection_unit_id
                for sample_unit in all_sample_unit_objects
                if sample_unit.id is not None
            },
            plant_unit_to_inspection_unit={
                plant_id: inspection_unit.id
                for inspection_unit in inspection_units
                for plant_id in inspection_unit.plant_ids
            },
        )


class AQIMConsignmentGenerator:
    """Generate a consignments based on existing AQIM records"""

    def __init__(self, sample_units_per_inspection_unit, filename, separator=","):
        self.infile = open(filename)
        self.reader = csv.DictReader(self.infile, delimiter=separator)
        self.sample_units_per_inspection_unit = sample_units_per_inspection_unit

    def generate_consignment(self):
        """Generate a new consignment"""
        try:
            record = next(self.reader)
        except StopIteration:
            raise RuntimeError(
                "More consignments requested than number of records in AQIM data"
            ) from None
        pathway = record["CARGO_FORM"]
        sample_units_per_inspection_unit = self.sample_units_per_inspection_unit
        sample_units_per_inspection_unit = get_sample_units_per_inspection_unit(sample_units_per_inspection_unit, pathway)
        unit = record["UNIT"]

        # Generate sample_units based on quantity in AQIM records.
        # If quantity is given in inspection_units, use item_per_inspection_unit to convert to sample_units.
        if unit in ["box/Carton"]:
            num_sample_units = int(record["QUANTITY"]) * sample_units_per_inspection_unit
        elif unit in ["Stems"]:
            num_sample_units = int(record["QUANTITY"])
        else:
            raise RuntimeError(f"Unsupported quantity unit: {unit}")

        sample_units = np.zeros(num_sample_units, dtype=np.int64)

        # rounding up to keep the max per inspection_unit and have enough inspection_units
        num_inspection_units = int(math.ceil(sample_units_per_inspection_unit / float(sample_units_per_inspection_unit)))
        num_inspection_units = max(num_inspection_units, 1)
        inspection_units = []
        for i in range(num_inspection_units):
            lower = i * sample_units_per_inspection_unit
            # slicing does not go over the size even if our last inspection_unit is smaller
            upper = (i + 1) * sample_units_per_inspection_unit
            inspection_units.append(InspectionUnit(sample_units[lower:upper], material_type=record["COMMODITY_LIST"]))
        assert sum([inspection_unit.sample_units_per_inspection_unit for inspection_unit in inspection_units]) == sample_units_per_inspection_unit

        date = record["CALENDAR_YR"]
        return Consignment(
            material_type=record["COMMODITY_LIST"],
            num_sample_units=num_sample_units,
            sample_units=sample_units,
            sample_units_per_inspection_unit=sample_units_per_inspection_unit,
            num_inspection_units=num_inspection_units,
            date=date,
            inspection_units=inspection_units,
            origin=record["ORIGIN"],
            port=record["LOCATION"],
            pathway=pathway,
        )


def get_sample_units_per_inspection_unit(sample_units_per_inspection_unit, pathway):
    """Based on config and pathway, return number of sample_units per inspection_unit."""
    if pathway.lower() == "airport" and "air" in sample_units_per_inspection_unit:
        sample_units_per_inspection_unit = sample_units_per_inspection_unit["air"]["default"]
    elif pathway.lower() == "maritime" and "maritime" in sample_units_per_inspection_unit:
        sample_units_per_inspection_unit = sample_units_per_inspection_unit["maritime"]["default"]
    else:
        sample_units_per_inspection_unit = sample_units_per_inspection_unit["default"]
    return sample_units_per_inspection_unit


def get_plants_per_sample_unit(plants_per_sample_unit, pathway):
    """Based on config and pathway, return number of sample_units per inspection_unit."""
    if pathway.lower() == "airport" and "air" in plants_per_sample_unit:
        plants_per_sample_unit = plants_per_sample_unit["air"]["default"]
    elif pathway.lower() == "maritime" and "maritime" in plants_per_sample_unit:
        plants_per_sample_unit = plants_per_sample_unit["maritime"]["default"]
    else:
        plants_per_sample_unit = plants_per_sample_unit["default"]
    return plants_per_sample_unit


def get_consignment_generator(config):
    """Based on config, return consignment generator object."""
    config = config["consignment"]
    generation_method = config["generation_method"]
    if (generation_method == "input_file") and (
        config["input_file"]["file_type"] == "F280"
    ):
        # Backward compatibility: check for both old and new terminology
        sample_units_config = config.get("sample_units_per_inspection_unit", config.get("items_per_box"))
        consignment_generator = F280ConsignmentGenerator(
            sample_units_per_inspection_unit=sample_units_config,
            filename=config["input_file"]["file_name"],
        )
    elif (generation_method == "input_file") and (
        config["input_file"]["file_type"] == "AQIM"
    ):
        consignment_generator = AQIMConsignmentGenerator(
            sample_units_per_inspection_unit=config["items_per_box"],
            filename=config["input_file"]["file_name"],
        )
    elif generation_method == "parameter_based":
        start_date = config.get("start_date", "2020-01-01")
        consignment_generator = ParameterConsignmentGenerator(
            parameters=config["parameter_based"],
            sample_units_per_inspection_unit=config["items_per_box"],
            start_date=start_date,
        )
    elif (generation_method == "input_file") and (
        config["input_file"]["file_type"] == "PIS"
    ):
        consignment_generator = PISConsignmentGenerator(
            filename=config["input_file"]["file_name"],
        )
    elif generation_method == "RBS":
        if "input_file" in config and "rbs_file_name" in config["input_file"]:
        # RBS record-based generation from file
            consignment_generator = PISConsignmentGenerator(
                filename=config["input_file"]["rbs_file_name"],
            )
        else:
            print("No consignment data available")
    else:
        raise RuntimeError(
            f"Unknown consignment generation method: {generation_method}"
        )
    return consignment_generator

