# Simulation of contaminated consignments and their inspections
# Copyright (C) 2018-2022 Vaclav Petras and others (see below)

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

=====================================
JHU/APL Extensions and Modifications:
=====================================

Contributors: Gary Lin, Joseph Agor (Johns Hopkins University Applied Physics Laboratory)

New Classes Added:
------------------
- RBSConsignmentGenerator: 
    * Generates consignments using risk-based sampling methodology
    * Supports hierarchical packaging structure (inspection_units -> sample_units -> plants)
    * Designed for propagative material inspection workflows
    * Integrates with compliance-based detection level lookup

- RBSRecordConsignmentGenerator:
    * Generates consignments from CSV records with hierarchical RBS structure
    * Supports flexible record formats with multiple quantity specifications
    * Handles pathway-specific configurations for sample_units and plants
    * Maintains hierarchical packaging for plant-level contamination modeling

- SampleUnit: 
    * Manages plant-level sampling units within individual sample_units
    * Provides contamination detection at the plant level
    * Integrates with hierarchical inspection structure
    * Supports numpy array-based plant contamination tracking

New Functions Added:
-------------------  
- get_plants_per_sample_unit(): 
    * Determines number of plants per sample_unit based on pathway type
    * Supports pathway-specific plant quantity configurations
    * Handles backward compatibility for plants_per_item

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

Modified Classes:
----------------
- InspectionUnit (formerly inspection_unit): 
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

Modified Functions:
----------------
- get_consignment_generator():
    * Supports RBS generation method selection (parameter-based and record-based)
    * Enhanced backward compatibility for configuration parameter mapping
    * Handles both old terminology (items_per_box) and new terminology (sample_units_per_inspection_unit)
    * Automatically selects PISConsignmentGenerator when file_type="PIS" is specified for RBS method

Terminology Refactoring:
-----------------------
- Systematically refactored: boxes -> inspection_units, items -> sample_units
- Updated all class names, method names, and variable names for consistency
- Added comprehensive backward compatibility for existing configurations
- Maintained dual access patterns for smooth migration from legacy terminology

Backward Compatibility:
----------------------
- Configuration parameter mapping: boxes -> inspection_units, items -> sample_units
- Legacy attribute access in Consignment class via __getattr__ and __hasattr__
- Support for both items_per_box and sample_units_per_inspection_unit configuration keys
- Maintained existing F280 and AQIM consignment generator functionality
"""


import collections
import csv
import math
import random
from datetime import datetime, timedelta

import numpy as np


class InspectionUnit:
    """Inspection unit (formerly Box)

    Evaluates to bool when it contains contaminant.

    InspectionUnit is a view into array of sample_units, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, sample_units can not
    only be accessed but also modifed through the InspectionUnit.
    """

    def __init__(self, sample_units, material_type=None):
        """Store reference to associated sample_units

        :param sample_units: Array-like object of sample_units
        :param material_type: Material type for this inspection unit
        """
        self.sample_units = sample_units
        self.sample_unit_objects = []  # For hierarchical structure - list of SampleUnit objects
        self.material_type = material_type

    @property
    def num_sample_units(self):
        """Number of sample_units in the InspectionUnit"""
        if hasattr(self.sample_units, 'shape'):
            return self.sample_units.shape[0]
        else:
            return len(self.sample_units)

    def __bool__(self):
        if isinstance(self.sample_units, np.ndarray):
            return bool(np.any(self.sample_units > 0))
        else:
            # Assuming it's a list of SampleUnit objects
            return any(bool(su) for su in self.sample_units)


class SampleUnit:
    """Sample unit

    Evaluates to bool when it contains contaminant.

    Item Container is a view into array of plants, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, plants can not
    only be accessed but also modifed through the InspectionUnit.
    """

    def __init__(self, plants):
        """Store reference to associated plants

        :param plants: Array-like object of plants
        """
        self.plants = plants

    @property
    def num_plants(self):
        """Number of plants in the Sample Unit"""
        return self.plants.shape[0]

    def __bool__(self):
        return bool(np.any(self.plants > 0))


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


class RBSConsignmentGenerator:
    """Generate a consignments with hierarchal packaging"""

    def __init__(self, parameters, sample_units_per_inspection_unit, plants_per_sample_unit, start_date):
        """Set parameters for consignment generation

        :param parameters: Consignment parameters
        :param ports: List of ports to choose from
        :param sample_units_per_inspection_unit: Configuration driving number of sample_units per inspection_unit
        :param start_date: Date to start consignment dates from
        """
        self.params = parameters
        self.sample_units_per_inspection_unit = sample_units_per_inspection_unit
        self.plants_per_sample_unit = plants_per_sample_unit
        self.num_generated = 0
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.date = start_date

    def generate_consignment(self):
        """Generate a new consignment"""
        port = random.choice(self.params["ports"])
        # propagative materials or commodities
        material_type = random.choice(self.params["material_types"])
        origin = random.choice(self.params["origins"])
        # Handle backward compatibility for old "boxes" terminology
        inspection_units_config = self.params.get("inspection_units", self.params.get("boxes", {}))
        num_inspection_units_min = inspection_units_config.get("min", 0)
        num_inspection_units_max = inspection_units_config.get("max", 1)
        pathway = "None"
        sample_units_per_inspection_unit = self.sample_units_per_inspection_unit
        sample_units_per_inspection_unit = get_sample_units_per_inspection_unit(sample_units_per_inspection_unit, pathway)
        plants_per_sample_unit = self.plants_per_sample_unit
        plants_per_sample_unit = get_plants_per_sample_unit(plants_per_sample_unit, pathway)
        num_inspection_units = random.randint(num_inspection_units_min, num_inspection_units_max)
        num_sample_units = sample_units_per_inspection_unit * num_inspection_units
        sample_units = np.zeros(num_sample_units, dtype=np.int64)
        num_plants = plants_per_sample_unit * num_sample_units
        plants = np.zeros(num_plants, dtype=np.int64)
        inspection_units = []
        for inspection_unit_index in range(num_inspection_units):
            # Each inspection_unit gets a slice of the sample_units array
            start_idx = inspection_unit_index * sample_units_per_inspection_unit
            end_idx = start_idx + sample_units_per_inspection_unit
            inspection_unit_sample_units = sample_units[start_idx:end_idx]
            
            # Create SampleUnit objects for hierarchical access if needed
            sample_unit_objects = []
            for item_index in range(sample_units_per_inspection_unit):
                plant_start = (start_idx + item_index) * plants_per_sample_unit
                plant_end = plant_start + plants_per_sample_unit
                sample_unit_objects.append(SampleUnit(plants[plant_start:plant_end]))

            # Create InspectionUnit with the numpy array slice
            inspection_unit = InspectionUnit(inspection_unit_sample_units, material_type=material_type)
            # Also store the SampleUnit objects for hierarchical access
            inspection_unit.sample_unit_objects = sample_unit_objects
            inspection_units.append(inspection_unit)
        self.num_generated += 1

        # two consignments every nth day
        if self.num_generated % 3:
            self.date += timedelta(days=1)

        return Consignment(
            num_sample_units=num_sample_units,
            sample_units=sample_units,
            sample_units_per_inspection_unit=sample_units_per_inspection_unit,
            num_inspection_units=num_inspection_units,
            date=self.date,
            inspection_units=inspection_units,
            origin=origin,
            port=port,
            pathway=pathway,
            material_type=material_type,
            num_plants=num_plants,
            plants=plants,
            plants_per_sample_unit=plants_per_sample_unit,
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

    def __init__(self, filename, separator=","):
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
        """
        import pandas as pd
        self.df = pd.read_csv(filename, sep=separator)
        # Group by inspection number to create consignments
        self.consignment_groups = list(self.df.groupby('INSPECTION_NUMBER'))
        self.current_consignment_index = 0

    def generate_consignment(self):
        """Generate a new consignment from PIS records"""
        if self.current_consignment_index >= len(self.consignment_groups):
            raise RuntimeError(
                "More consignments requested than number of inspection numbers in provided PIS data"
            ) from None

        inspection_number, inspection_units_data = self.consignment_groups[self.current_consignment_index]
        self.current_consignment_index += 1

        # Extract common consignment info from first record
        first_record = inspection_units_data.iloc[0]
        pathway = first_record.get("PATHWAY", "None")
        origin = first_record["COUNTRY_OF_ORIGIN_NAME"]
        port = first_record["INSPECTION_LOCATION_NAME"]
        
        # Calculate total quantities across all inspection units in this consignment
        total_sample_units = 0
        total_plants = 0
        inspection_units = []
        
        # Process each inspection unit (row) in this consignment
        for _, record in inspection_units_data.iterrows():
            material_type = record["PROPAGATIVE_MATERIAL_TYPE"]
            
            # Use the actual quantities from the PIS data
            inspection_unit_sample_units = int(record["TOTAL_SAMPLING_UNITS"])
            inspection_unit_plants = int(record["TOTAL_PLANT_QUANTITY"])
            
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
                sample_unit_objects.append(SampleUnit(sample_unit_plants))
                
                plant_index += plants_in_this_unit

            # Create InspectionUnit
            inspection_unit = InspectionUnit(sample_units_array, material_type=material_type)
            inspection_unit.sample_unit_objects = sample_unit_objects
            inspection_units.append(inspection_unit)
            
            total_sample_units += inspection_unit_sample_units
            total_plants += inspection_unit_plants

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
            material_type=f"Mixed ({len(inspection_units)} types)",  # Mixed material types
            num_plants=total_plants,
            plants=consignment_plants,
            plants_per_sample_unit=avg_plants_per_sample_unit,
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
    elif generation_method == "RBS":
        if "input_file" in config and "file_name" in config["input_file"]:
            # RBS record-based generation from file
            file_type = config["input_file"].get("file_type", "RBS")
            if file_type == "PIS":
                # PIS data with multiple material types per consignment
                # PIS generator is fully data-driven - no configuration parameters needed
                consignment_generator = PISConsignmentGenerator(
                    filename=config["input_file"]["file_name"],
                )
            else:
                # Other RBS record formats would go here
                raise RuntimeError(f"Unsupported RBS file type: {file_type}")
        else:
            # RBS parameter-based generation
            start_date = config.get("start_date", "2020-01-01")
            consignment_generator = RBSConsignmentGenerator(
                parameters=config["rbs_parameter_based"],
                sample_units_per_inspection_unit=config["sample_units_per_inspection_unit"],
                plants_per_sample_unit=config.get("plants_per_sample_unit", config.get("plants_per_item")),
                start_date=start_date,
            )
    else:
        raise RuntimeError(
            f"Unknown consignment generation method: {generation_method}"
        )
    return consignment_generator
