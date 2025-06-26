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
"""

import collections
import csv
import math
import random
from datetime import datetime, timedelta

import numpy as np


class Box:
    """Box or inspection unit

    Evaluates to bool when it contains contaminant.

    Box is a view into array of items, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, items can not
    only be accessed but also modifed through the box.
    """

    def __init__(self, items, sampleunit=None):
        """Store reference to associated items

        :param items: Array-like object of items
        """
        self.items = items
        self.sampleunit = sampleunit  # extended with plant unit

    @property
    def num_items(self):
        """Number of items in the box"""
        return self.items.shape[0]

    def __bool__(self):
        if self.sampleunit is None:
            return bool(np.any(self.items > 0))
        else:
            return any(bool(su) for su in self.sampleunit)


class SampleUnit:
    """Sample unit

    Evaluates to bool when it contains contaminant.

    Item Container is a view into array of plants, i.e. a slice of that array. The
    assumption is that the original, and possibly modifed, plants can not
    only be accessed but also modifed through the box.
    """

    def __init__(self, plants):
        """Store reference to associated plants

        :param plants: Array-like object of plants
        """
        self.plants = plants

    @property
    def num_plants(self):
        """Number of plants in the box"""
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
        flower,
        num_items,
        items,
        items_per_box,
        num_boxes,
        date,
        boxes,
        origin,
        port,
        pathway,
        num_plants=None,
        plants=None,
        plants_per_item=None,
    ):
        """Store reference to associated attributes

        :param flower: string
        :param num_items: integer
        :param items_per_box: integer
        :param num_boxes: integer
        :param date: Array-like object of dates
        :param boxes: Array-like object of boxes
        :param origin: string
        :param port: string
        :param pathway: string
        :param num_plants : string (optional)
        :param plants : Array-like object of plants (optional)
        :param plants_per_item : string (optional)
        """
        super().__init__(
            flower=flower,
            num_items=num_items,
            items=items,
            items_per_box=items_per_box,
            num_boxes=num_boxes,
            date=date,
            boxes=boxes,
            origin=origin,
            port=port,
            pathway=pathway,
            num_plants=num_plants,
            plants=plants,
            plants_per_item=plants_per_item,
        )
        self.flower = flower
        self.num_items = num_items
        self.items = items
        self.items_per_box = items_per_box
        self.num_boxes = num_boxes
        self._date = date
        self.boxes = boxes
        self.origin = origin
        self.port = port
        self.pathway = pathway
        self.num_plants = num_plants
        self.plants = plants
        self.plants_per_item = plants_per_item

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
            """Count contaminated items in box."""
            return np.count_nonzero(self.items)
        else:
            """Count contaminated plants in box."""
            return np.count_nonzero(self.plants)

    def item_in_box_to_item_index(self, box_index, item_in_box_index):
        """Convert item index in a box to item index in the consignment"""
        if box_index == 0:
            return item_in_box_index
        items = 0
        for box in self.boxes[:box_index]:
            items += box.num_items
        return items + item_in_box_index
    
    def get_box_and_sampleunit_index(self, item_index):
        box_index = item_index // self.items_per_box
        sampleunit_index = item_index % self.items_per_box
        return box_index, sampleunit_index


class ParameterConsignmentGenerator:
    """Generate a consignments based on configuration parameters"""

    def __init__(self, parameters, items_per_box, start_date):
        """Set parameters for consignment generation

        :param parameters: Consignment parameters
        :param ports: List of ports to choose from
        :param items_per_box: Configuration driving number of items per box
        :param start_date: Date to start consignment dates from
        """
        self.params = parameters
        self.items_per_box = items_per_box
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
        num_boxes_min = self.params["boxes"].get("min", 0)
        num_boxes_max = self.params["boxes"]["max"]
        pathway = "None"
        items_per_box = self.items_per_box
        items_per_box = get_items_per_box(items_per_box, pathway)
        num_boxes = random.randint(num_boxes_min, num_boxes_max)
        num_items = items_per_box * num_boxes
        items = np.zeros(num_items, dtype=np.int64)
        boxes = []
        for i in range(num_boxes):
            lower = i * items_per_box
            upper = (i + 1) * items_per_box
            boxes.append(Box(items[lower:upper]))
        self.num_generated += 1
        # two consignments every nth day
        if self.num_generated % 3:
            self.date += timedelta(days=1)

        return Consignment(
            flower=flower,
            num_items=num_items,
            items=items,
            items_per_box=items_per_box,
            num_boxes=num_boxes,
            date=self.date,
            boxes=boxes,
            origin=origin,
            port=port,
            pathway=pathway,
        )


class HierarchalConsignmentGenerator:
    """Generate a consignments with hierarchal packaging"""

    def __init__(self, parameters, items_per_box, plants_per_item, start_date):
        """Set parameters for consignment generation

        :param parameters: Consignment parameters
        :param ports: List of ports to choose from
        :param items_per_box: Configuration driving number of items per box
        :param start_date: Date to start consignment dates from
        """
        self.params = parameters
        self.items_per_box = items_per_box
        self.plants_per_item = plants_per_item
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
        num_boxes_min = self.params["boxes"].get("min", 0)
        num_boxes_max = self.params["boxes"]["max"]
        pathway = "None"
        items_per_box = self.items_per_box
        items_per_box = get_items_per_box(items_per_box, pathway)
        plants_per_item = self.plants_per_item
        plants_per_item = get_plants_per_item(plants_per_item, pathway)
        num_boxes = random.randint(num_boxes_min, num_boxes_max)
        num_items = items_per_box * num_boxes
        items = np.zeros(num_items, dtype=np.int64)
        num_plants = plants_per_item * num_items
        plants = np.zeros(num_plants, dtype=np.int64)
        boxes = []
        for box_index in range(num_boxes):
            sampleunits = []
            item_start = box_index * items_per_box
            item_end = item_start + items_per_box

            for item_index in range(items_per_box):
                plant_start = (item_start + item_index) * plants_per_item
                plant_end = plant_start + plants_per_item

                sampleunits.append(SampleUnit(plants[plant_start:plant_end]))

            boxes.append(Box(items[item_start:item_end], sampleunits))
        self.num_generated += 1
        # two consignments every nth day
        if self.num_generated % 3:
            self.date += timedelta(days=1)

        return Consignment(
            flower=flower,
            num_items=num_items,
            items=items,
            items_per_box=items_per_box,
            num_boxes=num_boxes,
            date=self.date,
            boxes=boxes,
            origin=origin,
            port=port,
            pathway=pathway,
            num_plants=num_plants,
            plants=plants,
            plants_per_item=plants_per_item,
        )


class F280ConsignmentGenerator:
    """Generate a consignments based on existing F280 records"""

    def __init__(self, items_per_box, filename, separator=","):
        self.infile = open(filename)
        self.reader = csv.DictReader(self.infile, delimiter=separator)
        self.items_per_box = items_per_box

    def generate_consignment(self):
        """Generate a new consignment"""
        try:
            record = next(self.reader)
        except StopIteration:
            raise RuntimeError(
                "More consignments requested than number of records in provided F280"
            ) from None

        num_items = int(record["QUANTITY"])
        items = np.zeros(num_items, dtype=np.int64)

        pathway = record["PATHWAY"]
        items_per_box = self.items_per_box
        items_per_box = get_items_per_box(items_per_box, pathway)

        # rounding up to keep the max per box and have enough boxes
        num_boxes = int(math.ceil(num_items / float(items_per_box)))
        num_boxes = max(num_boxes, 1)
        boxes = []
        for i in range(num_boxes):
            lower = i * items_per_box
            # slicing does not go over the size even if our last box is smaller
            upper = (i + 1) * items_per_box
            boxes.append(Box(items[lower:upper]))
        assert sum([box.num_items for box in boxes]) == num_items

        date = datetime.strptime(record["REPORT_DT"], "%Y-%m-%d")
        return Consignment(
            flower=record["COMMODITY"],
            num_items=num_items,
            items=items,
            items_per_box=items_per_box,
            num_boxes=num_boxes,
            date=date,
            boxes=boxes,
            origin=record["ORIGIN_NM"],
            port=record["LOCATION"],
            pathway=pathway,
        )


class AQIMConsignmentGenerator:
    """Generate a consignments based on existing AQIM records"""

    def __init__(self, items_per_box, filename, separator=","):
        self.infile = open(filename)
        self.reader = csv.DictReader(self.infile, delimiter=separator)
        self.items_per_box = items_per_box

    def generate_consignment(self):
        """Generate a new consignment"""
        try:
            record = next(self.reader)
        except StopIteration:
            raise RuntimeError(
                "More consignments requested than number of records in AQIM data"
            ) from None
        pathway = record["CARGO_FORM"]
        items_per_box = self.items_per_box
        items_per_box = get_items_per_box(items_per_box, pathway)
        unit = record["UNIT"]

        # Generate items based on quantity in AQIM records.
        # If quantity is given in boxes, use item_per_box to convert to items.
        if unit in ["Box/Carton"]:
            num_items = int(record["QUANTITY"]) * items_per_box
        elif unit in ["Stems"]:
            num_items = int(record["QUANTITY"])
        else:
            raise RuntimeError(f"Unsupported quantity unit: {unit}")

        items = np.zeros(num_items, dtype=np.int64)

        # rounding up to keep the max per box and have enough boxes
        num_boxes = int(math.ceil(num_items / float(items_per_box)))
        num_boxes = max(num_boxes, 1)
        boxes = []
        for i in range(num_boxes):
            lower = i * items_per_box
            # slicing does not go over the size even if our last box is smaller
            upper = (i + 1) * items_per_box
            boxes.append(Box(items[lower:upper]))
        assert sum([box.num_items for box in boxes]) == num_items

        date = record["CALENDAR_YR"]
        return Consignment(
            flower=record["COMMODITY_LIST"],
            num_items=num_items,
            items=items,
            items_per_box=items_per_box,
            num_boxes=num_boxes,
            date=date,
            boxes=boxes,
            origin=record["ORIGIN"],
            port=record["LOCATION"],
            pathway=pathway,
        )


def get_items_per_box(items_per_box, pathway):
    """Based on config and pathway, return number of items per box."""
    if pathway.lower() == "airport" and "air" in items_per_box:
        items_per_box = items_per_box["air"]["default"]
    elif pathway.lower() == "maritime" and "maritime" in items_per_box:
        items_per_box = items_per_box["maritime"]["default"]
    else:
        items_per_box = items_per_box["default"]
    return items_per_box


def get_plants_per_item(plants_per_item, pathway):
    """Based on config and pathway, return number of items per box."""
    if pathway.lower() == "airport" and "air" in plants_per_item:
        plants_per_item = plants_per_item["air"]["default"]
    elif pathway.lower() == "maritime" and "maritime" in plants_per_item:
        plants_per_item = plants_per_item["maritime"]["default"]
    else:
        plants_per_item = plants_per_item["default"]
    return plants_per_item


def get_consignment_generator(config):
    """Based on config, return consignment generator object."""
    config = config["consignment"]
    generation_method = config["generation_method"]
    if (generation_method == "input_file") and (
        config["input_file"]["file_type"] == "F280"
    ):
        consignment_generator = F280ConsignmentGenerator(
            items_per_box=config["items_per_box"],
            filename=config["input_file"]["file_name"],
        )
    elif (generation_method == "input_file") and (
        config["input_file"]["file_type"] == "AQIM"
    ):
        consignment_generator = AQIMConsignmentGenerator(
            items_per_box=config["items_per_box"],
            filename=config["input_file"]["file_name"],
        )
    elif generation_method == "parameter_based":
        start_date = config.get("start_date", "2020-01-01")
        consignment_generator = ParameterConsignmentGenerator(
            parameters=config["parameter_based"],
            items_per_box=config["items_per_box"],
            start_date=start_date,
        )
    elif generation_method == "hierarchal":
        start_date = config.get("start_date", "2020-01-01")
        consignment_generator = HierarchalConsignmentGenerator(
            parameters=config["parameter_based"],
            items_per_box=config["items_per_box"],
            plants_per_item=config["plants_per_item"],
            start_date=start_date,
        )
    else:
        raise RuntimeError(
            f"Unknown consignment generation method: {generation_method}"
        )
    return consignment_generator
