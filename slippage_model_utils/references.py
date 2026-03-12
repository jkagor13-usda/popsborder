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

from difflib import get_close_matches
from typing import List

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

# risk_unit_attributes = {
# sample_units,
#                  risk_unit_id=None,
#                  sample_unit_ids=None,
#                  inspection_unit_ids=None,
#                  plant_ids=None,
#                  material_type=None,
#                  producer=None,
#                  origin=None,
#                  port=None,
#                  pathway=None,
# "Binary1"
# }

COLUMN_NAME_ALIASES = {
    'comm_ID': [
        'comm_id', 'commid', 'comm id', 'commodity_id', 'commodity id',
        'commodityid', 'comm', 'commodity identifier'
    ],
    'INSPECTION_ID': [
        'inspection_id', 'inspectionid', 'inspection id', 'insp_id',
        'insp id', 'inspid', 'inspection identifier', 'inspection number id'
    ],
    'REF_COMMODITY_ID': [
        'ref_commodity_id', 'refcommodityid', 'ref commodity id',
        'reference_commodity_id', 'reference commodity id', 'ref comm id',
        'ref_comm_id', 'commodity reference', 'commodity ref'
    ],
    'CERTIFIED_FACILITY_NAME': [
        'certified_facility_name', 'certified facility name',
        'facility_name', 'facility name', 'certified_facility',
        'certified facility', 'cert facility', 'cert_facility_name'
    ],
    'CERTIFIED_FACILITY_NUMBER': [
        'certified_facility_number', 'certified facility number',
        'facility_number', 'facility number', 'facility_num',
        'facility num', 'cert facility number', 'cert_facility_num'
    ],
    'COMMODITY_CLASSIFICATION': [
        'commodity_classification', 'commodity classification',
        'classification', 'commodity_class', 'commodity class',
        'comm_classification', 'comm classification', 'product classification'
    ],
    'COMMODITY_COMMON_NAME': [
        'commodity_common_name', 'commodity common name',
        'common_name', 'common name', 'commodity_name', 'commodity name',
        'product_name', 'product name', 'commodity', 'item name'
    ],
    'COMMODITY_DISPLAY_NAME': [
        'commodity_display_name', 'commodity display name',
        'display_name', 'display name', 'comm_display_name',
        'comm display name', 'product display name'
    ],
    'COMMODITY_TAXONOMIC_DISPLAY_NAME': [
        'commodity_taxonomic_display_name', 'commodity taxonomic display name',
        'taxonomic_display_name', 'taxonomic display name',
        'taxonomic_name', 'taxonomic name', 'scientific display name'
    ],
    'COMMODITY_HOST_TYPE': [
        'commodity_host_type', 'commodity host type', 'host_type',
        'host type', 'host', 'comm_host_type', 'comm host type'
    ],
    'COMMODITY_TYPE': [
        'commodity_type', 'commodity type', 'comm_type', 'comm type',
        'product_type', 'product type', 'type', 'item type'
    ],
    'COUNTRY_OF_ORIGIN_NAME': [
        'country_of_origin_name', 'country of origin name',
        'country_of_origin', 'country of origin', 'origin_country',
        'origin country', 'country', 'origin', 'source_country',
        'source country', 'coo', 'origin name'
    ],
    'CONSIGNEE_NAME': [
        'consignee_name', 'consignee name', 'consignee', 'receiver_name',
        'receiver name', 'receiver', 'recipient', 'recipient_name',
        'recipient name'
    ],
    'DESTINATION_STATE_NAME': [
        'destination_state_name', 'destination state name',
        'destination_state', 'destination state', 'dest_state',
        'dest state', 'state', 'destination', 'target state'
    ],
    'DISPOSITION_CODE': [
        'disposition_code', 'disposition code', 'disposition',
        'disp_code', 'disp code', 'result_code', 'result code',
        'outcome_code', 'outcome code'
    ],
    'GENUS_NAME': [
        'genus_name', 'genus name', 'genus', 'taxonomic_genus',
        'taxonomic genus', 'scientific genus'
    ],
    'ENTRY_NUMBER': [
        'entry_number', 'entry number', 'entry_num', 'entry num',
        'entry', 'entry_id', 'entry id', 'entry identifier'
    ],
    'ENTRY_LINE_NUMBER': [
        'entry_line_number', 'entry line number', 'entry_line_num',
        'entry line num', 'line_number', 'line number', 'entry_line',
        'entry line', 'line_num', 'line num'
    ],
    'PGA_LINE_NUMBER': [
        'pga_line_number', 'pga line number', 'pga_line_num',
        'pga line num', 'pga_line', 'pga line', 'pga number'
    ],
    'PRODUCER_ID': [
        'producer_id', 'producer id', 'producerid', 'grower_id',
        'grower id', 'growerid', 'producer identifier'
    ],
    'PRODUCER_NAME': [
        'producer_name', 'producer name', 'producer', 'grower_name',
        'grower name', 'grower', 'manufacturer', 'supplier_name',
        'supplier name', 'supplier'
    ],
    'PROPAGATIVE_MATERIAL_TYPE': [
        'propagative_material_type', 'propagative material type',
        'material_type', 'material type', 'prop_material_type',
        'prop material type', 'propagative_type', 'propagative type',
        'prop_type', 'prop type', 'material'
    ],
    'QUANTITY': [
        'quantity', 'qty', 'amount', 'count', 'volume', 'number',
        'total', 'total quantity'
    ],
    'QUANTITY_UNITS_NAME': [
        'quantity_units_name', 'quantity units name', 'quantity_units',
        'quantity units', 'units_name', 'units name', 'units', 'uom',
        'unit_of_measure', 'unit of measure'
    ],
    'WADS_CODE': [
        'wads_code', 'wads code', 'wads', 'wads_id', 'wads id'
    ],
    'SAMPLING_UNITS': [
        'sampling_units', 'sampling units', 'samples', 'sample_units',
        'sample units', 'sampling', 'units sampled'
    ],
    'IS_RBS': [
        'is_rbs', 'is rbs', 'isrbs', 'rbs_flag', 'rbs flag',
        'rbs_indicator', 'rbs indicator', 'rbs'
    ],
    'RBS_STATUS': [
        'rbs_status', 'rbs status', 'rbs', 'risk_based_status',
        'risk based status', 'rbs_flag', 'rbs flag'
    ],
    'GROWING_MEDIA_PRESENCE': [
        'growing_media_presence', 'growing media presence',
        'media_presence', 'media presence', 'growing_media',
        'growing media', 'soil_presence', 'soil presence', 'media'
    ],
    'CREATED_DATETIME': [
        'created_datetime', 'created datetime', 'created_date',
        'created date', 'creation_date', 'creation date',
        'date_created', 'date created', 'created', 'creation_time',
        'creation time'
    ],
    'INSPECTION_DATETIME': [
        'inspection_datetime', 'inspection datetime', 'inspection_date',
        'inspection date', 'insp_datetime', 'insp datetime',
        'insp_date', 'insp date', 'date_inspected', 'date inspected'
    ],
    'BROKER_NAME': [
        'broker_name', 'broker name', 'broker', 'agent_name',
        'agent name', 'agent', 'intermediary', 'broker agent'
    ],
    'CATEGORY': [
        'category', 'cat', 'main_category', 'main category',
        'primary_category', 'primary category', 'type'
    ],
    'SUBCATEGORY': [
        'subcategory', 'sub_category', 'sub category', 'subcat',
        'sub_cat', 'sub cat', 'secondary_category', 'secondary category'
    ],
    'IMPORTER_NAME': [
        'importer_name', 'importer name', 'importer', 'import_company',
        'import company', 'importing_company', 'importing company'
    ],
    'INSPECTION_LOCATION_NAME': [
        'inspection_location_name', 'inspection location name',
        'location_name', 'location name', 'inspection_location',
        'inspection location', 'location', 'site_name', 'site name',
        'facility_name', 'facility name', 'inspection_site',
        'inspection site', 'site'
    ],
    'INSPECTION_LOCATION_ID': [
        'inspection_location_id', 'inspection location id',
        'location_id', 'location id', 'site_id', 'site id',
        'facility_id', 'facility id', 'inspection_site_id',
        'inspection site id'
    ],
    'INSPECTION_NUMBER': [
        'inspection_number', 'inspection number', 'inspection_num',
        'inspection num', 'insp_number', 'insp number', 'insp_num',
        'insp num', 'inspection_id', 'inspection id', 'case_number',
        'case number'
    ],
    'PATHWAY_ID': [
        'pathway_id', 'pathway id', 'pathwayid', 'route_id',
        'route id', 'pathway identifier'
    ],
    'PATHWAY': [
        'pathway', 'route', 'entry_pathway', 'entry pathway',
        'import_pathway', 'import pathway', 'channel', 'entry_route',
        'entry route'
    ],
    'SHIPPER_NAME': [
        'shipper_name', 'shipper name', 'shipper', 'shipping_company',
        'shipping company', 'carrier', 'carrier_name', 'carrier name',
        'transporter', 'logistics_provider', 'logistics provider'
    ],
    'INSPECTION_LOCATION_STATE_CODE': [
        'inspection_location_state_code', 'inspection location state code',
        'location_state_code', 'location state code', 'state_code',
        'state code', 'insp_state_code', 'insp state code',
        'location_state', 'location state'
    ],
    'DOCUMENT_REVIEW_OVERTIME_ID': [
        'document_review_overtime_id', 'document review overtime id',
        'doc_review_overtime_id', 'doc review overtime id',
        'doc_review_ot_id', 'doc review ot id', 'document_ot_id',
        'document ot id'
    ],
    'DOCUMENT_REVIEW_OVERTIME_NAME': [
        'document_review_overtime_name', 'document review overtime name',
        'doc_review_overtime_name', 'doc review overtime name',
        'doc_review_ot_name', 'doc review ot name', 'document_overtime',
        'document overtime'
    ],
    'INSPECTION_RESULTS_OVERTIME_ID': [
        'inspection_results_overtime_id', 'inspection results overtime id',
        'insp_results_overtime_id', 'insp results overtime id',
        'results_ot_id', 'results ot id', 'inspection_ot_id',
        'inspection ot id'
    ],
    'INSPECTION_RESULTS_OVERTIME_NAME': [
        'inspection_results_overtime_name', 'inspection results overtime name',
        'insp_results_overtime_name', 'insp results overtime name',
        'results_overtime', 'results overtime', 'inspection_overtime',
        'inspection overtime'
    ],
    'OFFSHORE_CUT_SHIP_INSP_NEEDED': [
        'offshore_cut_ship_insp_needed', 'offshore cut ship insp needed',
        'offshore_inspection_needed', 'offshore inspection needed',
        'offshore_insp_needed', 'offshore insp needed',
        'cut_ship_insp_needed', 'cut ship insp needed'
    ],
    'TAXONOMY_ORDER': [
        'taxonomy_order', 'taxonomy order', 'taxonomic_order',
        'taxonomic order', 'order', 'tax_order', 'tax order',
        'scientific_order', 'scientific order'
    ],
    'TAXONOMY_FAMILY': [
        'taxonomy_family', 'taxonomy family', 'taxonomic_family',
        'taxonomic family', 'family', 'tax_family', 'tax family',
        'scientific_family', 'scientific family'
    ],
    'TAXONOMY_GENUS': [
        'taxonomy_genus', 'taxonomy genus', 'taxonomic_genus',
        'taxonomic genus', 'genus', 'tax_genus', 'tax genus',
        'scientific_genus', 'scientific genus'
    ],
    'TAXONOMY_SPECIES': [
        'taxonomy_species', 'taxonomy species', 'taxonomic_species',
        'taxonomic species', 'species', 'tax_species', 'tax species',
        'scientific_species', 'scientific species'
    ],
    'TAXONOMY_SUBSPECIES': [
        'taxonomy_subspecies', 'taxonomy subspecies', 'taxonomic_subspecies',
        'taxonomic subspecies', 'subspecies', 'tax_subspecies',
        'tax subspecies', 'variety', 'var'
    ],
    'MODE_OF_TRANSPORT': [
        'mode_of_transport', 'mode of transport', 'transport_mode',
        'transport mode', 'transportation_mode', 'transportation mode',
        'mode', 'transport_type', 'transport type', 'shipping_method',
        'shipping method'
    ],
    'inspection': [
        'inspection', 'insp', 'inspection_flag', 'inspection flag',
        'inspected', 'inspection_indicator', 'inspection indicator'
    ],
    'shipment': [
        'shipment', 'ship', 'shipment_id', 'shipment id',
        'consignment', 'cargo', 'load'
    ],
    'action': [
        'action', 'action_taken', 'action taken', 'disposition',
        'decision', 'result', 'outcome', 'action_code', 'action code'
    ],
    'fiscal_year': [
        'fiscal_year', 'fiscal year', 'fy', 'fiscal_yr', 'fiscal yr',
        'year_fiscal', 'year fiscal', 'budget_year', 'budget year'
    ],
    'calendar_year': [
        'calendar_year', 'calendar year', 'cy', 'cal_year', 'cal year',
        'year_calendar', 'year calendar', 'year'
    ],
    'HOST_PROXIMITY_ID': [
        'host_proximity_id', 'host proximity id', 'proximity_id',
        'proximity id', 'host_prox_id', 'host prox id'
    ],
    'HOST_PROXIMITY': [
        'host_proximity', 'host proximity', 'proximity', 'host_prox',
        'host prox', 'host_distance', 'host distance'
    ],
    'year': [
        'year', 'yr', 'year_value', 'year value', 'annual_year',
        'annual year'
    ],
    'month': [
        'month', 'mo', 'month_value', 'month value', 'month_name',
        'month name', 'month_number', 'month number'
    ],
    'RISK_UNIT': [
        'risk_unit', 'risk unit', 'risk', 'unit', 'risk_level',
        'risk level', 'ru', 'risk_category', 'risk category'
    ],
    'TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT': [
        'total_sampling_units_for_risk_unit', 'total sampling units for risk unit',
        'total_sampling_units', 'total sampling units', 'total_samples',
        'total samples', 'risk_unit_samples', 'risk unit samples',
        'total_units', 'total units', 'total_samples_risk_unit',
        'total samples risk unit'
    ],
    'SAMPLING_UNITS_FOR_INSPECTION_UNIT': [
        'sampling_units_for_inspection_unit', 'sampling units for inspection unit',
        'inspection_sampling_units', 'inspection sampling units',
        'samples_per_unit', 'samples per unit', 'sampling_unit',
        'sampling unit', 'inspection_samples', 'inspection samples'
    ],
    'REQUIRED_NUMBER_OF_BOXES': [
        'required_number_of_boxes', 'required number of boxes',
        'required_boxes', 'required boxes', 'number_of_boxes',
        'number of boxes', 'boxes_required', 'boxes required',
        'box_count', 'box count', 'boxes', 'required_box_count',
        'required box count'
    ],
    'Row_ID': [
        'row_id', 'row id', 'rowid', 'id', 'record_id', 'record id',
        'index', 'row_number', 'row number', 'record_number',
        'record number', 'pk', 'primary_key', 'primary key'
    ],
    'producer_group': [
        'producer', 'producer group', 'producer_group'
    ]
}


def find_column_name(var: str = None,
                     available_columns: List = None,
                     alias_dict: dict = None,
                     cutoff: float = 0.6):
    """
    Find the correct column name from a variable string.

    Parameters:
    -----------
    var : str
        The input string to match
    available_columns : list
        List of actual column names in the DataFrame
    alias_dict : dict
        Dictionary mapping canonical column names to their aliases
    cutoff : float
        Similarity threshold for fuzzy matching (0.0 to 1.0)

    Returns:
    --------
    str or None
        The matched column name, or None if no match found
    """
    # Use the module-level constant if none provided
    if alias_dict is None:
        alias_dict = COLUMN_NAME_ALIASES

    # Normalize input
    var_normalized = var.lower().strip().replace(' ', '_')

    # Exact match in available columns (case-insensitive)
    for col in available_columns:
        if col.lower() == var_normalized:
            print(col)
            return col

    # Check against alias dictionary
    for canonical_name, aliases in alias_dict.items():
        if canonical_name in available_columns:
            if var_normalized in aliases:
                return canonical_name

    # Fuzzy match against available columns
    match = get_close_matches(var_normalized,
                              [c.lower() for c in available_columns],
                              n=1,
                              cutoff=cutoff)
    if match:
        # Find the original column name (with correct case)
        for col in available_columns:
            if col.lower() == match[0]:
                return col

    # Fuzzy match against all possible aliases
    all_aliases = []
    alias_to_canonical = {}

    for canonical_name, aliases in alias_dict.items():
        if canonical_name in available_columns:
            for alias in aliases:
                all_aliases.append(alias)
                alias_to_canonical[alias] = canonical_name

    match = get_close_matches(var_normalized, all_aliases, n=1, cutoff=cutoff)
    if match:
        return alias_to_canonical[match[0]]

    return None


