# Consignment configuration

## Consignment generation method

The consignments can be either purely synthetic or based on inspection records
(e.g., Form 280 or AQIM). Configuration for the consignments is under the
`consignment` key in the configuration file.

The `generation_method` key can be either `parameter_based` to generate
synthetic consignments based on user-provided list of parameter values, or
`input_file` to create consignments that match inspection records in a CSV
file.

```yaml
consignment:
  generation_method: parameter_based
```

## Items per box

To create the boxes (of items) for each consignment, a value for `items_per_box`
needs to be specified.

```yaml
consignment:
  items_per_box:
    default: 200
```

Only the `default` value for `items_per_box` is required, but the user may also
vary the value by transport pathway. For example, F280 and AQIM inspections
records include information about the consignment pathway, which can be used to
vary the value of `items_per_box`. The following is a configuration for
generating consignments using a file called `AQIM_sample.csv` with
`items_per_box` values that vary by `air` and `maritime` pathways. If the
consignment arrives via an air pathway, one box will contain 200 items. If the
consignment arrives via a maritime pathway, one box will contain 700 items. If
the pathway is not `air` or `maritime`, a default value of 100 items per box is
used.

```yaml
consignment:
  generation_method: input_file
  items_per_box:
    default: 100
    air:
      default: 200
    maritime:
      default: 700
  input_file:
    file_type: AQIM
    file_name: AQIM_sample.csv
```

Notice that the values for `items_per_box` are under a `default` key. In the
future, the simulation may support other keys to vary the number of
`items_per_box` by commodity, origin, or port.

## Synthetic consignments

The main keys for the `parameter_based` consignment generator are the `min` and
`max` values of `boxes`. These values determine the range of sizes of
consignments within the simulation. In the example configuration below,
consignments will have 10 to 100 boxes per consignments.

```yaml
consignment:
  boxes:
    min: 10
    max: 100
```

The generator adds origin, flower (commodity type), and port (where consignment
was received). These are randomly selected from the lists specified in the
configuration. These values are not currently used, but may be used to configure
other parameters in the future (e.g., variable contamination rates by origin or
inspection efficacy by commodity).

```yaml
origins:
  - Netherlands
  - Mexico
  - Israel
  - Japan
  - New Zealand
  - India
  - Tanzania
flowers:
  - Hyacinthus
  - Rosa
  - Gerbera
  - Agapanthus
  - Aegilops
  - Protea
  - Liatris
  - Mokara
  - Anemone
  - Actinidia
ports:
  - NY JFK CBP
  - FL Miami Air CBP
  - HI Honolulu CBP
  - AZ Phoenix CBP
  - VA Dulles CBP
  - CA San Francisco CBP
  - WA Seattle Air CBP
  - TX Brownsville CBP
  - WA Blaine CBP
```

## Create consignments to match an input file

To use a file of inspection records, set the `generation_method` to `input_file`
and specify the `file_type` and `file_name`. Currently, the options for
`file_type` are `F280`, `AQIM`, and `PIS`. Additional type of inspection data can be
supported upon request, or you can format the inspection records in the same way
as F280 or AQIM data, described below. The `file_name` is a path absolute or relative to the place where the Python program is running.

```yaml
consignment:
  consignment_generator: input_file
  input_file:
    file_type: AQIM
    file_name: aqim_sample.csv
```

### F280-based consignments

Consignments in the simulation can be based on real F280 records. In that case,
a CSV file needs to be specified using the `file_name` key.

The CSV is expected to have the following columns:

- QUANTITY which will be used as number of items,
- PATHWAY which is used to determine the `items_per_box` value (case
  insensitive),
- REPORT_DT is used for date,
- COMMODITY as the flower (commodity type),
- ORIGIN_NM as origin, and
- LOCATION as port (where consignment was received).

The CSV file should be comma-separated (`,`) using double quote for text fields
(`"`). The path is absolute or relative to the place where the Python program is
running.

### AQIM-based consignments

Consignments in the simulation can also be based on AQIM inspection records. In
that case, a CSV file needs to be specified using the `file_name` key.

The CSV is expected to have the following columns:

- UNIT which is used to specify the unit (must be items or boxes) used in
  QUANTITY.
- QUANTITY which is used as number of items or number of boxes depending on
  UNIT specified,
- CARGO_FORM which is used to determine the `items_per_box` value similar to
  PATHWAY in F280 (case insensitive),
- CALENDAR_YR is used for date (YYYY only),
- COMMODITY_LIST is used as the flower (commodity type),
- ORIGIN as origin, and
- LOCATION as port of entry (where consignment was received).

The CSV file should be comma-separated (`,`) using double quote for text fields
(`"`). The path is absolute or relative to the place where the Python program is
running.




### PIS-based consignments (`file_type: PIS`)

`PISConsignmentGenerator` handles **Plant Inspection System (PIS)** data, which supports a hierarchical structure:

- Consignment → **inspection units** → **sample units** → **plants**.

Configure it using `generation_method: input_file` and `file_type: PIS`:

```yaml
consignment:
  generation_method: input_file
  input_file:
    file_type: PIS
    file_name: data/pis_inspections.csv
```

The CSV is expected to have the following columns (minimum set):

- `INSPECTION_NUMBER` which is used as the unique consignment/inspection ID.
- `PROPAGATIVE_MATERIAL_TYPE` which is used as the material/commodity type for this inspection unit.
- `SAMPLING_UNITS_FOR_INSPECTION_UNIT` which is used as the actual number of sample units in the inspection unit.
- `QUANTITY` which is used as the total number of plants in the inspection unit.
- `COUNTRY_OF_ORIGIN_NAME` which is used as the origin country.
- `INSPECTION_LOCATION_NAME` which is used as the port/location (where the inspection was conducted).
- `PATHWAY` which is used as the transport pathway.
- `CREATED_DATETIME` which is used as a timestamp in `MM:SS.S` format (minutes:seconds.tenths) representing a time offset within a base date.
- `producer_group` or `PRODUCER_NAME` which is used as the producer name; `producer_group` is used when present and not equal to `"NO_GROUP_MATCH"`.
- A risk-unit label column, which must be **one** of the following (the first matching column name is used):
  - `RISK UNIT`
  - `RISK_UNIT`
  - `risk_unit`
  - `RISK_UNIT_NUMBER`  
  This column is used to group sample units into risk units.

Additional columns may be used depending on your `RiskUnitConfig` settings from `slippage_model_utils.UnitAttributes`.

The CSV file should be comma-separated (`,`) using double quote for text fields (`"`). The path is absolute or relative to the place where the Python program is running.



### Creating a PIS-style input file (`pis_inspections.csv`) with synthetic data

For RBS/PIS workflows, the simulation expects a PIS-style CSV with columns such as
`INSPECTION_NUMBER`, `COUNTRY_OF_ORIGIN_NAME`, `PROPAGATIVE_MATERIAL_TYPE`,
`SAMPLING_UNITS_FOR_INSPECTION_UNIT`, `QUANTITY`, `RISK_UNIT`, etc.
You can generate such a file synthetically using the
`SyntheticConsignmentDataGenerator` class in `popsborder.generator`.

#### Required inputs

To generate synthetic PIS-style consignments, you typically need:

- A **base PIS dataset** with real records to train from  
  (e.g., `train.csv` in your validation data directory).
- An optional **producer group mapping** table (e.g., `producer_group_mapping.csv`)
  containing columns such as:
  - `PRODUCER_NAME`
  - `grouping`
- A PoPS Border **configuration file** (e.g., `config_test.yml`) that will later
  be used by the simulation.

The base PIS dataset should already contain the core PIS columns documented above
(e.g., `INSPECTION_NUMBER`, `COUNTRY_OF_ORIGIN_NAME`,
`PROPAGATIVE_MATERIAL_TYPE`, `QUANTITY`,
`SAMPLING_UNITS_FOR_INSPECTION_UNIT`, `INSPECTION_LOCATION_NAME`, `PATHWAY`,
`RISK_UNIT`/`RISK UNIT`/`risk_unit`/`RISK_UNIT_NUMBER`, etc.).

#### Basic workflow

1. **Load configuration and base data.**
1. **Initialize** `SyntheticConsignmentDataGenerator` with the config, the
   producer group mapping, and the base PIS data file.
1. **Generate** a synthetic dataset using one of the supported sampling methods.
1. **Post-process** the synthetic dataset (e.g., clean names, construct risk units,
   create engineered features).
1. **Save** the final synthetic dataset to a CSV file, e.g. `pis_inspections.csv`.
1. **Point the simulation config** to that file via `consignment.input_file.file_name`.

#### Example: generating `pis_inspections.csv`

Below is an example Python workflow that creates a synthetic PIS-style CSV and
then uses it as the `file_type: PIS` input for consignments:

```python
from pathlib import Path
import pandas as pd

from popsborder.generator import SyntheticConsignmentDataGenerator, save_to_csv
from popsborder.inputs import load_configuration
from popsborder.inspections import construct_risk_units
from slippage_model_utils.engineered_feature_creator import create_engineered_features
from slippage_model_utils.paths import BoxPaths, DefaultPaths
from slippage_model_utils.r_script_wrapper import RVariableCreator

# ----------------------------------------------------------------------
# 1. Set up paths
# ----------------------------------------------------------------------
default_paths = DefaultPaths()
box_paths = BoxPaths()

val_data_path = box_paths.validation_data()
data_dir = default_paths.slippage_data_dir()

# Base PIS training data
pis_data_train = val_data_path / "train.csv"
pis_data_train2 = val_data_path / "training_data_for_test_set.csv"

# Producer group mapping table
producer_group_mapping_path = box_paths.disambiguated_producer_table_mapping()
producer_group_mapping = pd.read_csv(producer_group_mapping_path)

# Output synthetic CSV
synthetic_data_file_name = "pis_inspections.csv"
synth_out_path = data_dir / synthetic_data_file_name

# PoPS Border configuration file
config_file = "config_test.yml"
config = load_configuration(config_file)

# ----------------------------------------------------------------------
# 2. Initialize synthetic generator
# ----------------------------------------------------------------------
synthetic_data_generator = SyntheticConsignmentDataGenerator(
    config=config,
    producer_group_mapping=producer_group_mapping,
    input_data_file=pis_data_train,
)

# ----------------------------------------------------------------------
# 3. Generate synthetic consignment records
#    (sampling_method can be 'naive', 'sequential', or 'gmm')
# ----------------------------------------------------------------------
n_consignments = 1000  # number of synthetic inspections to generate
synth_data = synthetic_data_generator.generate_from_input_data(
    n_consignments=n_consignments,
    sampling_method="sequential",
)

# ----------------------------------------------------------------------
# 4. Post-process synthetic data
#    - Add Row_ID
#    - Clean and group producer/importer names
#    - Reconstruct risk units
#    - Create engineered features
# ----------------------------------------------------------------------
# Add a unique Row_ID per record
synth_data.loc[:, "Row_ID"] = "CR-" + (synth_data.index + 1).astype(str)

# Text preprocessing and grouping for producer/importer using R wrapper
creator = RVariableCreator()

# Keep raw producer name
synth_data["PRODUCER_NAME_RAW"] = synth_data["PRODUCER_NAME"]
synth_data["PRODUCER_NAME1"] = creator.batch_basic_text_preproc(
    text_fields=synth_data["PRODUCER_NAME_RAW"]
)

# Prepare mapping table expected by entity_resolution
producer_group_mapping = producer_group_mapping.rename(
    columns={"PRODUCER_NAME": "name", "grouping": "group"}
)

# Apply producer grouping
synth_data = creator.entity_resolution(
    df=synth_data,
    entity_resolution_lookup_table=producer_group_mapping,
    use_parquet=False,
)

# Clean importer name
synth_data["IMPORTER_NAME_RAW"] = synth_data["IMPORTER_NAME"]
synth_data["IMPORTER_NAME1"] = creator.batch_basic_text_preproc(
    synth_data["IMPORTER_NAME_RAW"]
)

# Reconstruct risk units according to configuration
# (uses PIS fields such as INSPECTION_LOCATION_NAME, COUNTRY_OF_ORIGIN_NAME,
#  PROPAGATIVE_MATERIAL_TYPE, PRODUCER_NAME, etc.)
synth_data["PRODUCER_NAME"] = synth_data["PRODUCER_GROUP_NAME1"]
synth_data["IMPORTER_NAME"] = synth_data["IMPORTER_NAME1"]
synth_data = construct_risk_units(config=config, data=synth_data)

# Create engineered features using training data
dt_train = pd.read_csv(pis_data_train2)
synth_data = create_engineered_features(
    synth_data=synth_data,
    dt_train=dt_train,
    producer_group_mapping=producer_group_mapping,
)

# Final producer/importer grouping columns
synth_data["producer_group"] = synth_data["PRODUCER_GROUP_TOP"]
synth_data["IMPORTER_NAME"] = synth_data["IMPORTER_NAME_TOP"]

# ----------------------------------------------------------------------
# 5. Save final synthetic PIS-style dataset
# ----------------------------------------------------------------------
save_to_csv(synth_data, synth_out_path)



---

Next: [Contamination](contamination.md)
