# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import pandas as pd
import numpy as np
import os
import time

### Inputs
# Output Writing Directory Definition
out_dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Joe Data Analysis'

base_dir = r"C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)"
print("Reading in PIS Data")
start_time = time.time()
files = [
    "PIS_RBS_data_JHUAPL_joined_1.xlsx",
    "PIS_RBS_data_JHUAPL_joined_2.xlsx",
    "PIS_RBS_data_JHUAPL_joined_3.xlsx",
]

dfs = [pd.read_excel(f"{base_dir}\\{f}") for f in files]

end_time = time.time()
total_time = end_time - start_time
total_time_mins = total_time / 60
print(f'Time to read in PIS data:  {total_time_mins} mins')

df_pis = pd.concat(dfs, ignore_index=True)
df_pis_10k = df_pis.head(50_000).reset_index(drop=True)
df_pis_10k.to_csv(os.path.join(base_dir, "PIS_RBS_data_JHUAPL_joined_50k.xlsx"),index=False)


print("Reading in RBS Calculator Data")
start_time = time.time()
df_rbs_calc = pd.read_excel(f"{base_dir}\\{'PIS_RBS_calculator.xlsx'}")
end_time = time.time()
total_time = end_time - start_time
total_time_mins = total_time / 60
print(f'Time to read in RBS Calculator data:  {total_time_mins} mins')










# Support functions for matching
def subset_sum_pick(values, target):
    """
    values: list[int] positive
    target: int
    Returns indices in 'values' that sum to target, or None.
    """
    dp = 1
    parent = {0: (-1, -1)}  # sum -> (prev_sum, idx)

    for i, v in enumerate(values):
        shifted = dp << v
        new = shifted & ~dp
        dp |= shifted

        # record parents for new sums for backtracking
        s = new
        while s:
            lsb = s & -s
            sum_val = lsb.bit_length() - 1
            parent[sum_val] = (sum_val - v, i)
            s -= lsb

        if (dp >> target) & 1:
            break

    if not ((dp >> target) & 1):
        return None

    chosen = []
    cur = target
    while cur != 0:
        prev, idx = parent[cur]
        chosen.append(idx)
        cur = prev
    chosen.reverse()
    return chosen

def solve_inspection_bitset(pis_grp, rbs_grp, scale=1):
    """
    Returns dict {pis_row_index: Risk_Unit} or None.
    Assumes exact partition: sum(PIS) == sum(targets).
    """
    pis_qty = (pis_grp["QUANTITY"].values * scale).round().astype(int)
    pis_idx = pis_grp.index.to_numpy()

    targets = (rbs_grp["TOTAL_PLANT_QUANTITY"].values * scale).round().astype(int)
    ru_labels = rbs_grp["Risk_Unit"].tolist()

    if pis_qty.sum() != targets.sum():
        return None

    # Remaining pools
    remaining_rows = list(range(len(pis_qty)))  # positions in pis_qty
    remaining_targets = list(range(len(targets)))

    assignment = {}

    # --- Easy win: exact single-row matches ---
    # Map qty -> list of row positions
    from collections import defaultdict
    qty_to_rows = defaultdict(list)
    for pos in remaining_rows:
        qty_to_rows[int(pis_qty[pos])].append(pos)

    new_remaining_targets = []
    for tpos in remaining_targets:
        t = int(targets[tpos])
        if qty_to_rows.get(t):
            rowpos = qty_to_rows[t].pop()
            assignment[pis_idx[rowpos]] = ru_labels[tpos]
            remaining_rows.remove(rowpos)
        else:
            new_remaining_targets.append(tpos)
    remaining_targets = new_remaining_targets

    # Solve remaining targets largest-first
    remaining_targets.sort(key=lambda tpos: targets[tpos], reverse=True)

    for tpos in remaining_targets:
        target = int(targets[tpos])
        ru = ru_labels[tpos]

        values = [int(pis_qty[rp]) for rp in remaining_rows]
        pick = subset_sum_pick(values, target)
        if pick is None:
            return None

        picked_rowpos = [remaining_rows[i] for i in pick]
        for rp in picked_rowpos:
            assignment[pis_idx[rp]] = ru

        used = set(picked_rowpos)
        remaining_rows = [rp for rp in remaining_rows if rp not in used]

    # If we assigned exactly all rows, great
    if len(assignment) != len(pis_grp):
        return None

    return assignment



def merge_pis_rbs_calc_data(df_pis,df_rbs_calc):
    # IDs of interest from df_pis
    rbs_only_inspection_ids = set(
        df_pis.loc[
            (df_pis['RBS_STATUS'].eq('RBS Complete')) & (df_pis['IS_RBS'].eq(1)),
            'INSPECTION_ID'
        ].dropna()
    )

    # Filter rbs calc and pis inspection data to only those IDs
    df_rbs_subset_rbs_only = df_rbs_calc[df_rbs_calc['INSPECTION_ID'].isin(rbs_only_inspection_ids)].copy()
    # Filter df_rbs_calc to only those IDs
    df_pis_subset_rbs_only = df_pis[df_pis['INSPECTION_ID'].isin(rbs_only_inspection_ids)].copy()

    #### Clean/filter RBS once (outside any loop)
    print("Cleaning RBS Data")
    cols_required = [
        'TOTAL_SAMPLING_UNITS',
        'TOTAL_PLANT_QUANTITY',
        'RISK_RATING',
        'CONFIDENCE_LEVEL',
        'DETECTION_LEVEL',
        'REQUIERD_NUMBER_OF_BOXES',
        'PULL_NUMBERS',
    ]

    rbs = df_rbs_subset_rbs_only.copy()

    # Turn whitespace-only strings in required cols into NA (only for object/string cols)
    obj_cols = [c for c in cols_required if c in rbs.columns and rbs[c].dtype == "object"]
    rbs[obj_cols] = rbs[obj_cols].replace(r'^\s*$', pd.NA, regex=True)

    # Drop rows missing any required fields
    rbs_valid = rbs.dropna(subset=cols_required)

    # (optional but often needed) ensure TOTAL_PLANT_QUANTITY is numeric
    rbs_valid["TOTAL_PLANT_QUANTITY"] = pd.to_numeric(rbs_valid["TOTAL_PLANT_QUANTITY"], errors="coerce")
    rbs_valid = rbs_valid.dropna(subset=["TOTAL_PLANT_QUANTITY"])

    ### Aggregate quantities by INSPECTION_ID once
    print("Aggregating quantities...")
    rbs_qty = rbs_valid.groupby("INSPECTION_ID")["TOTAL_PLANT_QUANTITY"].sum()
    pis_qty = df_pis_subset_rbs_only.groupby("INSPECTION_ID")["QUANTITY"].sum()



    ### Compare for the IDs you care about
    print("Looking for inspection IDs with matching quantities...")
    ids = pd.Index(rbs_only_inspection_ids)

    # align to the same index (missing becomes NaN)
    rbs_aligned = rbs_qty.reindex(ids)
    pis_aligned = pis_qty.reindex(ids)

    # define match conditions
    both_present = rbs_aligned.notna() & pis_aligned.notna()
    qty_equal = both_present & (pis_aligned == rbs_aligned)

    inspect_ids_quantity_matched = ids[qty_equal].tolist()
    inspect_ids_quantity_not_matched = ids[~qty_equal].tolist()



    ### Re-subset data to those IDS that have matching quantity and total quantity values
    matched_ids = pd.Index(inspect_ids_quantity_matched)

    df_rbs_subset_rbs_only = df_rbs_calc.loc[df_rbs_calc["INSPECTION_ID"].isin(matched_ids)].copy()
    df_pis_subset_rbs_only = df_pis.loc[df_pis["INSPECTION_ID"].isin(matched_ids)].copy()

    # Ensure whitespace-only strings are NA before dropna (important if your cols are object dtype)
    obj_cols = [c for c in cols_required if
                c in df_rbs_subset_rbs_only.columns and df_rbs_subset_rbs_only[c].dtype == "object"]
    df_rbs_subset_rbs_only.loc[:, obj_cols] = df_rbs_subset_rbs_only.loc[:, obj_cols].replace(r"^\s*$", pd.NA,
                                                                                              regex=True)

    df_rbs_subset_rbs_only = df_rbs_subset_rbs_only.dropna(subset=cols_required)

    # Start from your already-filtered valid RBS
    rbs_valid = df_rbs_subset_rbs_only.copy()

    # Ensure a stable per-group order
    rbs_valid = rbs_valid.reset_index(drop=False).rename(columns={"index": "_row"})
    rbs_valid = rbs_valid.sort_values(["INSPECTION_ID", "_row"], kind="mergesort")

    # Create 1..n counter within each INSPECTION_ID
    ru_num = rbs_valid.groupby("INSPECTION_ID").cumcount() + 1

    # Create risk unit ids for the different inspection numbers
    rbs_valid["Risk_Unit"] = rbs_valid["INSPECTION_ID"].astype(str) + "_" + ru_num.astype(str)

    # Keep a numeric reference to the risk unit ids
    rbs_valid["Risk_Unit_Num"] = ru_num


    # Tier 1 matching (only one risk unit defined)
    pis = df_pis_subset_rbs_only.copy()
    rbs = rbs_valid.copy()

    pis["QUANTITY"] = pd.to_numeric(pis["QUANTITY"], errors="coerce").fillna(0)
    rbs["TOTAL_PLANT_QUANTITY"] = pd.to_numeric(rbs["TOTAL_PLANT_QUANTITY"], errors="coerce")

    ru_count = rbs.groupby("INSPECTION_ID").size()

    # One RU label per inspection (works for m==1)
    single_ru = rbs.groupby("INSPECTION_ID")["Risk_Unit"].first()

    pis["risk_unit"] = pd.NA
    single_ids = ru_count[ru_count == 1].index
    pis.loc[pis["INSPECTION_ID"].isin(single_ids), "risk_unit"] = (
        pis.loc[pis["INSPECTION_ID"].isin(single_ids), "INSPECTION_ID"].map(single_ru)
    )

    multi_ids = ru_count[ru_count > 1].index

    unresolved = []
    count=0
    for insp_id in multi_ids:
        print(f'ID {count+1} out of {len(multi_ids)}')
        count += 1
        pis_grp = pis[(pis["INSPECTION_ID"] == insp_id) & (pis["risk_unit"].isna())]
        rbs_grp = rbs[rbs["INSPECTION_ID"] == insp_id]

        if pis_grp.empty or rbs_grp.empty:
            continue

        # Try scale=1 first. If your quantities have decimals, use scale=100 or 1000.
        assign = solve_inspection_bitset(pis_grp, rbs_grp, scale=1)

        if assign is None:
            unresolved.append(insp_id)
            continue

        pis.loc[list(assign.keys()), "risk_unit"] = list(assign.values())

    print("Number of Unresolved inspections:", len(unresolved))
    return pis, rbs_valid, unresolved


# Function to proportionally allocate sample units to inspection units based on quantity
def allocate_sampling_units(pis: pd.DataFrame) -> pd.DataFrame:
    df = pis.copy()

    # Fill NaN values
    df['TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT'] = df['TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT'].fillna(0).astype(float)
    df['QUANTITY'] = df['QUANTITY'].fillna(0).astype(float)

    # Group
    g = df.groupby("risk_unit", sort=False, dropna=False)

    # Calculate group-level metrics (aligned to each row)
    group_size = g.transform('size')
    qty_sum = g['QUANTITY'].transform('sum')

    # Vectorized allocation logic
    df['SAMPLING_UNITS_FOR_INSPECTION_UNIT'] = np.where(
        group_size == 1,
        # Single row: use total directly
        df['TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT'],
        # Multiple rows: proportional allocation (handle divide by zero)
        np.where(
            qty_sum > 0,
            (df['QUANTITY'] / qty_sum) * df['TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT'],
            0.0
        )
    )

    return df






updated_pis_data, rbs_valid, unresolved_ids = merge_pis_rbs_calc_data(df_pis,df_rbs_calc)

updated_pis_data = updated_pis_data.rename(columns={
    "risk_unit": "RISK_UNIT",
    "Total_Sampling_Units_for_Risk_Unit": "TOTAL_SAMPLING_UNITS_FOR_RISK_UNIT",
    "Sampling_Units_for_Inspection_Unit": "SAMPLING_UNITS_FOR_INSPECTION_UNIT",
})


updated_pis_data = updated_pis_data.merge(
    rbs_valid[["Risk_Unit", "REQUIRED_NUMBER_OF_BOXES"]],
    left_on="RISK_UNIT",
    right_on="Risk_Unit",
    how="left"
)




updated_pis_data = updated_pis_data.rename(columns={
    "REQUIERD_NUMBER_OF_BOXES": "REQUIRED_NUMBER_OF_BOXES"
})

updated_pis_data = updated_pis_data.rename(columns={
    "RISK_UNIT": "risk_unit"
})

num_groups = updated_pis_data.groupby(
    ["INSPECTION_ID", "risk_unit"]
).ngroups


## Add Total_Sampling_Units_for_Risk_Unit to PIS via a merge
## Create a lookup table
ru_lookup = (
    rbs_valid[["INSPECTION_ID", "Risk_Unit", "TOTAL_SAMPLING_UNITS", "REQUIERD_NUMBER_OF_BOXES"]]
    .rename(columns={"Risk_Unit": "risk_unit",
                     "TOTAL_SAMPLING_UNITS": "Total_Sampling_Units_for_Risk_Unit",
                     "REQUIERD_NUMBER_OF_BOXES": "REQUIRED_NUMBER_OF_BOXES"})
    .copy()
)

# Merge into PIS data using lookup
pis = updated_pis_data.merge(
    ru_lookup,
    on=["INSPECTION_ID", "risk_unit"],
    how="left",
    validate="many_to_one",
)


#pis.to_csv(r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\updated_pis_data.csv')




pis_updated = allocate_sampling_units(pis)

## Quick verification checks of the generated sampling unit columns
# check sums match within each risk unit group
check_sum = (
    pis_updated.groupby(["INSPECTION_ID", "risk_unit"])
       .agg(
           total_sampling_units=("Total_Sampling_Units_for_Risk_Unit", "first"),
           sampling_units_sum=('SAMPLING_UNITS_FOR_INSPECTION_UNIT', "sum"),
           rows=('SAMPLING_UNITS_FOR_INSPECTION_UNIT', "size"),
       )
)

bad = check_sum[abs(check_sum["total_sampling_units"] - check_sum["sampling_units_sum"])>0.00000001]
print("Groups with mismatched sums:", len(bad))

pis_updated.to_csv(r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\updated_pis_data.csv')



