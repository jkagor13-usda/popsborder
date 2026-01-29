# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

import msoffcrypto
import io
import pandas as pd
import numpy as np
import os






def read_data(file_path, password):
    print("Reading in Data")
    print(f'   Reading from {file_path}')

    # Open and decrypt the file
    with open(file_path, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)

        decrypted = io.BytesIO()
        office_file.decrypt(decrypted)

    # Load the decrypted content with pandas
    df = pd.read_excel(decrypted, engine="openpyxl")

    return df


################################################################
#################### DATA ANALYSIS #############################
################################################################

def execute_shared_column_analysis(table1, table2, out_dir):
    ### Shared Column Analysis ####
    # Mapping: table1 column -> table2 column
    shared_columns = {
        "COUNTRY_OF_ORIGIN_NAME": "COUNTRY_OF_ORIGIN_NAME",
        "INSPECTION_ID": "INSPECTION_ID",
        "INSPECTION_LOCATION_NAME": "INSPECTION_LOCATION_NAME",
        "INSPECTION_LOCATION_STATE_CODE": "INSPECTION_LOCATION_STATE_CODE",
        "INSPECTION_NUMBER": "INSPECTION_NUMBER",
        "PATHWAY": "PATHWAY",
        "PRODUCER_ID": "PRODUCER_ID",
        "PROPAGATIVE_MATERIAL_TYPE": "PROPAGATIVE_MATERIAL_TYPE",
        "SUBCATEGORY": "SUBCATEGORY"
    }

    results = []

    for col_table1, col_table2 in shared_columns.items():
        vals1 = set(table1[col_table1].dropna().unique())
        vals2 = set(table2[col_table2].dropna().unique())
        overlap = vals1 & vals2

        results.append({
            "Table1_Column": col_table1,
            "Table2_Column": col_table2,
            "Unique_in_table1": len(vals1),
            "Unique_in_table2": len(vals2),
            "Unique_in_both": len(overlap),
            "Only_in_table1": len(vals1 - vals2),
            "Only_in_table2": len(vals2 - vals1)
        })

    summary_shared_columns = pd.DataFrame(results)

    # Write out data
    filename = os.path.join(out_dir, f'summary_shared_columns.csv')
    summary_shared_columns.to_csv(filename,index=False)