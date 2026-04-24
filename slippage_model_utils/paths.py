# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path


class DefaultPaths:

    def __init__(self):
        paths_file = Path(__file__)
        self.root = paths_file.parent.parent

    def slippage_data_dir(self) -> Path:
        return self.root / "development_files" / "slippage_data"

    def input_data_dir(self) -> Path:
        return self.root / "data_input"

    def impact_data_dir(self) -> Path:
        return self.root / "impact_data"

    def output_dir(self) -> Path:
        return self.root / "output"

    def validation_output_dir(self) -> Path:
        return self.output_dir() / "validation_outputs"

    def prediction_model_compliance_dir(self) -> Path:
        return self.root / "Prediction_Model_Compliance_Tables"

    def tmp_dir(self) -> Path:
        """Root temporary data directory"""
        return self.root / "tmp"

    def compliance_dir(self) -> Path:
        """Directory for compliance lookup files"""
        compliance_path = self.tmp_dir() / "compliance"
        # Ensure directory exists
        compliance_path.mkdir(parents=True, exist_ok=True)
        return compliance_path


class BoxPaths:

    def __init__(self, box_root=Path.home() / "Box"):
        self.box_root = box_root

    def box(self) -> Path:
        return self.box_root

    def shared_ppq_data(self) -> Path:
        primary_path = self.box() / "NHH15 - USDA APHIS EDISON" / "05 PPQ Engagement" / "PPQ RBS Data (Folder shared with APHIS)"
        if primary_path.exists():
            return primary_path
        else:
            raise FileNotFoundError("No path found to use as the PPQ Data Folder")

    def rbs_calc_data(self) -> Path:
         return self.shared_ppq_data() / "PIS_RBS_calculator.xlsx"

    def validation_data(self) -> Path:
        primary_path = self.shared_ppq_data() / "APL Created Data Related Items" / "Joe Data Analysis" / "validation_data"
        if primary_path.exists():
            return primary_path
        else:
            raise FileNotFoundError("No path found to use as the PPQ Data Folder")

    def apl_created_data_folder(self) -> Path:
         return self.shared_ppq_data() / "APL Created Data Related Items"

    def disambiguated_look_up_tables_folder(self) -> Path:
         return self.apl_created_data_folder() / "disambiguated_look-up_tables"

    def producer_folder(self) -> Path:
         return self.disambiguated_look_up_tables_folder() / "producer"

    def disambiguated_producer_table_mapping(self) -> Path:
         return self.disambiguated_look_up_tables_folder() / "erf_train_450.csv"

    def model_testing_data_folder(self) -> Path:
        return self.apl_created_data_folder() / "Model_Testing"
