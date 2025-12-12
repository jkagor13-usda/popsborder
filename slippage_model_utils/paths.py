# COPYRIGHT NOTICE
# © 2023, 2024, 2025 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path


class DefaultPaths:

    def __init__(self):
        paths_file = Path(__file__)
        self.root = paths_file.parent.parent.parent

    def slippage_data_dir(self) -> Path:
        return self.root / "slippage_data"


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
