from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MIN_REPLICATIONS_FOR_INTERVAL = 5
REPORT_LEVEL_ORDER = ["Consignment", "Inspection", "Sample", "Plant"]


def _require_python_docx():
    try:
        import docx
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        from docx.shared import Inches
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "python-docx is required to export Word reports. Install it with "
            "`pip install python-docx`."
        ) from exc
    return docx, WD_PARAGRAPH_ALIGNMENT, Inches


def _read_csv_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Unable to read CSV at {csv_path}: {exc}") from exc


def _format_report_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _format_interval_value(mean_value: Any, lower: Any = None, upper: Any = None, *, suffix: str = "") -> str:
    base = _format_report_value(mean_value)
    if base == "n/a":
        return base
    if suffix:
        base = f"{base}{suffix}"
    if lower is None or upper is None or pd.isna(lower) or pd.isna(upper):
        return base
    return f"{base} ({float(lower):,.2f}{suffix} - {float(upper):,.2f}{suffix})"


def _add_key_value_table(document, rows: List[tuple[str, Any]]) -> None:
    if not rows:
        return
    table = document.add_table(rows=1, cols=2)
    table.style = "Light List Accent 1"
    header = table.rows[0].cells
    header[0].text = "Field"
    header[1].text = "Value"
    for key, value in rows:
        row = table.add_row().cells
        row[0].text = str(key)
        row[1].text = _format_report_value(value)


def _add_dataframe_table(document, df: Optional[pd.DataFrame], *, max_rows: int = 20) -> None:
    if df is None or df.empty:
        paragraph = document.add_paragraph()
        paragraph.add_run("No data available.").italic = True
        return
    preview = df.head(max_rows).copy()
    table = document.add_table(rows=1, cols=len(preview.columns))
    table.style = "Light List Accent 1"
    for idx, col in enumerate(preview.columns):
        table.rows[0].cells[idx].text = str(col)
    for _, values in preview.iterrows():
        row = table.add_row().cells
        for idx, value in enumerate(values.tolist()):
            row[idx].text = _format_report_value(value)


def _document_content_width_inches(document) -> float:
    _, _, inches = _require_python_docx()
    section = document.sections[-1]
    usable_width = section.page_width - section.left_margin - section.right_margin
    return usable_width / inches(1)


def _add_plot_image(document, fig) -> None:
    _, _, inches = _require_python_docx()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    document.add_picture(buffer, width=inches(_document_content_width_inches(document)))
    buffer.close()


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    if not pd.api.types.is_string_dtype(series):
        return numeric
    extracted = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"^\s*([-+]?\d*\.?\d+)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def _normalize_report_level(value: Any) -> str:
    text = str(value).strip()
    if text == "Plants":
        return "Plant"
    return text


def _ensure_report_levels(table: Optional[pd.DataFrame], value_cols: List[str]) -> Optional[pd.DataFrame]:
    if table is None or table.empty or not {"Scenario", "Level"}.issubset(table.columns):
        return table
    normalized = table.copy()
    normalized["Level"] = normalized["Level"].map(_normalize_report_level)
    scenarios = normalized["Scenario"].dropna().astype(str).unique().tolist()
    if not scenarios:
        return normalized
    scaffold = pd.MultiIndex.from_product(
        [scenarios, REPORT_LEVEL_ORDER],
        names=["Scenario", "Level"],
    ).to_frame(index=False)
    merged = scaffold.merge(normalized, on=["Scenario", "Level"], how="left")
    for col in value_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    merged["Level"] = pd.Categorical(merged["Level"], categories=REPORT_LEVEL_ORDER, ordered=True)
    return merged.sort_values(["Scenario", "Level"]).reset_index(drop=True)


def _sort_report_levels(table: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if table is None or table.empty or "Level" not in table.columns:
        return table
    sorted_table = table.copy()
    sorted_table["Level"] = sorted_table["Level"].map(_normalize_report_level)
    sorted_table["Level"] = pd.Categorical(sorted_table["Level"], categories=REPORT_LEVEL_ORDER, ordered=True)
    sort_cols = ["Level"]
    if "Scenario" in sorted_table.columns:
        sort_cols = ["Scenario", "Level"]
    return sorted_table.sort_values(sort_cols).reset_index(drop=True)


def _build_bar_plot(df: pd.DataFrame, *, title: str, x_col: str, y_col: str, color: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col].astype(float), color=color)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def _build_faceted_bar_plot(
        df: pd.DataFrame,
        *,
        title: str,
        x_col: str,
        y_col: str,
        facet_col: str,
        color: str,
        facet_order: Optional[List[str]] = None,
        ylabel: Optional[str] = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    needed_cols = [x_col, y_col, facet_col]
    plot_df = df[needed_cols].copy()
    plot_df[y_col] = _coerce_numeric_series(plot_df[y_col])
    plot_df = plot_df.dropna(subset=[facet_col, y_col])
    if plot_df.empty:
        return None

    levels = facet_order[:] if facet_order else plot_df[facet_col].astype(str).unique().tolist()

    fig, axes = plt.subplots(len(levels), 1, figsize=(8.2, 2.8 * len(levels)), squeeze=False, sharex=True, sharey=False)
    for idx, level in enumerate(levels):
        ax = axes[idx][0]
        facet_df = plot_df[plot_df[facet_col].astype(str) == str(level)].copy()
        if not facet_df.empty:
            facet_df[x_col] = facet_df[x_col].astype(str)
            ax.bar(facet_df[x_col], facet_df[y_col], color=color)
        ax.set_title(str(level))
        ax.set_xlabel("Scenario" if idx == len(levels) - 1 else "")
        if idx == 0:
            ax.set_ylabel(ylabel or y_col)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _build_grouped_plot(df: pd.DataFrame, *, title: str, x_col: str, value_cols: List[str], ylabel: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_df = df[[x_col, *value_cols]].copy()
    for column in value_cols:
        plot_df[column] = _coerce_numeric_series(plot_df[column])
    plot_df = plot_df.dropna(how="all", subset=value_cols)
    if plot_df.empty:
        return None
    labels = plot_df[x_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(value_cols), 1)
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    colors = ["#1f77b4", "#6baed6", "#9ecae1", "#3182bd"]
    for idx, column in enumerate(value_cols):
        series = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0)
        ax.bar(x + (idx - (len(value_cols) - 1) / 2) * width, series, width=width, label=column, color=colors[idx % len(colors)])
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _build_faceted_grouped_plot(
        df: pd.DataFrame,
        *,
        title: str,
        x_col: str,
        value_cols: List[str],
        ylabel: str,
        facet_col: str,
        facet_order: Optional[List[str]] = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    needed_cols = [x_col, facet_col, *value_cols]
    plot_df = df[needed_cols].copy()
    for column in value_cols:
        plot_df[column] = _coerce_numeric_series(plot_df[column])
    plot_df = plot_df.dropna(how="all", subset=value_cols)
    if plot_df.empty:
        return None

    levels = facet_order[:] if facet_order else plot_df[facet_col].astype(str).unique().tolist()

    fig, axes = plt.subplots(len(levels), 1, figsize=(8.2, 3.0 * len(levels)), squeeze=False, sharex=True, sharey=False)
    colors = ["#1f77b4", "#6baed6", "#9ecae1", "#3182bd"]
    for idx, level in enumerate(levels):
        ax = axes[idx][0]
        facet_df = plot_df[plot_df[facet_col].astype(str) == str(level)].copy()
        if not facet_df.empty:
            labels = facet_df[x_col].astype(str).tolist()
            x = np.arange(len(labels))
            width = 0.8 / max(len(value_cols), 1)
            for value_idx, column in enumerate(value_cols):
                series = facet_df[column].fillna(0.0)
                ax.bar(
                    x + (value_idx - (len(value_cols) - 1) / 2) * width,
                    series,
                    width=width,
                    label=column,
                    color=colors[value_idx % len(colors)],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(str(level))
        ax.set_xlabel("Scenario" if idx == len(levels) - 1 else "")
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if idx == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _mean_by_scenario(df: Optional[pd.DataFrame], columns: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "name" not in df.columns:
        return None
    available = [col for col in columns if col in df.columns]
    if not available:
        return None
    grouped = df[["name", *available]].groupby("name", as_index=False).mean(numeric_only=True)
    return grouped


def _intervals_by_scenario(df: Optional[pd.DataFrame], columns: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "name" not in df.columns:
        return None
    available = [col for col in columns if col in df.columns]
    if not available:
        return None
    rows = []
    for scenario_name, group in df[["name", *available]].groupby("name"):
        row: Dict[str, Any] = {"name": scenario_name}
        for col in available:
            series = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}__n"] = int(series.size)
            row[f"{col}__lower"] = series.quantile(0.025) if not series.empty else np.nan
            row[f"{col}__upper"] = series.quantile(0.975) if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_intervals_to_table(
        table: Optional[pd.DataFrame],
        interval_df: Optional[pd.DataFrame],
        mappings: Dict[str, str],
        *,
        suffix_map: Optional[Dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    if table is None:
        return None
    suffix_map = suffix_map or {}
    if interval_df is None:
        return table
    merged = table.merge(interval_df, left_on="Scenario", right_on="name", how="left")
    for source_col, display_col in mappings.items():
        if display_col not in merged.columns:
            continue
        lower_col = f"{source_col}__lower"
        upper_col = f"{source_col}__upper"
        n_col = f"{source_col}__n"

        def _render(row):
            lower = row.get(lower_col)
            upper = row.get(upper_col)
            n = row.get(n_col, 0)
            if pd.isna(n) or int(n) < MIN_REPLICATIONS_FOR_INTERVAL:
                lower = None
                upper = None
            return _format_interval_value(
                row.get(display_col),
                lower,
                upper,
                suffix=suffix_map.get(display_col, ""),
            )

        merged[display_col] = merged.apply(_render, axis=1)

    drop_cols = [col for col in merged.columns if col == "name" or col.endswith("__lower") or col.endswith("__upper") or col.endswith("__n")]
    return merged.drop(columns=drop_cols)


def _run_summary_rows(results_df: Optional[pd.DataFrame], all_runs_df: Optional[pd.DataFrame]) -> List[tuple[str, Any]]:
    if results_df is None or results_df.empty:
        return []
    rows: List[tuple[str, Any]] = []
    scenario_count = int(results_df["name"].nunique()) if "name" in results_df.columns else len(results_df)
    rows.append(("Scenarios", scenario_count))
    if all_runs_df is not None and not all_runs_df.empty and "replication" in all_runs_df.columns:
        if {"name", "replication"}.issubset(all_runs_df.columns):
            replications_per_scenario = int(all_runs_df.groupby("name")["replication"].nunique().min())
            rows.append(("Replications per scenario", replications_per_scenario))
            rows.append(("Replication rows", len(all_runs_df)))
    if "num_inspections" in results_df.columns:
        rows.append(("Inspected consignments", int(results_df["num_inspections"].sum())))
    if "total_slipped_units" in results_df.columns:
        rows.append(("Mean slipped plant units", float(results_df["total_slipped_units"].sum())))
    return rows


def _scenario_details_table(scenario_table_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if scenario_table_df is None or scenario_table_df.empty:
        return None
    column_aliases = {
        "name": "Scenario label",
        "consignment/input_file/file_name": "Consignment (RBS) file",
        "inspection/compliance_table/file_name": "RBS compliance policy",
        "contamination/contamination_rate/beta_binomial_parameters/default/alpha": "Alpha",
        "contamination/contamination_rate/beta_binomial_parameters/default/beta": "Beta",
        "contamination/contamination_rate/beta_binomial_parameters/default/theta": "Theta",
    }
    available = [col for col in column_aliases if col in scenario_table_df.columns]
    if not available:
        return None
    details = scenario_table_df[available].rename(columns=column_aliases).copy()
    for col in ["Consignment (RBS) file", "RBS compliance policy"]:
        if col in details.columns:
            details[col] = details[col].map(lambda value: Path(str(value)).name if pd.notna(value) else "n/a")
    return details


def _inspection_action_summary(all_runs_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if all_runs_df is None or all_runs_df.empty or "name" not in all_runs_df.columns:
        return None
    needed = {
        "total_intercepted_inspection_units",
        "total_slipped_inspection_units",
        "inspection_clean_inspected",
        "inspection_clean_not_inspected",
        "consignment_intercepted",
        "consignment_slipped",
        "consignment_clean_inspected",
        "consignment_clean_not_inspected",
    }
    if not needed.issubset(all_runs_df.columns):
        return None
    return all_runs_df[
        [
            "name",
            "total_intercepted_inspection_units",
            "total_slipped_inspection_units",
            "inspection_clean_inspected",
            "inspection_clean_not_inspected",
            "consignment_intercepted",
            "consignment_slipped",
            "consignment_clean_inspected",
            "consignment_clean_not_inspected",
        ]
    ].groupby("name", as_index=False).mean(numeric_only=True)


def _slippage_level_table(results_df: Optional[pd.DataFrame], all_runs_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    base = _mean_by_scenario(results_df, ["total_slipped_units"])
    if base is None:
        return None
    sample_df = _mean_by_scenario(all_runs_df, ["total_slipped_sample_units"])
    inspection_df = _mean_by_scenario(all_runs_df, ["total_slipped_inspection_units"])
    table = base.rename(columns={"name": "Scenario", "total_slipped_units": "Slipped plant units"})
    if sample_df is not None:
        table = table.merge(
            sample_df.rename(columns={"name": "Scenario", "total_slipped_sample_units": "Slipped sample units"}),
            on="Scenario",
            how="left",
        )
    if inspection_df is not None:
        table = table.merge(
            inspection_df.rename(columns={"name": "Scenario", "total_slipped_inspection_units": "Slipped inspection units"}),
            on="Scenario",
            how="left",
        )
    interval_df = _intervals_by_scenario(
        all_runs_df,
        ["total_slipped_units", "total_slipped_sample_units", "total_slipped_inspection_units"],
    )
    return _apply_intervals_to_table(
        table,
        interval_df,
        {
            "total_slipped_units": "Slipped plant units",
            "total_slipped_sample_units": "Slipped sample units",
            "total_slipped_inspection_units": "Slipped inspection units",
        },
    )


def _action_level_table(results_df: Optional[pd.DataFrame], all_runs_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    required = {"intercepted", "false_neg", "total_contaminated_units", "total_slipped_units"}
    if not required.issubset(results_df.columns):
        return None
    table = results_df[
        ["name", "intercepted", "false_neg", "total_contaminated_units", "total_slipped_units"]
    ].copy()
    table["Plant intercepted"] = table["total_contaminated_units"] - table["total_slipped_units"]
    table = table.rename(
        columns={
            "name": "Scenario",
            "intercepted": "Consignment intercepted",
            "false_neg": "Consignment slipped",
            "total_slipped_units": "Plant slipped",
        }
    )
    keep_cols = ["Scenario", "Consignment intercepted", "Consignment slipped", "Plant intercepted", "Plant slipped"]
    sample_df = _mean_by_scenario(all_runs_df, ["total_contaminated_sample_units", "total_slipped_sample_units"])
    if sample_df is not None and {"total_contaminated_sample_units", "total_slipped_sample_units"}.issubset(sample_df.columns):
        sample_df["Sample intercepted"] = sample_df["total_contaminated_sample_units"] - sample_df["total_slipped_sample_units"]
        table = table.merge(
            sample_df.rename(columns={"name": "Scenario", "total_slipped_sample_units": "Sample slipped"})[
                ["Scenario", "Sample intercepted", "Sample slipped"]
            ],
            on="Scenario",
            how="left",
        )
        keep_cols.extend(["Sample intercepted", "Sample slipped"])
    inspection_summary = _inspection_action_summary(all_runs_df)
    if inspection_summary is not None:
        table = table.merge(
            inspection_summary.rename(
                columns={
                    "name": "Scenario",
                    "total_intercepted_inspection_units": "Inspection intercepted",
                    "total_slipped_inspection_units": "Inspection slipped",
                }
            )[["Scenario", "Inspection intercepted", "Inspection slipped"]],
            on="Scenario",
            how="left",
        )
        keep_cols.extend(["Inspection intercepted", "Inspection slipped"])
    action_runs = None
    if all_runs_df is not None and {"name", "intercepted", "false_neg", "total_contaminated_units", "total_slipped_units"}.issubset(all_runs_df.columns):
        action_runs = all_runs_df[
            ["name", "intercepted", "false_neg", "total_contaminated_units", "total_slipped_units"]
        ].copy()
        action_runs["Plant intercepted"] = action_runs["total_contaminated_units"] - action_runs["total_slipped_units"]
        action_runs = action_runs.rename(
            columns={
                "intercepted": "Consignment intercepted",
                "false_neg": "Consignment slipped",
                "total_slipped_units": "Plant slipped",
            }
        )
        if {"total_contaminated_sample_units", "total_slipped_sample_units"}.issubset(all_runs_df.columns):
            action_runs["Sample intercepted"] = all_runs_df["total_contaminated_sample_units"] - all_runs_df["total_slipped_sample_units"]
            action_runs["Sample slipped"] = all_runs_df["total_slipped_sample_units"]
        if inspection_summary is not None:
            action_runs = action_runs.merge(
                inspection_summary.rename(
                    columns={
                        "name": "Scenario",
                        "total_intercepted_inspection_units": "Inspection intercepted",
                        "total_slipped_inspection_units": "Inspection slipped",
                    }
                ),
                left_on="name",
                right_on="Scenario",
                how="left",
            )
            if "Scenario" in action_runs.columns:
                action_runs = action_runs.drop(columns=["Scenario"])
    table = _apply_intervals_to_table(
        table,
        _intervals_by_scenario(
            action_runs,
            [
                "Consignment intercepted",
                "Consignment slipped",
                "Plant intercepted",
                "Plant slipped",
                "Sample intercepted",
                "Sample slipped",
                "Inspection intercepted",
                "Inspection slipped",
            ],
        ),
        {col: col for col in keep_cols if col != "Scenario"},
    )
    return table[[col for col in keep_cols if col in table.columns]]


def _contamination_level_table(results_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty:
        return None
    required = {
        "name",
        "total_contaminated_units",
        "total_contaminated_sample_units",
        "total_contaminated_inspection_units",
        "num_plants",
        "num_sample_units",
        "num_inspection_units",
    }
    if not required.issubset(results_df.columns):
        return None
    table = results_df[
        [
            "name",
            "total_contaminated_units",
            "total_contaminated_sample_units",
            "total_contaminated_inspection_units",
            "num_plants",
            "num_sample_units",
            "num_inspection_units",
        ]
    ].copy()
    table["Plant contamination %"] = np.where(table["num_plants"] > 0, (table["total_contaminated_units"] / table["num_plants"]) * 100.0, np.nan)
    table["Sample unit contamination %"] = np.where(table["num_sample_units"] > 0, (table["total_contaminated_sample_units"] / table["num_sample_units"]) * 100.0, np.nan)
    table["Inspection unit contamination %"] = np.where(table["num_inspection_units"] > 0, (table["total_contaminated_inspection_units"] / table["num_inspection_units"]) * 100.0, np.nan)
    table = table.rename(
        columns={
            "name": "Scenario",
            "total_contaminated_units": "Plant contaminated",
            "total_contaminated_sample_units": "Sample unit contaminated",
            "total_contaminated_inspection_units": "Inspection unit contaminated",
        }
    )
    table = table[
        [
            "Scenario",
            "Plant contaminated",
            "Plant contamination %",
            "Sample unit contaminated",
            "Sample unit contamination %",
            "Inspection unit contaminated",
            "Inspection unit contamination %",
        ]
    ]
    interval_source = results_df[
        [
            "name",
            "total_contaminated_units",
            "total_contaminated_sample_units",
            "total_contaminated_inspection_units",
            "num_plants",
            "num_sample_units",
            "num_inspection_units",
        ]
    ].copy()
    interval_source["plant_pct"] = np.where(interval_source["num_plants"] > 0, (interval_source["total_contaminated_units"] / interval_source["num_plants"]) * 100.0, np.nan)
    interval_source["sample_pct"] = np.where(interval_source["num_sample_units"] > 0, (interval_source["total_contaminated_sample_units"] / interval_source["num_sample_units"]) * 100.0, np.nan)
    interval_source["inspection_pct"] = np.where(interval_source["num_inspection_units"] > 0, (interval_source["total_contaminated_inspection_units"] / interval_source["num_inspection_units"]) * 100.0, np.nan)
    return _apply_intervals_to_table(
        table,
        _intervals_by_scenario(
            interval_source,
            [
                "total_contaminated_units",
                "total_contaminated_sample_units",
                "total_contaminated_inspection_units",
                "plant_pct",
                "sample_pct",
                "inspection_pct",
            ],
        ),
        {
            "total_contaminated_units": "Plant contaminated",
            "total_contaminated_sample_units": "Sample unit contaminated",
            "total_contaminated_inspection_units": "Inspection unit contaminated",
            "plant_pct": "Plant contamination %",
            "sample_pct": "Sample unit contamination %",
            "inspection_pct": "Inspection unit contamination %",
        },
        suffix_map={
            "Plant contamination %": "%",
            "Sample unit contamination %": "%",
            "Inspection unit contamination %": "%",
        },
    )


def _inspection_workload_table(results_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty:
        return None
    required = {
        "name",
        "avg_plant_units_inspected_completion",
        "avg_sample_units_inspected_completion",
        "avg_inspection_units_opened_completion",
        "pct_plant_units_inspected_completion",
        "pct_sample_units_inspected_completion",
        "pct_inspection_units_opened_completion",
    }
    if not required.issubset(results_df.columns):
        return None
    table = results_df[
        [
            "name",
            "avg_plant_units_inspected_completion",
            "avg_sample_units_inspected_completion",
            "avg_inspection_units_opened_completion",
            "pct_plant_units_inspected_completion",
            "pct_sample_units_inspected_completion",
            "pct_inspection_units_opened_completion",
        ]
    ].rename(
        columns={
            "name": "Scenario",
            "avg_plant_units_inspected_completion": "Plants inspected",
            "avg_sample_units_inspected_completion": "Sample units inspected",
            "avg_inspection_units_opened_completion": "Inspection units opened",
            "pct_plant_units_inspected_completion": "Plant units inspected %",
            "pct_sample_units_inspected_completion": "Sample units inspected %",
            "pct_inspection_units_opened_completion": "Inspection units opened %",
        }
    )
    return _apply_intervals_to_table(
        table,
        _intervals_by_scenario(
            results_df,
            [
                "avg_plant_units_inspected_completion",
                "avg_sample_units_inspected_completion",
                "avg_inspection_units_opened_completion",
                "pct_plant_units_inspected_completion",
                "pct_sample_units_inspected_completion",
                "pct_inspection_units_opened_completion",
            ],
        ),
        {
            "avg_plant_units_inspected_completion": "Plants inspected",
            "avg_sample_units_inspected_completion": "Sample units inspected",
            "avg_inspection_units_opened_completion": "Inspection units opened",
            "pct_plant_units_inspected_completion": "Plant units inspected %",
            "pct_sample_units_inspected_completion": "Sample units inspected %",
            "pct_inspection_units_opened_completion": "Inspection units opened %",
        },
        suffix_map={
            "Plant units inspected %": "%",
            "Sample units inspected %": "%",
            "Inspection units opened %": "%",
        },
    )


def _mean_by_name(df: Optional[pd.DataFrame], columns: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "name" not in df.columns:
        return None
    available = [col for col in columns if col in df.columns]
    if not available:
        return None
    return df[["name", *available]].groupby("name", as_index=False).mean(numeric_only=True)


def _level_long_intervals(run_rows: Optional[pd.DataFrame], value_cols: List[str]) -> Optional[pd.DataFrame]:
    if run_rows is None or run_rows.empty:
        return None
    rows = []
    for (scenario, level), group in run_rows.groupby(["Scenario", "Level"]):
        row = {"Scenario": scenario, "Level": level}
        for col in value_cols:
            series = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}__n"] = int(series.size)
            row[f"{col}__lower"] = series.quantile(0.025) if not series.empty else np.nan
            row[f"{col}__upper"] = series.quantile(0.975) if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_long_intervals(table: Optional[pd.DataFrame], interval_df: Optional[pd.DataFrame], value_cols: List[str], *, percent_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if table is None:
        return None
    percent_cols = percent_cols or []
    if interval_df is None:
        return table
    merged = table.merge(interval_df, on=["Scenario", "Level"], how="left")
    for col in value_cols:
        lower_col = f"{col}__lower"
        upper_col = f"{col}__upper"
        n_col = f"{col}__n"

        def _render(row):
            lower = row.get(lower_col)
            upper = row.get(upper_col)
            n = row.get(n_col, 0)
            if pd.isna(n) or int(n) < MIN_REPLICATIONS_FOR_INTERVAL:
                lower = None
                upper = None
            suffix = "%" if col in percent_cols else ""
            return _format_interval_value(row.get(col), lower, upper, suffix=suffix)

        merged[col] = merged.apply(_render, axis=1)
    drop_cols = [col for col in merged.columns if col.endswith("__lower") or col.endswith("__upper") or col.endswith("__n")]
    return _sort_report_levels(merged.drop(columns=drop_cols))


def _slippage_level_report_table(
        results_df: Optional[pd.DataFrame],
        all_runs_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    inspection_summary = _mean_by_name(
        inspection_action_runs_df,
        [
            "total_slipped_inspection_units",
            "consignment_slipped",
            "consignment_intercepted",
            "consignment_clean_inspected",
            "consignment_clean_not_inspected",
            "inspection_clean_inspected",
            "inspection_clean_not_inspected",
            "total_intercepted_inspection_units",
        ],
    )
    sample_summary = _mean_by_name(all_runs_df, ["total_slipped_sample_units", "num_sample_units"])
    rows = []
    for _, row in results_df.iterrows():
        scenario = row["name"]
        plant_total = row.get("num_plants")
        rows.append(
            {
                "Scenario": scenario,
                "Level": "Plant",
                "Slipped": row.get("total_slipped_units"),
                "Slipped %": (row.get("total_slipped_units") / plant_total * 100.0) if plant_total not in (None, 0) else np.nan,
            }
        )
    if sample_summary is not None:
        for _, row in sample_summary.iterrows():
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Sample",
                    "Slipped": row.get("total_slipped_sample_units"),
                    "Slipped %": (row.get("total_slipped_sample_units") / row.get("num_sample_units") * 100.0) if row.get("num_sample_units") not in (None, 0) else np.nan,
                }
            )
    if inspection_summary is not None:
        for _, row in inspection_summary.iterrows():
            consignment_total = sum(
                float(row.get(col) or 0.0)
                for col in ["consignment_slipped", "consignment_intercepted", "consignment_clean_inspected", "consignment_clean_not_inspected"]
            )
            inspection_total = sum(
                float(row.get(col) or 0.0)
                for col in ["total_slipped_inspection_units", "total_intercepted_inspection_units", "inspection_clean_inspected", "inspection_clean_not_inspected"]
            )
            rows.extend(
                [
                    {
                        "Scenario": row["name"],
                        "Level": "Consignment",
                        "Slipped": row.get("consignment_slipped"),
                        "Slipped %": (row.get("consignment_slipped") / consignment_total * 100.0) if consignment_total else np.nan,
                    },
                    {
                        "Scenario": row["name"],
                        "Level": "Inspection",
                        "Slipped": row.get("total_slipped_inspection_units"),
                        "Slipped %": (row.get("total_slipped_inspection_units") / inspection_total * 100.0) if inspection_total else np.nan,
                    },
                ]
            )
    table = pd.DataFrame(rows)
    run_rows = None
    if all_runs_df is not None and "name" in all_runs_df.columns:
        run_rows = []
        for _, row in all_runs_df.iterrows():
            run_rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Plant",
                    "Slipped": row.get("total_slipped_units"),
                    "Slipped %": (row.get("total_slipped_units") / row.get("num_plants") * 100.0) if row.get("num_plants") not in (None, 0) else np.nan,
                }
            )
            if "total_slipped_sample_units" in row.index:
                run_rows.append(
                    {
                        "Scenario": row["name"],
                        "Level": "Sample",
                        "Slipped": row.get("total_slipped_sample_units"),
                        "Slipped %": (row.get("total_slipped_sample_units") / row.get("num_sample_units") * 100.0) if row.get("num_sample_units") not in (None, 0) else np.nan,
                    }
                )
        run_rows = pd.DataFrame(run_rows)
    if inspection_action_runs_df is not None and not inspection_action_runs_df.empty:
        action_run_rows = []
        for _, row in inspection_action_runs_df.iterrows():
            consignment_total = sum(
                float(row.get(col) or 0.0)
                for col in ["consignment_slipped", "consignment_intercepted", "consignment_clean_inspected", "consignment_clean_not_inspected"]
            )
            inspection_total = sum(
                float(row.get(col) or 0.0)
                for col in ["total_slipped_inspection_units", "total_intercepted_inspection_units", "inspection_clean_inspected", "inspection_clean_not_inspected"]
            )
            action_run_rows.extend(
                [
                    {
                        "Scenario": row["name"],
                        "Level": "Consignment",
                        "Slipped": row.get("consignment_slipped"),
                        "Slipped %": (row.get("consignment_slipped") / consignment_total * 100.0) if consignment_total else np.nan,
                    },
                    {
                        "Scenario": row["name"],
                        "Level": "Inspection",
                        "Slipped": row.get("total_slipped_inspection_units"),
                        "Slipped %": (row.get("total_slipped_inspection_units") / inspection_total * 100.0) if inspection_total else np.nan,
                    },
                ]
            )
        action_run_rows = pd.DataFrame(action_run_rows)
        run_rows = pd.concat([run_rows, action_run_rows], ignore_index=True) if run_rows is not None else action_run_rows
    if table.empty:
        return None
    table = _ensure_report_levels(table, ["Slipped", "Slipped %"])
    return _apply_long_intervals(table, _level_long_intervals(run_rows, ["Slipped", "Slipped %"]), ["Slipped", "Slipped %"], percent_cols=["Slipped %"])


def _slippage_level_plot_table(
        results_df: Optional[pd.DataFrame],
        all_runs_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    inspection_summary = _mean_by_name(
        inspection_action_runs_df,
        [
            "total_slipped_inspection_units",
            "consignment_slipped",
            "consignment_intercepted",
            "consignment_clean_inspected",
            "consignment_clean_not_inspected",
            "inspection_clean_inspected",
            "inspection_clean_not_inspected",
            "total_intercepted_inspection_units",
        ],
    )
    sample_summary = _mean_by_name(all_runs_df, ["total_slipped_sample_units"])
    rows = []
    for _, row in results_df.iterrows():
        rows.append(
            {
                "Scenario": row["name"],
                "Level": "Plant",
                "Slipped": row.get("total_slipped_units"),
            }
        )
    if sample_summary is not None:
        for _, row in sample_summary.iterrows():
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Sample",
                    "Slipped": row.get("total_slipped_sample_units"),
                }
            )
    if inspection_summary is not None:
        for _, row in inspection_summary.iterrows():
            rows.extend(
                [
                    {
                        "Scenario": row["name"],
                        "Level": "Consignment",
                        "Slipped": row.get("consignment_slipped"),
                    },
                    {
                        "Scenario": row["name"],
                        "Level": "Inspection",
                        "Slipped": row.get("total_slipped_inspection_units"),
                    },
                ]
            )
    if not rows:
        return None
    return _ensure_report_levels(pd.DataFrame(rows), ["Slipped"])


def _action_level_report_table(
        results_df: Optional[pd.DataFrame],
        all_runs_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    inspection_summary = _mean_by_name(inspection_action_runs_df, ["total_intercepted_inspection_units", "total_slipped_inspection_units"])
    sample_summary = _mean_by_name(all_runs_df, ["total_contaminated_sample_units", "total_slipped_sample_units"])
    rows = []
    for _, row in results_df.iterrows():
        rows.extend(
            [
                {"Scenario": row["name"], "Level": "Consignment", "Intercepted": row.get("intercepted"), "Slipped": row.get("false_neg")},
                {"Scenario": row["name"], "Level": "Plant", "Intercepted": row.get("total_contaminated_units", 0) - row.get("total_slipped_units", 0), "Slipped": row.get("total_slipped_units")},
            ]
        )
    if sample_summary is not None:
        for _, row in sample_summary.iterrows():
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Sample",
                    "Intercepted": row.get("total_contaminated_sample_units", 0) - row.get("total_slipped_sample_units", 0),
                    "Slipped": row.get("total_slipped_sample_units"),
                }
            )
    if inspection_summary is not None:
        for _, row in inspection_summary.iterrows():
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Inspection",
                    "Intercepted": row.get("total_intercepted_inspection_units"),
                    "Slipped": row.get("total_slipped_inspection_units"),
                }
            )
    if not rows:
        return None
    return _ensure_report_levels(pd.DataFrame(rows), ["Intercepted", "Slipped"])


def _contamination_level_report_table(
        results_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    rows = []
    consignment_summary = _mean_by_name(
        inspection_action_runs_df,
        [
            "consignment_intercepted",
            "consignment_slipped",
            "consignment_clean_inspected",
            "consignment_clean_not_inspected",
        ],
    )
    if consignment_summary is not None:
        for _, row in consignment_summary.iterrows():
            contaminated = float(row.get("consignment_intercepted") or 0.0) + float(row.get("consignment_slipped") or 0.0)
            total = contaminated + float(row.get("consignment_clean_inspected") or 0.0) + float(row.get("consignment_clean_not_inspected") or 0.0)
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Consignment",
                    "Contaminated": contaminated,
                    "Contamination %": (contaminated / total * 100.0) if total else np.nan,
                }
            )
    for _, row in results_df.iterrows():
        rows.extend(
            [
                {
                    "Scenario": row["name"],
                    "Level": "Plant",
                    "Contaminated": row.get("total_contaminated_units"),
                    "Contamination %": (row.get("total_contaminated_units") / row.get("num_plants") * 100.0) if row.get("num_plants") not in (None, 0) else np.nan,
                },
                {
                    "Scenario": row["name"],
                    "Level": "Sample",
                    "Contaminated": row.get("total_contaminated_sample_units"),
                    "Contamination %": (row.get("total_contaminated_sample_units") / row.get("num_sample_units") * 100.0) if row.get("num_sample_units") not in (None, 0) else np.nan,
                },
                {
                    "Scenario": row["name"],
                    "Level": "Inspection",
                    "Contaminated": row.get("total_contaminated_inspection_units"),
                    "Contamination %": (row.get("total_contaminated_inspection_units") / row.get("num_inspection_units") * 100.0) if row.get("num_inspection_units") not in (None, 0) else np.nan,
                },
            ]
        )
    return _ensure_report_levels(pd.DataFrame(rows), ["Contaminated", "Contamination %"]) if rows else None


def _inspection_workload_report_table(
        results_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if results_df is None or results_df.empty or "name" not in results_df.columns:
        return None
    rows = []
    consignment_summary = _mean_by_name(
        inspection_action_runs_df,
        [
            "consignment_intercepted",
            "consignment_slipped",
            "consignment_clean_inspected",
            "consignment_clean_not_inspected",
        ],
    )
    for _, row in results_df.iterrows():
        rows.extend(
            [
                {
                    "Scenario": row["name"],
                    "Level": "Plant",
                    "Units inspected": row.get("avg_plant_units_inspected_completion"),
                    "Units inspected %": row.get("pct_plant_units_inspected_completion"),
                },
                {
                    "Scenario": row["name"],
                    "Level": "Sample",
                    "Units inspected": row.get("avg_sample_units_inspected_completion"),
                    "Units inspected %": row.get("pct_sample_units_inspected_completion"),
                },
                {
                    "Scenario": row["name"],
                    "Level": "Inspection",
                    "Units inspected": row.get("avg_inspection_units_opened_completion"),
                    "Units inspected %": row.get("pct_inspection_units_opened_completion"),
                },
            ]
        )
    if consignment_summary is not None and "num_inspections" in results_df.columns:
        for _, row in consignment_summary.iterrows():
            total = sum(
                float(row.get(col) or 0.0)
                for col in ["consignment_intercepted", "consignment_slipped", "consignment_clean_inspected", "consignment_clean_not_inspected"]
            )
            inspected = float(
                results_df.loc[results_df["name"] == row["name"], "num_inspections"].iloc[0]
            ) if not results_df.loc[results_df["name"] == row["name"]].empty else np.nan
            rows.append(
                {
                    "Scenario": row["name"],
                    "Level": "Consignment",
                    "Units inspected": inspected,
                    "Units inspected %": (inspected / total * 100.0) if total and not pd.isna(inspected) else np.nan,
                }
            )
    return _ensure_report_levels(pd.DataFrame(rows), ["Units inspected", "Units inspected %"]) if rows else None


def _add_plot_if_available(document, fig) -> None:
    if fig is None:
        return
    try:
        _add_plot_image(document, fig)
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            pass


def _build_report_document(
        *,
        results_df: Optional[pd.DataFrame],
        all_runs_df: Optional[pd.DataFrame],
        scenario_table_df: Optional[pd.DataFrame],
        inspection_action_runs_df: Optional[pd.DataFrame],
        metadata: Optional[Dict[str, Any]],
        title: str,
):
    docx, alignment, _ = _require_python_docx()
    document = docx.Document()
    heading = document.add_heading(title, level=0)
    heading.alignment = alignment.CENTER
    document.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if metadata:
        document.add_heading("Run Metadata", level=1)
        _add_key_value_table(document, list(metadata.items()))

    document.add_heading("Run Summary", level=1)
    run_summary = _run_summary_rows(results_df, all_runs_df)
    if run_summary:
        _add_key_value_table(document, run_summary)
    else:
        document.add_paragraph("No summary metrics were available for the report.")

    if scenario_table_df is not None:
        document.add_heading("Scenario Details", level=1)
        document.add_paragraph(
            "The table below summarizes the scenario setup that was executed, including the main input files and contamination settings when available."
        )
        _add_dataframe_table(document, _scenario_details_table(scenario_table_df))

    document.add_heading("Slippage Level", level=1)
    document.add_paragraph(
        "This section lists the mean slipped counts by scenario across the Consignment, Inspection, Sample, and Plant levels."
    )
    slippage_df = _slippage_level_report_table(results_df, all_runs_df, inspection_action_runs_df)
    _add_dataframe_table(document, slippage_df)
    slippage_plot_df = _slippage_level_plot_table(results_df, all_runs_df, inspection_action_runs_df)
    if slippage_plot_df is not None and {"Scenario", "Level", "Slipped"}.issubset(slippage_plot_df.columns):
        _add_plot_if_available(
            document,
            _build_faceted_bar_plot(
                slippage_plot_df,
                title="Slippage by Scenario",
                x_col="Scenario",
                y_col="Slipped",
                facet_col="Level",
                color="#d62728",
                facet_order=REPORT_LEVEL_ORDER,
                ylabel="Slipped units",
            ),
        )

    document.add_heading("Action Level", level=1)
    document.add_paragraph(
        "This section lists intercepted versus slipped outcomes by scenario across the Consignment, Inspection, Sample, and Plant levels."
    )
    action_df = _action_level_report_table(results_df, all_runs_df, inspection_action_runs_df)
    _add_dataframe_table(document, action_df)
    if action_df is not None and {"Scenario", "Level", "Intercepted", "Slipped"}.issubset(action_df.columns):
        plot_cols = [col for col in ["Intercepted", "Slipped"] if col in action_df.columns]
        if plot_cols:
            _add_plot_if_available(
                document,
                _build_faceted_grouped_plot(
                    action_df,
                    title="Action Outcomes by Scenario",
                    x_col="Scenario",
                    value_cols=plot_cols,
                    ylabel="Units",
                    facet_col="Level",
                    facet_order=REPORT_LEVEL_ORDER,
                ),
            )

    document.add_heading("Contamination Level", level=1)
    document.add_paragraph(
        "This section summarizes contaminated counts and contamination percentages by scenario across the Consignment, Inspection, Sample, and Plant levels."
    )
    contamination_df = _contamination_level_report_table(results_df, inspection_action_runs_df)
    _add_dataframe_table(document, contamination_df)
    if contamination_df is not None and {"Scenario", "Level"}.issubset(contamination_df.columns):
        plot_cols = [
            col
            for col in ["Contamination %"]
            if col in contamination_df.columns
        ]
        if plot_cols:
            _add_plot_if_available(
                document,
                _build_faceted_grouped_plot(
                    contamination_df,
                    title="Contamination Percentages by Scenario",
                    x_col="Scenario",
                    value_cols=plot_cols,
                    ylabel="Percent contaminated",
                    facet_col="Level",
                    facet_order=REPORT_LEVEL_ORDER,
                ),
            )

    document.add_heading("Inspection Workload Level", level=1)
    document.add_paragraph(
        "This section summarizes the inspection effort by scenario, including inspected counts and the corresponding percentages."
    )
    workload_df = _inspection_workload_report_table(results_df, inspection_action_runs_df)
    _add_dataframe_table(document, workload_df)
    if workload_df is not None and {"Scenario", "Level"}.issubset(workload_df.columns):
        plot_cols = [
            col
            for col in ["Units inspected %"]
            if col in workload_df.columns
        ]
        if plot_cols:
            _add_plot_if_available(
                document,
                _build_faceted_grouped_plot(
                    workload_df,
                    title="Inspection Workload Percentages by Scenario",
                    x_col="Scenario",
                    value_cols=plot_cols,
                    ylabel="Percent inspected",
                    facet_col="Level",
                    facet_order=REPORT_LEVEL_ORDER,
                ),
            )

    return document


def build_run_report_docx_bytes(
        *,
        results_df: Optional[pd.DataFrame] = None,
        all_runs_df: Optional[pd.DataFrame] = None,
        scenario_table_df: Optional[pd.DataFrame] = None,
        inspection_action_runs_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title: str = "Simulation Run Report",
) -> bytes:
    document = _build_report_document(
        results_df=results_df,
        all_runs_df=all_runs_df,
        scenario_table_df=scenario_table_df,
        inspection_action_runs_df=inspection_action_runs_df,
        metadata=metadata,
        title=title,
    )
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def export_run_report_to_word(
        *,
        report_path: str | Path,
        results_df: Optional[pd.DataFrame] = None,
        all_runs_df: Optional[pd.DataFrame] = None,
        scenario_table_df: Optional[pd.DataFrame] = None,
        inspection_action_runs_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title: str = "Simulation Run Report",
) -> Path:
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _build_report_document(
        results_df=results_df,
        all_runs_df=all_runs_df,
        scenario_table_df=scenario_table_df,
        inspection_action_runs_df=inspection_action_runs_df,
        metadata=metadata,
        title=title,
    )
    document.save(output_path)
    return output_path


def export_run_report_from_output_dir(
        output_dir: str | Path,
        *,
        report_path: Optional[str | Path] = None,
        scenario_table_path: Optional[str | Path] = None,
        inspection_action_runs_path: Optional[str | Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title: str = "Simulation Run Report",
) -> Path:
    output_root = Path(output_dir)
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_root}")

    results_df = _read_csv_if_exists(output_root / "pis_contamination_scenario_results.csv")
    all_runs_df = _read_csv_if_exists(output_root / "all_runs.csv")
    if inspection_action_runs_path is None:
        inspection_action_runs_path = output_root / "inspection_action_runs.csv"
    inspection_action_runs_df = _read_csv_if_exists(Path(inspection_action_runs_path))
    if scenario_table_path is None:
        scenario_table_path = output_root.parent / "scenario_table.csv"
    scenario_table_df = _read_csv_if_exists(Path(scenario_table_path))

    if report_path is None:
        report_path = output_root / "simulation_run_report.docx"

    merged_metadata = {
        "output_dir": str(output_root),
        "results_file": str(output_root / "pis_contamination_scenario_results.csv"),
    }
    if metadata:
        merged_metadata.update(metadata)

    return export_run_report_to_word(
        report_path=report_path,
        results_df=results_df,
        all_runs_df=all_runs_df,
        scenario_table_df=scenario_table_df,
        inspection_action_runs_df=inspection_action_runs_df,
        metadata=merged_metadata,
        title=title,
    )
