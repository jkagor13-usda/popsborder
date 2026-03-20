# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

from pathlib import Path
from typing import Optional
import pickle
import shutil

from gui.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import pandas as pd
import streamlit as st
import plotly.express as px


from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.page_styles import apply_shared_page_styles, render_page_intro
from gui.slippage_ui import get_slippage_state, set_paths, create_default_paths
from popsborder.inputs import build_compliance_lookup_table, load_compliance_lookup_csv
from popsborder.inspections import normalize_rbs_variables_using_risk_unit_config


st.set_page_config(
    page_title="Inspection Process",
    page_icon=":mag_right:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
apply_shared_page_styles()
paths = state["paths"]
state.setdefault("compliance_mapping_path", Path("data_input/compliance_mapping_detection_confidence_levels.csv"))
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)
COMPLIANCE_ROOT = TMP_DIR / "compliance"
COMPLIANCE_ROOT.mkdir(parents=True, exist_ok=True)
COMPLIANCE_SOURCE_ROOT = COMPLIANCE_ROOT / "_sources"
COMPLIANCE_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)


def _load_compliance_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def _compliance_level_counts(series: pd.Series):
    if series is None:
        return {"Low": 0, "Medium": 0, "High": 0}
    normalized = series.fillna("").astype(str).str.lower().str.strip()
    return {
        "Low": normalized.str.contains("low").sum(),
        "Medium": normalized.str.contains("med").sum(),
        "High": normalized.str.contains("high").sum(),
    }


def _pick_compliance_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "compliance" in col.lower():
            return col
    string_columns = [col for col in df.columns if df[col].dtype == "object"]
    return string_columns[0] if string_columns else None


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _compliance_cell_style(value: object) -> str:
    text = str(value).strip().lower()
    if "tissue" in text:
        return "background-color: #6a994e; color: white; font-weight: 600;"
    if "high" in text:
        return "background-color: #386641; color: white; font-weight: 600;"
    if "medium" in text:
        return "background-color: #f2cc8f; color: #222; font-weight: 600;"
    if "low" in text:
        return "background-color: #f4a261; color: #222; font-weight: 600;"
    if "poor" in text:
        return "background-color: #bc4749; color: white; font-weight: 600;"
    return ""


def _numeric_heat_style(value: object, base_color: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    numeric = max(0.0, min(1.0, numeric))
    alpha = 0.15 + (0.75 * numeric)
    return f"background-color: rgba({base_color}, {alpha:.3f}); font-weight: 600;"


def _save_policy_artifact(
    policy_name: str,
    compliance_csv_path: Path,
    mapping_csv_path: Optional[Path] = None,
    *,
    direct_lookup_csv: bool = False,
) -> Path:
    policy_name = policy_name.strip() or "rbs_compliance_policy"
    policy_path = COMPLIANCE_ROOT / f"{policy_name}.pkl"
    if direct_lookup_csv:
        compliance_table = load_compliance_lookup_csv(compliance_csv_path)
        preview_df = pd.read_csv(compliance_csv_path)
    else:
        if mapping_csv_path is None or not mapping_csv_path.exists():
            raise FileNotFoundError("Compliance mapping detection/confidence file is required.")
        compliance_table = build_compliance_lookup_table(
            compliance_table_filepath=compliance_csv_path,
            mapping_filepath=mapping_csv_path,
        )
        preview_df = pd.read_csv(compliance_csv_path).merge(
            pd.read_csv(mapping_csv_path),
            on="Compliance",
            how="left",
        )
    updated_vars, _, _ = normalize_rbs_variables_using_risk_unit_config(
        compliance_table.get("rbs_variables", [])
    )
    compliance_table["rbs_variables"] = updated_vars
    compliance_table["_policy_preview"] = preview_df.to_dict(orient="records")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(policy_path, "wb") as handle:
        pickle.dump(compliance_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return policy_path


st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.title("Page 3 - Inspection Process")
render_page_intro(
    "Ingest the compliance lookup table, configure low/medium/high types, and review RBS parameters."
)

tabs = st.tabs(["Upload table", "Create policy manually", "Saved RBS compliance policy"])

with tabs[0]:
    st.subheader("Define RBS Compliance Policy")
    
    st.markdown("**Compliance level upload**")
    compliance_upload = st.file_uploader(
        "Upload CSV with compliance levels",
        type=["csv"],
        key="base_compliance_upload",
    )
    upload_preview = None
    if compliance_upload is not None:
        try:
            compliance_upload.seek(0)
            upload_preview = pd.read_csv(compliance_upload)
        except Exception:  # pylint: disable=broad-except
            st.info("Unable to preview upload.")
            upload_preview = None
    if upload_preview is not None:
        selected_column = _pick_compliance_column(upload_preview)
        origin_col = _first_matching_column(upload_preview, ["Origin Location Country Name", "Origin"])
        material_col = _first_matching_column(upload_preview, ["Propagative Material type", "PM Type"])
        display_cols = [c for c in [origin_col, material_col, selected_column] if c]
        styled_base = upload_preview[display_cols].copy() if display_cols else upload_preview.copy()
        if selected_column and selected_column in styled_base.columns:
            styled_base = styled_base.style.applymap(_compliance_cell_style, subset=[selected_column])
        st.markdown("**Compliance policy table**")
        st.dataframe(styled_base, use_container_width=True, height=360)

    st.markdown("**Detection/confidence mapping upload**")
    mapping_upload = st.file_uploader(
        "Upload CSV with compliance level mapping of detection/confidence values",
        type=["csv"],
        key="compliance_mapping_upload",
    )
    mapping_preview = None
    if mapping_upload is not None:
        try:
            mapping_upload.seek(0)
            mapping_preview = pd.read_csv(mapping_upload)
        except Exception:  # pylint: disable=broad-except
            st.info("Unable to preview mapping upload.")
            mapping_preview = None

    if mapping_preview is not None:
        detection_col = _first_matching_column(mapping_preview, ["Detection Level"])
        confidence_col = _first_matching_column(mapping_preview, ["Confidence Levels", "Confidence Level"])
        compliance_col = _first_matching_column(mapping_preview, ["Compliance"])
        if compliance_col and detection_col and confidence_col:
            mapping_table = mapping_preview[[compliance_col, detection_col, confidence_col]].copy()
            mapping_table.columns = ["Compliance", "Detection Level", "Confidence Level"]
            styled_mapping = (
                mapping_table.style
                .applymap(_compliance_cell_style, subset=["Compliance"])
                .applymap(lambda v: _numeric_heat_style(v, "31, 119, 180"), subset=["Detection Level"])
                .applymap(lambda v: _numeric_heat_style(v, "76, 149, 108"), subset=["Confidence Level"])
                .format({"Detection Level": "{:.2f}", "Confidence Level": "{:.2f}"})
            )
            st.markdown("**Detection/confidence mapping policy table**")
            st.dataframe(styled_mapping, use_container_width=True, height=260)
    save_name = st.text_input("Save as name", value="rbs_compliance_policy")
    can_save_uploaded_policy = compliance_upload is not None and mapping_upload is not None
    if st.button("Save policy", type="secondary", disabled=not can_save_uploaded_policy):
        try:
            if compliance_upload is None or mapping_upload is None:
                raise ValueError("Both the compliance table and the detection/confidence mapping file are required.")
            compliance_upload.seek(0)
            new_compliance_df = pd.read_csv(compliance_upload)
            compliance_csv_path = COMPLIANCE_SOURCE_ROOT / f"{save_name}_compliance_table.csv"
            compliance_csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_compliance_df.to_csv(compliance_csv_path, index=False)

            mapping_upload.seek(0)
            mapping_df = pd.read_csv(mapping_upload)
            mapping_path = COMPLIANCE_SOURCE_ROOT / f"{save_name}_mapping.csv"
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_df.to_csv(mapping_path, index=False)
            state["compliance_mapping_path"] = mapping_path

            policy_path = _save_policy_artifact(save_name, compliance_csv_path, mapping_path)
            set_paths(compliance_lookup=policy_path)
            st.success(f"Saved RBS compliance policy to {policy_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save compliance table: {exc}")

with tabs[1]:
    st.subheader("Manual Creation of RBS Compliance Policy ")

    rbs_path = state["paths"].rbs_data
    rbs_cols = []
    if rbs_path and Path(rbs_path).exists():
        try:
            rbs_df = pd.read_csv(rbs_path, nrows=2000)
            rbs_cols = [c for c in rbs_df.columns if rbs_df[c].dtype == "object"]
        except Exception:  # pylint: disable=broad-except
            rbs_cols = []
    if not rbs_cols:
        rbs_cols = ["Origin Location Country Name", "Propagative Material type", "Pathway", "Inspection Location"]

    multi_cols = st.multiselect("Select feature columns to combine", options=rbs_cols, default=rbs_cols[:2])
    selections = []
    for col_name in multi_cols:
        values = []
        if rbs_path and Path(rbs_path).exists():
            try:
                values = sorted(rbs_df[col_name].dropna().astype(str).unique().tolist())
            except Exception:  # pylint: disable=broad-except
                values = []
        selected_vals = st.multiselect(f"Values for {col_name}", options=values or [], key=f"comb_vals_{col_name}")
        selections.append({"column": col_name, "values": selected_vals})

    level_choice = st.selectbox("Compliance level for the new rows", ["Low", "Medium", "High"])

    detection_value = st.slider(
        "Detection Level",
        min_value=0.0,
        max_value=1.,
        step=.01,
        value=0.1,          
        help="Detection Level for Hypergeometric Sampling."
    )

    confidence_value = st.slider(
        "Confidence Level",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Confidence Level for Hypergeometric Sampling."
    )

    if st.button("Add rows to compliance table", type="primary"):
        rows = []
        selected_cols = [s["column"] for s in selections]
        from itertools import product

        value_lists = []
        for sel in selections:
            vals = sel.get("values", [])
            if not vals:
                value_lists = []
                break
            value_lists.append(vals)
        if not value_lists:
            st.warning("Select at least one value for each chosen feature before adding rows.")
        else:
            for combo in product(*value_lists):
                row = {c: "" for c in selected_cols}
                for col, val in zip(selected_cols, combo):
                    row[col] = val
                row["Compliance"] = level_choice
                row["Detection Level"] = detection_value
                row["Confidence Levels"] = confidence_value
                rows.append(row)
            manual_df = state.get("manual_compliance_df")
            new_df = pd.DataFrame(rows)
            if manual_df is None:
                manual_df = new_df
            else:
                # Align columns
                all_cols = list(set(manual_df.columns).union(set(new_df.columns)))
                manual_df = manual_df.reindex(columns=all_cols, fill_value="")
                new_df = new_df.reindex(columns=all_cols, fill_value="")
                manual_df = pd.concat([manual_df, new_df], ignore_index=True)
            state["manual_compliance_df"] = manual_df
            st.success(f"Added {len(rows)} rows at level {level_choice}.")
            st.dataframe(manual_df, use_container_width=True)


    manual_df = state.get("manual_compliance_df")
    manual_name = st.text_input("Save as name (manual)", value="manual_compliance_policy")
    can_save_manual_policy = manual_df is not None and not manual_df.empty
    if st.button("Save manual policy", type="secondary", disabled=not can_save_manual_policy):
        target_path = COMPLIANCE_SOURCE_ROOT / f"{manual_name}.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        manual_df.to_csv(target_path, index=False)
        policy_path = _save_policy_artifact(manual_name, target_path, direct_lookup_csv=True)
        set_paths(compliance_lookup=policy_path)
        st.success(f"Manual RBS compliance policy saved to {policy_path}")

with tabs[2]:
    st.subheader("Saved RBS compliance policy")
    saved_files = sorted(COMPLIANCE_ROOT.glob("*.pkl"))
    if not saved_files:
        st.info("No RBS compliance policies saved yet in tmp/compliance.")
    else:
        sel = st.selectbox("Select a saved RBS compliance policy", saved_files, format_func=lambda p: p.name)
        st.caption(f"Location: {sel}")
        try:
            with open(sel, "rb") as handle:
                policy = pickle.load(handle)
            rbs_variables = policy.get("rbs_variables", [])
            st.caption(f"RBS variables: {', '.join(rbs_variables) if rbs_variables else 'none'}")
            preview_rows = policy.get("_policy_preview")
            if preview_rows:
                policy_df = pd.DataFrame(preview_rows)
                compliance_col = _first_matching_column(policy_df, ["Compliance"])
                detection_col = _first_matching_column(policy_df, ["Detection Level"])
                confidence_col = _first_matching_column(policy_df, ["Confidence Levels", "Confidence Level"])
                styled_policy = policy_df.style
                if compliance_col:
                    styled_policy = styled_policy.applymap(_compliance_cell_style, subset=[compliance_col])
                if detection_col:
                    styled_policy = styled_policy.applymap(
                        lambda v: _numeric_heat_style(v, "31, 119, 180"),
                        subset=[detection_col],
                    )
                if confidence_col:
                    styled_policy = styled_policy.applymap(
                        lambda v: _numeric_heat_style(v, "76, 149, 108"),
                        subset=[confidence_col],
                    )
                st.dataframe(styled_policy, use_container_width=True, height=360)
            else:
                policy_rows = []
                for key, value in policy.items():
                    if key in {"rbs_variables", "_policy_preview"}:
                        continue
                    if isinstance(key, tuple):
                        row = {rbs_variables[idx]: key[idx] for idx in range(min(len(rbs_variables), len(key)))}
                        row["Detection Level"] = value[0] if isinstance(value, tuple) and len(value) > 0 else ""
                        row["Confidence Level"] = value[1] if isinstance(value, tuple) and len(value) > 1 else ""
                        policy_rows.append(row)
                policy_df = pd.DataFrame(policy_rows)
                if not policy_df.empty:
                    st.dataframe(policy_df, use_container_width=True, height=320)
        except Exception as exc:  # pylint: disable=broad-except
            st.info(f"Unable to preview this policy file: {exc}")

st.divider()
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button(
        "Reset and Return Home",
        type="secondary",
        key="nav_reset_page4",
        help="Delete temporary files and restart from the home page",
    ):
        try:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            st.session_state.clear()
            state["paths"] = create_default_paths()
            st.switch_page("frontend.py")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to reset temporary files: {exc}")
with nav_cols[1]:
    if st.button("Previous Page", type="primary", key="nav_back_page3_proc"):
        st.switch_page("pages/2_Contamination_Fit.py")
with nav_cols[2]:
    if st.button("Next Page", type="primary", key="nav_forward_page5_proc"):
        st.switch_page("pages/4_Scenario_Experiments.py")

st.info("Inspection scenarios are managed on the experiment setup page.")
