from pathlib import Path
from typing import Optional
import shutil

import pandas as pd
import streamlit as st
import plotly.express as px


from gui.models import init_state
from gui.navigation import render_sidebar_navigation
from gui.slippage_ui import get_slippage_state, set_paths, create_default_paths


st.set_page_config(
    page_title="Inspection Process",
    page_icon=":mag_right:",
    layout="wide",
)
init_state()

state = get_slippage_state()
render_sidebar_navigation()
paths = state["paths"]
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)
COMPLIANCE_ROOT = TMP_DIR / "compliance"
COMPLIANCE_ROOT.mkdir(parents=True, exist_ok=True)


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


st.title("Page 3 - Inspection Process")
st.warning(
    "**Test Deployment Notice: This is a test deployment with limited functionality and is under active development. "
    "Features may be incomplete and subject to change. Results have not been validated.**"
)
st.caption(
    "Ingest the compliance lookup table, configure low/medium/high types, and review RBS parameters."
)

tabs = st.tabs(["Upload table", "Create policy manually", "Saved compliance tables"])

with tabs[0]:
    st.subheader("Compliance table (upload)")
    compliance_upload = st.file_uploader("Upload compliance table CSV", type=["csv"])
    if compliance_upload is not None:
        try:
            compliance_upload.seek(0)
            upload_preview = pd.read_csv(compliance_upload)
            st.dataframe(upload_preview.head(50), use_container_width=True)
            selected_column = _pick_compliance_column(upload_preview)
            counts = _compliance_level_counts(upload_preview[selected_column]) if selected_column else {"Low": 0, "Medium": 0, "High": 0}
            st.markdown("**Summary**")
            stat_cols = st.columns(4)
            stat_cols[0].metric("Rows", f"{len(upload_preview)}")
            stat_cols[1].metric("Number of Low Compliance Items", counts["Low"])
            stat_cols[2].metric("Number of Medium Compliance Items", counts["Medium"])
            stat_cols[3].metric("Number of High Compliance Items", counts["High"])
        except Exception:  # pylint: disable=broad-except
            st.info("Unable to preview upload.")
    save_name = st.text_input("Save as name", value="compliance_table")
    if compliance_upload is not None and st.button("Save uploaded table", type="secondary"):
        try:
            compliance_upload.seek(0)
            new_compliance_df = pd.read_csv(compliance_upload)
            target_path = COMPLIANCE_ROOT / f"{save_name}.csv"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            new_compliance_df.to_csv(target_path, index=False)
            set_paths(compliance_lookup=target_path)
            st.success(f"Compliance table saved to {target_path}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to save compliance table: {exc}")

with tabs[1]:
    st.subheader("Compliance table (manual)")

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
    manual_name = st.text_input("Save as name (manual)", value="manual_compliance_table")
    if manual_df is not None and st.button("Save manual compliance table", type="secondary"):
        target_path = COMPLIANCE_ROOT / f"{manual_name}.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        manual_df.to_csv(target_path, index=False)
        set_paths(compliance_lookup=target_path)
        st.success(f"Manual compliance table saved to {target_path}")

with tabs[2]:
    st.subheader("Saved compliance tables")
    saved_files = sorted(COMPLIANCE_ROOT.glob("*.csv"))
    if not saved_files:
        st.info("No compliance tables saved yet in tmp/compliance.")
    else:
        sel = st.selectbox("Select a saved compliance table", saved_files, format_func=lambda p: p.name)
        st.caption(f"Location: {sel}")
        compliance_df = _load_compliance_table(sel)
        if compliance_df is not None:
            st.dataframe(compliance_df, use_container_width=True, height=280)
            selected_column = _pick_compliance_column(compliance_df)
            counts = _compliance_level_counts(compliance_df[selected_column]) if selected_column else {"Low": 0, "Medium": 0, "High": 0}
            stat_cols = st.columns(3)
            stat_cols[0].metric("Low type rows", counts["Low"])
            stat_cols[1].metric("Medium type rows", counts["Medium"])
            stat_cols[2].metric("High type rows", counts["High"])

            origin_col = None
            for cand in compliance_df.columns:
                if "origin" in cand.lower():
                    origin_col = cand
                    break
            compliance_filter = st.selectbox("Filter by compliance level for map", ["All", "Low", "Medium", "High"])
            df_map = compliance_df.copy()
            if selected_column and compliance_filter != "All":
                df_map = df_map[df_map[selected_column].astype(str).str.lower().str.contains(compliance_filter.lower())]
            if origin_col is None:
                st.info("No origin column found to plot on map.")
            elif df_map.empty:
                st.info("No rows match the selected compliance filter.")
            else:
                grouped = df_map.groupby(origin_col).size().reset_index(name="records")
                if px is None:
                    st.info("Plotly is not available to render the map.")
                else:
                    fig = px.choropleth(
                        grouped,
                        locations=origin_col,
                        locationmode="country names",
                        color="records",
                        hover_name=origin_col,
                        color_continuous_scale="YlGnBu",
                        title="Rows by origin (filtered by compliance level)",
                    )
                    fig.update_layout(
                        height=600,
                        margin=dict(l=0, r=0, t=50, b=0),
                        geo=dict(
                            showcoastlines=True,
                            coastlinecolor="gray",
                            showcountries=True,
                            showland=True,
                            landcolor="#f7f7f7",
                            scope="world",
                            projection_type="equirectangular",
                        ),
                        dragmode=False,
                    )
                    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
                    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": True})
        else:
            st.info("Unable to preview this file.")

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
