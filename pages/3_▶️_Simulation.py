import time
import numpy as np
import streamlit as st
from lib.models import init_state
from lib.sim import run_simulation

st.set_page_config(page_title="Simulation", page_icon="▶️", layout="wide")
init_state()

st.title("▶️ Simulation")
st.caption("Run the (wireframe) model and explore outcomes. Replace with real engine later.")

s = st.session_state.scenario
i = st.session_state.inspection

with st.sidebar:
    st.subheader("Execution")
    autorun = st.checkbox("Auto-run on open", value=True)
    run_btn = st.button("Run simulation now", use_container_width=True)

if autorun or run_btn:
    with st.spinner("Running wireframe simulation..."):
        time.sleep(0.2)
        out = run_simulation(s, i)
else:
    out = run_simulation(s, i)  # lightweight; keeps charts populated

raw = out["raw"]

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean contaminated / run", f"{out['mean_contaminated']:.0f}")
k2.metric("Mean detected / run", f"{out['mean_detected']:.1f}")
k3.metric("Mean missed / run", f"{out['mean_missed']:.1f}")
k4.metric("Reject probability", f"{100*out['p_reject']:.1f}%")

st.markdown(
    f"""
**Items sampled/run:** {out['items_sampled_per_run']:,}  
**Sampling fraction:** {out['sampling_fraction_pct']:.3f}%  
**Baseline detection probability (wireframe):** {out['base_detection_prob']:.4f}  
    """.strip()
)

tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Table / Export"])

with tab1:
    st.subheader("Detection vs. Misses (mean per run)")
    agg = raw[["detected", "missed"]].mean().round(1)
    st.bar_chart(agg, height=280)

    st.subheader("Reject decision (rolling mean)")
    st.area_chart(raw["rejected"].rolling(10, min_periods=1).mean(), height=200)

with tab2:
    st.subheader("Outcome distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Detected (histogram)")
        st.bar_chart(np.histogram(raw["detected"], bins=20)[0])
    with c2:
        st.write("Missed (histogram)")
        st.bar_chart(np.histogram(raw["missed"], bins=20)[0])
    with c3:
        st.write("Rejected (rolling mean)")
        st.line_chart(raw["rejected"].rolling(25, min_periods=1).mean())

with tab3:
    st.subheader("Run-level outcomes")
    st.dataframe(raw.head(500), use_container_width=True)
    st.download_button(
        "Download CSV",
        data=raw.to_csv(index=False),
        file_name="simulation_runs.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()
st.markdown(
    """
**Note:** Results here are placeholders for UI/flow testing.  
Swap the engine in `lib/sim.py` with your real PoPS Border implementation when ready.
"""
)
