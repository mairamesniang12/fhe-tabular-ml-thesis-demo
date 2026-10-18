"""
Dashboard tab for the Streamlit app.

Every number below is read from CSV files in data/, which contain the
experimental results used in Chapter 5. This module only visualizes them;
it does not run new experiments or measurements.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA = os.path.join(os.path.dirname(__file__), "data")


def _csv(name):
    """Read one experiment-results CSV from the data directory."""
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required dashboard data file not found: {path}"
        )
    return pd.read_csv(path)


# ---- Part A: Comparative benchmarking ----
_fhe = _csv("fhe_results_final.csv").merge(
    _csv("plain_matched.csv")[["dataset", "model", "acc_plain"]],
    on=["dataset", "model"],
)
PART_A = _fhe.rename(
    columns={
        "acc_fhe_simulate_full": "acc_fhe_sim",
        "latency_fhe_mean_s": "latency",
    }
)[
    [
        "dataset",
        "model",
        "acc_plain",
        "acc_fhe_sim",
        "latency",
        "latency_fhe_std_s",
    ]
]


# ---- Part B: Sensitivity sweeps ----
_sw = _csv("sweep_depth_bits.csv")

DEPTH_DATA = _sw[_sw.sweep == "depth"].rename(
    columns={
        "value": "depth",
        "acc_fhe_simulate_full": "acc_fhe_sim",
        "latency_fhe_mean_s": "latency",
    }
)

BITS_DATA = _sw[_sw.sweep == "bits"].rename(
    columns={
        "value": "bits",
        "acc_fhe_simulate_full": "acc_fhe_sim",
        "latency_fhe_mean_s": "latency",
    }
)

# The Pima T=100 run was saved in a separate file (see Chapter 5).
_nt = [_csv("sweep_ntrees.csv")]
if os.path.isfile(os.path.join(DATA, "sweep_ntrees_pima_T100.csv")):
    _nt.append(_csv("sweep_ntrees_pima_T100.csv"))

NTREES_DATA = pd.concat(_nt, ignore_index=True).rename(
    columns={
        "value": "n_trees",
        "acc_fhe_simulate_full": "acc_fhe_sim",
        "latency_fhe_mean_s": "latency",
    }
)


# ---- Part C: Deployment, differential privacy, and H1 ----
DP_DATA = _csv("dp_simulation_results.csv")
DP_DATA = DP_DATA.rename(
    columns={
        "acc_dp_mean": "acc_mean",
        "acc_dp_std": "acc_std",
    }
)

_q = _csv("deployment_queries_cached.csv")
DEPLOYMENT = {
    "Client encryption": _q.encryption_s.mean(),
    "HTTP/request-processing overhead": _q.overhead_non_fhe_s.mean(),
    "Server-side call (FHEModelServer.run)": _q.server_inference_s.mean(),
    "Client decryption": _q.decryption_s.mean(),
}


# ============================== helpers ==============================

def _with_ci_errors(df):
    """Add asymmetric error-bar columns from the stored 95% CI."""
    df = df.copy()
    df["err_plus"] = (df["latency_ci95_high_s"] - df["latency"]).clip(lower=0)
    df["err_minus"] = (df["latency"] - df["latency_ci95_low_s"]).clip(lower=0)
    return df


def _sweep_charts(df, x, x_label):
    d = _with_ci_errors(df).sort_values(["dataset", x])
    fig_lat = px.line(
        d, x=x, y="latency", color="dataset", markers=True,
        error_y="err_plus", error_y_minus="err_minus",
        labels={x: x_label, "latency": "FHE latency (s/sample)"},
    )
    fig_acc = px.line(
        d, x=x, y="acc_fhe_sim", color="dataset", markers=True,
        labels={x: x_label, "acc_fhe_sim": "FHE-simulate accuracy"},
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_lat, use_container_width=True)
    c2.plotly_chart(fig_acc, use_container_width=True)


# ============================== UI ==============================

def render_dashboard():
    st.caption(
        "All values are read from the result files of Chapter 5 "
        "(CSV files in data/). No new experiment is run here."
    )

    tab_a, tab_b, tab_c = st.tabs(
        ["Part A: Benchmark", "Part B: Sensitivity", "Part C: Deployment & DP"]
    )

    # ---------------- Part A ----------------
    with tab_a:
        st.subheader("Decision Tree, Random Forest and XGBoost under FHE")
        st.caption(
            "Accuracy on the full test sets (plaintext and FHE-simulate); "
            "latency: real-FHE time per sample, mean ± standard deviation "
            "over repeated runs."
        )

        a = PART_A.copy()
        st.dataframe(
            a.rename(
                columns={
                    "acc_plain": "Acc. plain",
                    "acc_fhe_sim": "Acc. FHE-sim",
                    "latency": "Latency (s)",
                    "latency_fhe_std_s": "Latency std (s)",
                }
            ).round(4),
            use_container_width=True,
            hide_index=True,
        )

        fig_lat = px.bar(
            a, x="dataset", y="latency", color="model", barmode="group",
            error_y="latency_fhe_std_s",
            labels={"latency": "FHE latency (s/sample)", "dataset": "Dataset"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        acc_long = a.melt(
            id_vars=["dataset", "model"],
            value_vars=["acc_plain", "acc_fhe_sim"],
            var_name="mode", value_name="accuracy",
        )
        acc_long["mode"] = acc_long["mode"].map(
            {"acc_plain": "Plaintext", "acc_fhe_sim": "FHE-simulate"}
        )
        acc_long["config"] = acc_long["dataset"] + " / " + acc_long["model"]
        fig_acc = px.bar(
            acc_long, x="config", y="accuracy", color="mode", barmode="group",
            labels={"config": "Dataset / model", "accuracy": "Accuracy"},
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # ---------------- Part B ----------------
    with tab_b:
        st.subheader("Hyperparameter analysis (WDBC and Pima)")
        st.caption(
            "Latency: mean with 95% confidence interval (run-to-run timing "
            "noise only). Accuracy: FHE-simulate on the full test set."
        )

        st.markdown("**Tree depth D** (Decision Tree, p = 5)")
        _sweep_charts(DEPTH_DATA, "depth", "Tree depth D")

        st.markdown("**Number of trees T** (Random Forest, D = 4, p = 5)")
        _sweep_charts(NTREES_DATA, "n_trees", "Number of trees T")

        st.markdown("**Quantization bit-width p** (Decision Tree, D = 4)")
        _sweep_charts(BITS_DATA, "bits", "Bit-width p")

    # ---------------- Part C ----------------
    with tab_c:
        st.subheader("Client-server deployment (WDBC, Decision Tree)")
        st.caption(
            "Mean over the warm per-query measurements (key cached on the "
            "server). The server-side call is not a pure FHE computation "
            "time: see Chapter 5 for the comparison with in-process execution."
        )

        dep = pd.DataFrame(
            {"Operation": list(DEPLOYMENT.keys()),
             "Mean time (s)": list(DEPLOYMENT.values())}
        )
        fig_dep = px.bar(
            dep, x="Mean time (s)", y="Operation", orientation="h",
            text=dep["Mean time (s)"].round(4),
        )
        st.plotly_chart(fig_dep, use_container_width=True)
        st.metric(
            "Total end-to-end (sum of the four components)",
            f"{sum(DEPLOYMENT.values()):.3f} s",
        )

        st.subheader("FHE + differential privacy (illustrative simulation)")
        st.caption(
            "Output perturbation simulation with an ASSUMED sensitivity "
            "(Δf = 1); not a proven (ε, δ)-DP guarantee."
        )
        dp = DP_DATA.copy()
        dp["config"] = dp["dataset"] + " / " + dp["model"]
        fig_dp = px.line(
            dp, x="epsilon", y="acc_mean", color="config", markers=True,
            error_y="acc_std", log_x=True,
            labels={"epsilon": "ε", "acc_mean": "Accuracy under noise"},
        )
        st.plotly_chart(fig_dp, use_container_width=True)
        st.dataframe(
            dp.drop(columns=["config"]).round(4),
            use_container_width=True,
            hide_index=True,
        )
