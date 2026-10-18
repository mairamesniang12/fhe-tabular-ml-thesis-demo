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

NTREES_DATA = _csv("sweep_ntrees.csv").rename(
    columns={
        "value": "n_trees",
        "acc_fhe_simulate_full": "acc_fhe_sim",
        "latency_fhe_mean_s": "latency",
    }
)


# ---- Part C: Deployment, differential privacy, and H1 ----
DP_DATA = _csv("dp.csv")
DP_DATA = DP_DATA.rename(
    columns={
        "acc_dp_mean": "acc_mean",
        "acc_dp_std": "acc_std",
    }
)

_q = _csv("deploy_queries.csv")
DEPLOYMENT = {
    "Client encryption": _q.encryption_s.mean(),
    "HTTP/request-processing overhead": _q.overhead_non_fhe_s.mean(),
    "Server-side call (FHEModelServer.run)": _q.server_inference_s.mean(),
    "Client decryption": _q.decryption_s.mean(),
}
