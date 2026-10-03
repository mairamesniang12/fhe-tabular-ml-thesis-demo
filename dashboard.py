"""
Dashboard tab for the Streamlit app.

Every number below is copied verbatim from the thesis's real
experimental results (master_results_latest.csv, the depth/n_trees/
bits sensitivity sweeps, dp_simulation_results.csv, the two-process
client-server logs, and neural_preprocessing_results.csv) -- the same
data already written into Chapter 5. This module only visualizes
already-measured results; it does not run any new experiment or
measurement itself.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---- Part A: Comparative Benchmarking (5 datasets x 3 models) ----
PART_A = pd.DataFrame([
    {"dataset": "Spambase", "model": "DT", "acc_plain": 0.8990, "acc_fhe_sim": 0.8708, "latency": 1.627},
    {"dataset": "WDBC", "model": "DT", "acc_plain": 0.9211, "acc_fhe_sim": 0.9123, "latency": 1.424},
    {"dataset": "Adult", "model": "DT", "acc_plain": 0.8559, "acc_fhe_sim": 0.8454, "latency": 1.562},
    {"dataset": "Pima", "model": "DT", "acc_plain": 0.7922, "acc_fhe_sim": 0.7532, "latency": 1.542},
    {"dataset": "Heart", "model": "DT", "acc_plain": 0.7000, "acc_fhe_sim": 0.7667, "latency": 1.543},
    {"dataset": "Spambase", "model": "RF", "acc_plain": 0.9131, "acc_fhe_sim": 0.8740, "latency": 15.894},
    {"dataset": "WDBC", "model": "RF", "acc_plain": 0.9561, "acc_fhe_sim": 0.9561, "latency": 13.351},
    {"dataset": "Adult", "model": "RF", "acc_plain": 0.8540, "acc_fhe_sim": 0.8405, "latency": 17.532},
    {"dataset": "Pima", "model": "RF", "acc_plain": 0.7403, "acc_fhe_sim": 0.7273, "latency": 16.419},
    {"dataset": "Heart", "model": "RF", "acc_plain": 0.8167, "acc_fhe_sim": 0.8167, "latency": 15.470},
    {"dataset": "Spambase", "model": "XGB", "acc_plain": 0.9490, "acc_fhe_sim": 0.9207, "latency": 14.806},
    {"dataset": "WDBC", "model": "XGB", "acc_plain": 0.9561, "acc_fhe_sim": 0.9474, "latency": 11.120},
    {"dataset": "Adult", "model": "XGB", "acc_plain": 0.8777, "acc_fhe_sim": 0.8569, "latency": 17.453},
    {"dataset": "Pima", "model": "XGB", "acc_plain": 0.7662, "acc_fhe_sim": 0.7792, "latency": 13.577},
    {"dataset": "Heart", "model": "XGB", "acc_plain": 0.8500, "acc_fhe_sim": 0.8333, "latency": 15.053},
])

# ---- Part B: Sensitivity sweeps (WDBC vs Pima) ----
DEPTH_DATA = pd.DataFrame([
    {"dataset": "WDBC", "depth": 3, "acc_fhe_sim": 0.9386, "latency": 0.915},
    {"dataset": "WDBC", "depth": 5, "acc_fhe_sim": 0.9474, "latency": 1.594},
    {"dataset": "WDBC", "depth": 7, "acc_fhe_sim": 0.9298, "latency": 1.929},
    {"dataset": "WDBC", "depth": 10, "acc_fhe_sim": 0.9123, "latency": 3.934},
    {"dataset": "Pima", "depth": 3, "acc_fhe_sim": 0.7468, "latency": 0.907},
    {"dataset": "Pima", "depth": 5, "acc_fhe_sim": 0.7078, "latency": 2.299},
    {"dataset": "Pima", "depth": 7, "acc_fhe_sim": 0.6623, "latency": 5.039},
    {"dataset": "Pima", "depth": 10, "acc_fhe_sim": 0.7013, "latency": 12.964},
])

NTREES_DATA = pd.DataFrame([
    {"dataset": "WDBC", "n_trees": 10, "acc_fhe_sim": 0.9561, "latency": 9.231},
    {"dataset": "WDBC", "n_trees": 50, "acc_fhe_sim": 0.9561, "latency": 43.604},
    {"dataset": "WDBC", "n_trees": 100, "acc_fhe_sim": 0.9561, "latency": 88.938},
    {"dataset": "Pima", "n_trees": 10, "acc_fhe_sim": 0.7078, "latency": 11.403},
    {"dataset": "Pima", "n_trees": 50, "acc_fhe_sim": 0.7208, "latency": 53.687},
    {"dataset": "Pima", "n_trees": 100, "acc_fhe_sim": 0.7403, "latency": 106.737},
])

BITS_DATA = pd.DataFrame([
    {"dataset": "WDBC", "bits": 2, "acc_fhe_sim": 0.9211, "latency": 0.569},
    {"dataset": "WDBC", "bits": 4, "acc_fhe_sim": 0.9211, "latency": 1.226},
    {"dataset": "WDBC", "bits": 6, "acc_fhe_sim": 0.9386, "latency": 1.636},
    {"dataset": "WDBC", "bits": 8, "acc_fhe_sim": 0.9211, "latency": 2.108},
    {"dataset": "Pima", "bits": 2, "acc_fhe_sim": 0.7468, "latency": 0.713},
    {"dataset": "Pima", "bits": 4, "acc_fhe_sim": 0.7727, "latency": 1.898},
    {"dataset": "Pima", "bits": 6, "acc_fhe_sim": 0.7078, "latency": 2.411},
    {"dataset": "Pima", "bits": 8, "acc_fhe_sim": 0.7987, "latency": 2.610},
])

# ---- Part C: Deployment, DP, H1 ----
DEPLOYMENT = {
    "Client encryption": 0.0128,
    "Network transfer overhead": 2.3779,
    "Server FHE inference": 5.1194,
    "Client decryption": 0.0037,
}

DP_DATA = pd.DataFrame([
    {"dataset": "WDBC", "model": "DT", "epsilon": 0.1, "acc_mean": 0.4671, "acc_std": 0.0429},
    {"dataset": "WDBC", "model": "DT", "epsilon": 1.0, "acc_mean": 0.5070, "acc_std": 0.0441},
    {"dataset": "WDBC", "model": "DT", "epsilon": 10.0, "acc_mean": 0.8539, "acc_std": 0.0247},
    {"dataset": "WDBC", "model": "RF", "epsilon": 0.1, "acc_mean": 0.4667, "acc_std": 0.0428},
    {"dataset": "WDBC", "model": "RF", "epsilon": 1.0, "acc_mean": 0.5048, "acc_std": 0.0453},
    {"dataset": "WDBC", "model": "RF", "epsilon": 10.0, "acc_mean": 0.8654, "acc_std": 0.0285},
    {"dataset": "Pima", "model": "DT", "epsilon": 0.1, "acc_mean": 0.5347, "acc_std": 0.0413},
    {"dataset": "Pima", "model": "DT", "epsilon": 1.0, "acc_mean": 0.5455, "acc_std": 0.0424},
    {"dataset": "Pima", "model": "DT", "epsilon": 10.0, "acc_mean": 0.6880, "acc_std": 0.0252},
    {"dataset": "Pima", "model": "RF", "epsilon": 0.1, "acc_mean": 0.5344, "acc_std": 0.0424},
    {"dataset": "Pima", "model": "RF", "epsilon": 1.0, "acc_mean": 0.5412, "acc_std": 0.0430},
    {"dataset": "Pima", "model": "RF", "epsilon": 10.0, "acc_mean": 0.6425, "acc_std": 0.0311},
])

H1_DATA = pd.DataFrame([
    {"dataset": "WDBC", "model": "DT", "latency": 1.424},
    {"dataset": "WDBC", "model": "RF", "latency": 13.351},
    {"dataset": "WDBC", "model": "XGB", "latency": 11.120},
    {"dataset": "WDBC", "model": "MLP", "latency": 19.630},
    {"dataset": "Pima", "model": "DT", "latency": 1.542},
    {"dataset": "Pima", "model": "RF", "latency": 16.419},
    {"dataset": "Pima", "model": "XGB", "latency": 13.577},
    {"dataset": "Pima", "model": "MLP", "latency": 4.412},
])


def render_dashboard():
    st.markdown(
        "Tous les chiffres ci-dessous viennent directement des résultats réels "
        "du Chapitre 5 (`master_results_latest.csv` et les CSV de sensibilité) "
        "-- ce dashboard ne fait que les visualiser, il ne relance aucune mesure."
    )

    tab_a, tab_b, tab_c = st.tabs([
        "📊 Partie A — Benchmarking",
        "📈 Partie B — Sensibilité",
        "🔗 Partie C — Déploiement & Hybride",
    ])

    # ---------------- PART A ----------------
    with tab_a:
        st.subheader("Accuracy : plaintext vs FHE-simulate")
        fig = px.bar(
            PART_A.melt(
                id_vars=["dataset", "model"],
                value_vars=["acc_plain", "acc_fhe_sim"],
                var_name="type", value_name="accuracy",
            ),
            x="dataset", y="accuracy", color="type", barmode="group",
            facet_col="model",
            labels={"accuracy": "Accuracy", "dataset": "Dataset", "type": ""},
            color_discrete_map={"acc_plain": "#6b7280", "acc_fhe_sim": "#0b3d91"},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latence FHE réelle (n=30, mean ± std sur 3 runs)")
        fig2 = px.bar(
            PART_A, x="dataset", y="latency", color="model", barmode="group",
            labels={"latency": "Latence (s/échantillon)", "dataset": "Dataset"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Config : $D=4$, $T=15$, $p=5$ bits. Latence mesurée sur un "
            "sous-échantillon stratifié de 30 observations (seed=42)."
        )

    # ---------------- PART B ----------------
    with tab_b:
        colors = {"WDBC": "#1f77b4", "Pima": "#ff7f0e"}

        st.subheader("Effet de la profondeur de l'arbre")
        c1, c2 = st.columns(2)
        with c1:
            fig_d1 = px.line(
                DEPTH_DATA, x="depth", y="latency", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"latency": "Latence FHE (s)", "depth": "Profondeur D"},
            )
            st.plotly_chart(fig_d1, use_container_width=True)
        with c2:
            fig_d2 = px.line(
                DEPTH_DATA, x="depth", y="acc_fhe_sim", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"acc_fhe_sim": "Accuracy (FHE-sim)", "depth": "Profondeur D"},
            )
            st.plotly_chart(fig_d2, use_container_width=True)
        st.caption(
            "WDBC : accuracy maximale à D=5. Pima : accuracy maximale à D=3 -- "
            "une config optimale pour un dataset ne l'est pas forcément pour l'autre."
        )

        st.subheader("Effet du nombre d'arbres (Random Forest)")
        c3, c4 = st.columns(2)
        with c3:
            fig_n1 = px.line(
                NTREES_DATA, x="n_trees", y="latency", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"latency": "Latence FHE (s)", "n_trees": "Nombre d'arbres T"},
            )
            st.plotly_chart(fig_n1, use_container_width=True)
        with c4:
            fig_n2 = px.line(
                NTREES_DATA, x="n_trees", y="acc_fhe_sim", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"acc_fhe_sim": "Accuracy (FHE-sim)", "n_trees": "Nombre d'arbres T"},
            )
            st.plotly_chart(fig_n2, use_container_width=True)
        st.caption(
            "WDBC : accuracy constante quel que soit T. Pima : accuracy "
            "augmente avec T (+3.25 points entre T=10 et T=100)."
        )

        st.subheader("Effet de la largeur de quantification (bits)")
        c5, c6 = st.columns(2)
        with c5:
            fig_b1 = px.line(
                BITS_DATA, x="bits", y="latency", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"latency": "Latence FHE (s)", "bits": "Bits de quantification p"},
            )
            st.plotly_chart(fig_b1, use_container_width=True)
        with c6:
            fig_b2 = px.line(
                BITS_DATA, x="bits", y="acc_fhe_sim", color="dataset",
                markers=True, color_discrete_map=colors,
                labels={"acc_fhe_sim": "Accuracy (FHE-sim)", "bits": "Bits de quantification p"},
            )
            st.plotly_chart(fig_b2, use_container_width=True)
        st.caption(
            "Pima : p=6 bits est la PIRE config testée (0.7078), p=8 la "
            "meilleure (0.7987) -- pas de relation monotone sur ce dataset."
        )

    # ---------------- PART C ----------------
    with tab_c:
        st.subheader("Déploiement client-serveur : répartition de la latence")
        fig_dep = go.Figure(go.Pie(
            labels=list(DEPLOYMENT.keys()),
            values=list(DEPLOYMENT.values()),
            hole=0.4,
        ))
        fig_dep.update_layout(
            annotations=[dict(text=f"{sum(DEPLOYMENT.values()):.2f}s<br>total", x=0.5, y=0.5, showarrow=False)]
        )
        st.plotly_chart(fig_dep, use_container_width=True)
        st.caption(
            "Mesuré entre deux VRAIS processus séparés (fhe_client.py / "
            "fhe_server.py) communiquant par HTTP réel. Basé sur une seule "
            "requête -- à répéter 5-10x pour un intervalle de confiance."
        )

        st.subheader("FHE + Differential Privacy : compromis vie privée / utilité")
        fig_dp = px.bar(
            DP_DATA, x="epsilon", y="acc_mean", color="model",
            error_y="acc_std", barmode="group", facet_col="dataset",
            labels={"acc_mean": "Accuracy (FHE+DP)", "epsilon": "ε (budget de confidentialité)"},
        )
        fig_dp.update_xaxes(type="category")
        st.plotly_chart(fig_dp, use_container_width=True)
        st.caption(
            "Simulation Monte Carlo illustrative (20 répétitions, δ=1e-5, "
            "Δf=1 -- hypothèse simplificatrice, PAS une garantie DP formelle)."
        )

        st.subheader("H1 : arbres vs réseau de neurones sous FHE réelle")
        fig_h1 = px.bar(
            H1_DATA, x="dataset", y="latency", color="model", barmode="group",
            labels={"latency": "Latence FHE (s/échantillon)"},
        )
        st.plotly_chart(fig_h1, use_container_width=True)
        st.caption(
            "Sur WDBC, le MLP est plus lent que les 3 modèles à arbres. Sur "
            "Pima, le MLP est plus lent que DT mais PLUS RAPIDE que RF et "
            "XGB -- H1 n'est que partiellement vérifiée."
        )
