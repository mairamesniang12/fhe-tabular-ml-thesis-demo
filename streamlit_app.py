"""
Streamlit entry point.

CRITICAL Streamlit-specific detail: Streamlit re-runs this entire
script top-to-bottom on every widget interaction. Without
@st.cache_resource, clicking any button would re-trigger training +
compiling all 3 FHE models and re-launching fhe_server.py as a new
subprocess every single time. @st.cache_resource makes the setup run
exactly ONCE per session (cached across reruns), which is what makes
this app usable at all.

Everything displayed is either measured live during THIS exact click,
or read from real_metrics.json (written once by prepare_model.py from
real training/compilation/evaluation runs). Nothing is hand-typed.
"""
import base64
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from prepare_model import (
    N_LATENCY_REPEATS_AT_DEPLOY,
    N_REAL_FHE_SAMPLES_AT_DEPLOY,
    prepare_all_models,
)
from concrete.ml.deployment import FHEModelClient

SERVER_PORT = int(os.environ.get("FHE_SERVER_PORT", "5000"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/predict"
CLASS_NAMES = ["malignant", "benign"]


@st.cache_resource(show_spinner="Entraînement + compilation des 3 modèles FHE (une seule fois)...")
def setup():
    """Runs exactly once per Streamlit session (cached across reruns)."""
    real_metrics = prepare_all_models()

    server_process = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "fhe_server.py")]
    )
    time.sleep(5)

    wdbc = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(
        wdbc.data, wdbc.target, test_size=0.2, random_state=42, stratify=wdbc.target
    )
    feature_names = list(wdbc.feature_names)

    clients = {}
    eval_keys = {}
    for display_name, m in real_metrics.items():
        c = FHEModelClient(path_dir=m["deploy_dir"], key_dir=m["deploy_dir"] + "/keys")
        c.generate_private_and_evaluation_keys()
        clients[m["model_key"]] = c
        eval_keys[m["model_key"]] = c.get_serialized_evaluation_keys()

    return {
        "real_metrics": real_metrics,
        "server_process": server_process,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "clients": clients,
        "eval_keys": eval_keys,
    }


STATE = setup()
REAL_METRICS = STATE["real_metrics"]
X_TEST = STATE["X_test"]
Y_TEST = STATE["y_test"]
FEATURE_NAMES = STATE["feature_names"]
CLIENTS = STATE["clients"]
EVAL_KEYS = STATE["eval_keys"]
MODEL_DISPLAY_NAMES = list(REAL_METRICS.keys())


def run_inference(model_display_name, features, true_label):
    m = REAL_METRICS[model_display_name]
    model_key = m["model_key"]
    client = CLIENTS[model_key]
    serialized_eval_keys = EVAL_KEYS[model_key]

    x_sample = np.array(features, dtype=np.float64).reshape(1, -1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    client_log = []
    client_log.append("=" * 50)
    client_log.append("CLIENT SIDE (this Streamlit app)")
    client_log.append("=" * 50)
    client_log.append(f"[{timestamp}] Dataset : WDBC (Breast Cancer)")
    client_log.append(f"Model   : {model_display_name}")
    client_log.append("Security: 128-bit (TFHE, Concrete-ML)")
    client_log.append("")
    client_log.append("[Step 1] Quantizing + encrypting input (real Concrete-ML call)...")

    t0 = time.time()
    encrypted_input = client.quantize_encrypt_serialize(x_sample)
    encryption_time = time.time() - t0
    req_size = len(encrypted_input)

    client_log.append("  ✓ Plaintext → Ciphertext (client.quantize_encrypt_serialize)")
    client_log.append(f"  ✓ Encrypted request size : {req_size:,} bytes (measured)")
    client_log.append(f"  ⏱ Encryption time        : {encryption_time:.4f}s (measured)")
    client_log.append("")
    client_log.append("[Step 2] Sending encrypted request over real HTTP to fhe_server.py...")
    client_log.append("  → Server process receives ONLY the ciphertext + evaluation key")
    client_log.append("  → Server CANNOT see plaintext feature values")

    payload = {
        "model_key": model_key,
        "encrypted_input": base64.b64encode(encrypted_input).decode("utf-8"),
        "evaluation_keys": base64.b64encode(serialized_eval_keys).decode("utf-8"),
    }

    t0 = time.time()
    response = requests.post(SERVER_URL, json=payload, timeout=120)
    network_roundtrip_s = time.time() - t0
    response_json = response.json()
    server_inference_time = response_json["server_inference_time_s"]

    encrypted_result = base64.b64decode(response_json["encrypted_result"])
    resp_size = len(encrypted_result)

    server_log = []
    server_log.append("=" * 50)
    server_log.append("SERVER SIDE (fhe_server.py, separate OS process)")
    server_log.append("=" * 50)
    server_log.append(f"[Step 3] Encrypted request received (model={model_key})")
    server_log.append(f"  ✓ Ciphertext size received : {req_size:,} bytes")
    server_log.append("")
    server_log.append("[Step 4] Running real FHE inference (server.run)...")
    server_log.append("  → Homomorphic evaluation of the compiled circuit")
    server_log.append("  → Server sees ONLY encrypted data at every step")
    server_log.append(f"  ⏱ Server FHE inference time : {server_inference_time:.4f}s (measured)")
    server_log.append("")
    server_log.append("[Step 5] Returning encrypted result...")
    server_log.append(f"  ✓ Encrypted response size : {resp_size:,} bytes (measured)")
    server_log.append("  → Client will decrypt -- server never sees the plaintext result")

    t0 = time.time()
    result = client.deserialize_decrypt_dequantize(encrypted_result)
    decryption_time = time.time() - t0
    prediction = int(np.array(result).reshape(-1)[0])

    client_log.append("")
    client_log.append("[Step 6] Encrypted result received back over HTTP.")
    client_log.append("[Step 7] Decrypting (private key, client only)...")
    client_log.append(f"  ⏱ Decryption time : {decryption_time:.4f}s (measured)")
    client_log.append("")

    total_e2e = encryption_time + network_roundtrip_s + decryption_time

    client_log.append("=" * 50)
    client_log.append("TIMING SUMMARY (this single query, real measurements)")
    client_log.append("=" * 50)
    client_log.append(f"  Encryption        : {encryption_time:.4f}s")
    client_log.append(f"  Network round trip: {network_roundtrip_s:.4f}s "
                       f"(includes server inference)")
    client_log.append(f"    of which server inference: {server_inference_time:.4f}s")
    client_log.append(f"  Decryption         : {decryption_time:.4f}s")
    client_log.append(f"  TOTAL end-to-end   : {total_e2e:.4f}s")
    client_log.append("")
    client_log.append("🔒 Privacy preserved throughout -- the server process")
    client_log.append("   never had access to plaintext features or the private key.")

    return {
        "client_log": "\n".join(client_log),
        "server_log": "\n".join(server_log),
        "prediction": prediction,
        "req_size": req_size,
        "resp_size": resp_size,
        "total_e2e": total_e2e,
        "true_label": true_label,
    }


# ============================== UI ==============================
st.set_page_config(page_title="Démonstrateur FHE Client-Serveur", layout="wide")

st.title("🔒 Démonstrateur FHE Client-Serveur — WDBC (Breast Cancer)")
st.markdown(
    "Chaque nombre affiché est soit mesuré en direct lors de ce clic, soit calculé "
    "une fois au démarrage à partir d'entraînements/compilations Concrete-ML réels "
    "-- rien n'est codé en dur.\n\n"
    f"Le serveur (`fhe_server.py`, processus séparé, port interne {SERVER_PORT}, "
    "jamais exposé publiquement) ne reçoit et ne renvoie **que des ciphertexts**."
)

if "sample_features" not in st.session_state:
    st.session_state.sample_features = None
    st.session_state.true_label = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Configuration")
    model_choice = st.selectbox("Modèle FHE", MODEL_DISPLAY_NAMES)

    if st.button("🔄 Charger un échantillon de test réel"):
        idx = int(np.random.randint(0, len(X_TEST)))
        st.session_state.sample_features = X_TEST[idx].tolist()
        st.session_state.true_label = int(Y_TEST[idx])

    if st.session_state.sample_features is not None:
        st.json(
            {
                name: round(val, 4)
                for name, val in zip(FEATURE_NAMES, st.session_state.sample_features)
            }
        )
        st.caption(
            f"Label réel (connu seulement du client) : "
            f"{CLASS_NAMES[st.session_state.true_label]}"
        )

    run_disabled = st.session_state.sample_features is None
    run_clicked = st.button("🔐 Chiffrer et prédire", disabled=run_disabled, type="primary")

with col2:
    if run_clicked:
        with st.spinner("Chiffrement → réseau → inférence FHE → déchiffrement..."):
            result = run_inference(
                model_choice,
                st.session_state.sample_features,
                st.session_state.true_label,
            )

        st.subheader("💻 CLIENT LOG (mesures réelles)")
        st.code(result["client_log"], language=None)

        st.subheader("☁️ SERVER LOG (mesures réelles)")
        st.code(result["server_log"], language=None)

        st.subheader("📊 Résultats")
        pred_label = CLASS_NAMES[result["prediction"]]
        true_label = CLASS_NAMES[result["true_label"]]
        correct = result["prediction"] == result["true_label"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Prédiction (déchiffrée)", pred_label)
        c2.metric("Label réel", true_label, "✅ correct" if correct else "❌ incorrect")
        c3.metric("Latence bout-en-bout", f"{result['total_e2e']:.3f} s")

        st.caption(
            f"Requête chiffrée : {result['req_size']:,} bytes  |  "
            f"Réponse chiffrée : {result['resp_size']:,} bytes"
        )

        st.markdown("---")
        st.subheader("Métriques globales par modèle")
        st.caption(
            f"Calculées au démarrage : accuracy en clair et FHE-simulate sur "
            f"l'ensemble de test local complet ; latence FHE réelle mesurée sur "
            f"seulement {N_REAL_FHE_SAMPLES_AT_DEPLOY} échantillons × "
            f"{N_LATENCY_REPEATS_AT_DEPLOY} répétitions -- un budget de démarrage "
            f"volontairement réduit, PAS le protocole complet de la thèse (≥30 "
            f"échantillons). Ces chiffres illustrent la démo, ils ne remplacent "
            f"pas les résultats du Chapitre 5."
        )
        rows = []
        for name, mm in REAL_METRICS.items():
            rows.append(
                {
                    "Modèle": ("→ " if name == model_choice else "") + name,
                    "Acc. plain": f"{mm['acc_plain']*100:.1f}%",
                    "Acc. FHE-simulate": f"{mm['acc_fhe_simulate_full']*100:.1f}%",
                    "Latence FHE (petit échantillon)": (
                        f"{mm['latency_fhe_mean_s']:.3f}s ± {mm['latency_fhe_std_s']:.3f}s"
                    ),
                }
            )
        st.table(rows)
    else:
        st.info("Charge un échantillon, puis clique sur \"Chiffrer et prédire\".")
