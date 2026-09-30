"""
FHE inference server -- serves all 3 real deployed models (DT/RF/XGB),
routed by a "model_key" field in the request body.

Listens on 127.0.0.1 only -- never reachable from outside this Space's
container, only from streamlit_app.py (the client) running in the same
container over localhost.

This process only ever receives ciphertexts and the public evaluation
key for whichever model is requested. It never has the private key or
plaintext features for any model.
"""
import base64
import json
import os
import time

from flask import Flask, jsonify, request

from concrete.ml.deployment import FHEModelServer
from prepare_model import METRICS_FILE

SERVER_PORT = int(os.environ.get("FHE_SERVER_PORT", "5000"))

app = Flask(__name__)

with open(METRICS_FILE) as f:
    _metrics = json.load(f)

servers = {}
for display_name, m in _metrics.items():
    srv = FHEModelServer(path_dir=m["deploy_dir"])
    srv.load()
    servers[m["model_key"]] = srv

print(
    f"[fhe_server] Ready -- {len(servers)} model(s) loaded "
    f"({list(servers.keys())}) -- listening on internal port {SERVER_PORT}"
)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    model_key = payload["model_key"]
    encrypted_input = base64.b64decode(payload["encrypted_input"])
    serialized_eval_keys = base64.b64decode(payload["evaluation_keys"])

    server = servers[model_key]

    t0 = time.time()
    encrypted_result = server.run(encrypted_input, serialized_eval_keys)
    inference_time = time.time() - t0

    return jsonify(
        {
            "encrypted_result": base64.b64encode(encrypted_result).decode("utf-8"),
            "server_inference_time_s": inference_time,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": list(servers.keys())})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=SERVER_PORT)
