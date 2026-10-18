"""
FHE server process (ciphertexts only), multi-model version for the
Streamlit demonstrator.

- Models are packaged by prepare_model.py in fhe_deployment_wdbc_<key>/
  next to this file (key in {dt, rf, xgb}).
- Each model is loaded lazily on its first request and then kept in memory.
  The loading time is NOT included in the reported server-side call time.
- The request carries "model_key". The evaluation key is read from the
  request ("evaluation_keys") or, if absent, from the cache filled by
  /register_keys (keyed by "client_id", or by "model_key" by default).
"""
import base64
import os
import time

from flask import Flask, jsonify, request
from concrete.ml.deployment import FHEModelServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("FHE_SERVER_PORT", "5000"))
VALID_KEYS = {"dt", "rf", "xgb"}


def deploy_dir_for(model_key):
    return os.path.join(BASE_DIR, f"fhe_deployment_wdbc_{model_key}")


def proc_status_mb(field):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return None


app = Flask(__name__)
SERVERS = {}      # model_key -> loaded FHEModelServer
KEY_CACHE = {}    # client_id or model_key -> serialized evaluation keys


def get_server(model_key):
    if model_key not in SERVERS:
        s = FHEModelServer(path_dir=deploy_dir_for(model_key))
        s.load()
        SERVERS[model_key] = s
    return SERVERS[model_key]


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": sorted(SERVERS.keys()),
        "rss_mb": proc_status_mb("VmRSS"),
        "peak_rss_mb": proc_status_mb("VmHWM"),
    })


@app.route("/register_keys", methods=["POST"])
def register_keys():
    t0 = time.time()
    p = request.get_json()
    cid = p.get("client_id") or p.get("model_key")
    KEY_CACHE[cid] = base64.b64decode(p["evaluation_keys"])
    return jsonify({
        "stored_bytes": len(KEY_CACHE[cid]),
        "handler_time_s": time.time() - t0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    t_handler = time.time()
    p = request.get_json()

    model_key = p.get("model_key")
    if model_key not in VALID_KEYS:
        return jsonify({"error": f"unknown model_key: {model_key!r}"}), 400

    enc_in = base64.b64decode(p["encrypted_input"])
    if "evaluation_keys" in p:                      # keys sent with the request
        keys = base64.b64decode(p["evaluation_keys"])
    else:                                           # keys registered earlier
        cid = p.get("client_id") or model_key
        if cid not in KEY_CACHE:
            return jsonify({"error": "evaluation keys not registered"}), 400
        keys = KEY_CACHE[cid]

    server = get_server(model_key)                  # loading is not timed

    t0 = time.time()
    enc_out = server.run(enc_in, keys)
    inference_s = time.time() - t0

    return jsonify({
        "encrypted_result": base64.b64encode(enc_out).decode(),
        "server_inference_time_s": inference_s,
        "server_handler_time_s": time.time() - t_handler,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT)
