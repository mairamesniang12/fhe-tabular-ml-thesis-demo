import base64, os, time
from flask import Flask, jsonify, request
from concrete.ml.deployment import FHEModelServer

DEPLOY_DIR = os.environ.get(
    "DEPLOY_DIR",
    "/kaggle/working/fhe_deployment_wdbc_dt_v2"
)
PORT = int(os.environ.get("FHE_SERVER_PORT", "5000"))

def proc_status_mb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) / 1024  # kB -> MB
    return None

app = Flask(__name__)
server = FHEModelServer(path_dir=DEPLOY_DIR)
server.load()
KEY_CACHE = {}

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "rss_mb": proc_status_mb("VmRSS"),
        "peak_rss_mb": proc_status_mb("VmHWM")
    })

@app.route("/register_keys", methods=["POST"])
def register_keys():
    t0 = time.time()
    p = request.get_json()
    KEY_CACHE[p["client_id"]] = base64.b64decode(p["evaluation_keys"])
    return jsonify({
        "stored_bytes": len(KEY_CACHE[p["client_id"]]),
        "handler_time_s": time.time() - t0
    })

@app.route("/predict", methods=["POST"])
def predict():
    t_handler = time.time()
    p = request.get_json()
    enc_in = base64.b64decode(p["encrypted_input"])

    # "Legacy" mode: keys are included in each request;
    # normal mode: keys are retrieved from the cache.
    keys = (
        base64.b64decode(p["evaluation_keys"])
        if "evaluation_keys" in p
        else KEY_CACHE[p["client_id"]]
    )

    t0 = time.time()
    enc_out = server.run(enc_in, keys)
    inference_s = time.time() - t0

    return jsonify({
        "encrypted_result": base64.b64encode(enc_out).decode(),
        "server_inference_time_s": inference_s,
        "server_handler_time_s": time.time() - t_handler
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT)
