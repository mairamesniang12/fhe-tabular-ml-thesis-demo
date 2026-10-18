import base64, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd, requests
from scipy import stats
from concrete.ml.deployment import FHEModelClient

DEPLOY_DIR = "/kaggle/working/fhe_deployment_wdbc_dt_v2"
PORT, URL = 5000, "http://127.0.0.1:5000"
N_QUERIES, N_LEGACY = 10, 3

# ---------- Server initialization (separate process) ----------
env = dict(os.environ, DEPLOY_DIR=DEPLOY_DIR, FHE_SERVER_PORT=str(PORT))
t0 = time.time()
server_proc = subprocess.Popen(
    [sys.executable, "/kaggle/working/fhe_server_v2.py"], env=env,
    stdout=open("/kaggle/working/server_v2.log", "w"), stderr=subprocess.STDOUT)
ready = False
while time.time() - t0 < 300:
    if server_proc.poll() is not None:
        break
    try:
        if requests.get(URL + "/health", timeout=1).status_code == 200:
            ready = True
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(0.2)
if not ready:
    raise RuntimeError(
        "Server failed to start: check /kaggle/working/server_v2.log"
    )
server_init_s = time.time() - t0

# ---------- Client initialization ----------
key_dir = os.path.join(DEPLOY_DIR, "keys_bench")
shutil.rmtree(key_dir, ignore_errors=True)
t0 = time.time()
client = FHEModelClient(path_dir=DEPLOY_DIR, key_dir=key_dir)
client_load_s = time.time() - t0

t0 = time.time()
client.generate_private_and_evaluation_keys()
keygen_s = time.time() - t0

t0 = time.time()
eval_keys = client.get_serialized_evaluation_keys()
eval_key_serialize_s = time.time() - t0
eval_key_bytes = len(eval_keys)

# One-time evaluation key transfer (Base64 encoding + HTTP + server storage)
t0 = time.time()
r = requests.post(
    URL + "/register_keys",
    timeout=600,
    json={
        "client_id": "bench",
        "evaluation_keys": base64.b64encode(eval_keys).decode()
    }
)
r.raise_for_status()
key_transfer_s = time.time() - t0

X_s = np.load("/kaggle/working/bench_samples.npz")["X"]

def query(x, legacy=False):
    t0 = time.time()
    enc = client.quantize_encrypt_serialize(x)
    enc_s = time.time() - t0

    payload = {
        "client_id": "bench",
        "encrypted_input": base64.b64encode(enc).decode()
    }
    if legacy:
        payload["evaluation_keys"] = base64.b64encode(eval_keys).decode()

    http_payload_bytes = len(json.dumps(payload))  # Excluded from timing
    t0 = time.time()
    resp = requests.post(URL + "/predict", json=payload, timeout=600)
    roundtrip_s = time.time() - t0

    j = resp.json()
    out = base64.b64decode(j["encrypted_result"])

    t0 = time.time()
    client.deserialize_decrypt_dequantize(out)
    dec_s = time.time() - t0

    return {
        "encryption_s": enc_s,
        "request_ciphertext_bytes": len(enc),
        "http_payload_bytes": http_payload_bytes,
        "roundtrip_s": roundtrip_s,
        "server_inference_s": j["server_inference_time_s"],
        "server_handler_s": j["server_handler_time_s"],
        "response_bytes": len(out),
        "decryption_s": dec_s,
        "total_s": enc_s + roundtrip_s + dec_s,
        "overhead_non_fhe_s": roundtrip_s - j["server_inference_time_s"]
    }

query(X_s[0:1])  # Warm-up query (excluded from measurements)

rows = [query(X_s[i:i+1]) for i in range(N_QUERIES)]
legacy = [query(X_s[i:i+1], legacy=True) for i in range(N_LEGACY)]

def summarize(rows):
    df = pd.DataFrame(rows)
    out = {}

    for k in df.columns:
        v = df[k].astype(float).values
        sd = v.std(ddof=1)

        lo, hi = (
            (v.mean(), v.mean()) if sd == 0 else
            stats.t.interval(
                0.95,
                df=len(v) - 1,
                loc=v.mean(),
                scale=stats.sem(v)
            )
        )

        out[k] = {
            "mean": v.mean(),
            "std": sd,
            "ci95_low": lo,
            "ci95_high": hi
        }

    return pd.DataFrame(out).T

health = requests.get(URL + "/health").json()
server_proc.terminate()

init = {
    "server_init_s": server_init_s,
    "client_load_s": client_load_s,
    "keygen_s": keygen_s,
    "eval_key_serialize_s": eval_key_serialize_s,
    "eval_key_bytes": eval_key_bytes,
    "eval_key_transfer_s": key_transfer_s,
    "server_rss_mb_after_queries": health["rss_mb"],
    "server_peak_rss_mb": health["peak_rss_mb"],
    **json.load(open("/kaggle/working/deployment_setup_times.json"))
}

json.dump(
    init,
    open("/kaggle/working/deployment_init_costs.json", "w"),
    indent=2
)

pd.DataFrame(rows).to_csv(
    "/kaggle/working/deployment_queries_cached.csv",
    index=False
)

pd.DataFrame(legacy).to_csv(
    "/kaggle/working/deployment_queries_legacy.csv",
    index=False
)

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

print("=== INITIALIZATION COSTS (one-time) ===")
print(json.dumps(init, indent=2))

print(
    f"\n=== PER-QUERY COSTS, CACHED KEYS "
    f"(n={N_QUERIES}, after warm-up) ==="
)
print(summarize(rows))

print(
    f"\n=== LEGACY MODE, KEYS SENT WITH EVERY QUERY "
    f"(n={N_LEGACY}) ==="
)
print(summarize(legacy))
