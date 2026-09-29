"""
Minimal FHE client.
"""
import base64
import time

import numpy as np
import requests

from concrete.ml.deployment import FHEModelClient

DEPLOY_DIR = "/kaggle/working/fhe_deployment_wdbc_dt"  # must match the server
SERVER_URL = "http://localhost:5000/predict"

client = FHEModelClient(path_dir=DEPLOY_DIR, key_dir=DEPLOY_DIR + "/keys")
client.generate_private_and_evaluation_keys()
serialized_eval_keys = client.get_serialized_evaluation_keys()


def predict_one(x_sample: np.ndarray) -> dict:
    t0 = time.time()
    encrypted_input = client.quantize_encrypt_serialize(x_sample)
    encryption_time = time.time() - t0

    payload = {
        "encrypted_input": base64.b64encode(encrypted_input).decode("utf-8"),
        "evaluation_keys": base64.b64encode(serialized_eval_keys).decode("utf-8"),
    }

    t0 = time.time()
    response = requests.post(SERVER_URL, json=payload)
    transfer_time = time.time() - t0
    response_json = response.json()

    encrypted_result = base64.b64decode(response_json["encrypted_result"])

    t0 = time.time()
    result = client.deserialize_decrypt_dequantize(encrypted_result)
    decryption_time = time.time() - t0

    return {
        "prediction": result,
        "client_encryption_time_s": encryption_time,
        "request_size_bytes": len(encrypted_input),
        "response_size_bytes": len(encrypted_result),
        "network_roundtrip_s": transfer_time,
        "server_inference_time_s": response_json["server_inference_time_s"],
        "client_decryption_time_s": decryption_time,
        "total_end_to_end_s": (
            encryption_time + transfer_time + decryption_time
        ),
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    wdbc = load_breast_cancer()
    _, X_test, _, _ = train_test_split(
        wdbc.data, wdbc.target, test_size=0.2, random_state=42, stratify=wdbc.target)

    metrics = predict_one(X_test[:1])
    print("End-to-end client-server FHE inference:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
