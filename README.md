# FHE Client-Server Demonstrator (Concrete-ML)

A live, browser-based demonstration of encrypted machine learning
inference using **Fully Homomorphic Encryption (FHE)**, built with
[Concrete-ML](https://github.com/zama-ai/concrete-ml) (Zama) and
deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

Part of the MSc thesis *"Efficient Privacy-Preserving ML Using
Tree-Based and Hybrid Models Under FHE"* - Mairame Samba NIANG,
supervised by Dr. Célestin Wafo Soh, AIMS Sénégal, 2025–2026.

## What this demonstrates

This app trains and compiles three tree-based models (Decision Tree,
Random Forest, XGBoost) on the WDBC (Wisconsin Diagnostic Breast
Cancer) dataset, then lets you run a **real, end-to-end encrypted
inference query** through a genuine client-server architecture:

```
[Browser] <--> [Streamlit app: the CLIENT]
                  - holds the private key
                  - quantizes, encrypts, and decrypts
                       |
                       | HTTP (ciphertext only)
                       v
              [fhe_server.py: a separate OS process]
                  - never has the private key
                  - never sees plaintext features
                  - only ever computes on ciphertexts
```

The server process is architecturally incapable of seeing plaintext
patient data or the private key — it only ever receives and returns
encrypted bytes over a real local HTTP call.

Every number shown in the UI is either:
- **measured live** during the exact click you make (encryption time,
  network round trip, server inference time, decryption time,
  ciphertext sizes), or
- **computed once at startup** from real training/compilation/
  evaluation runs (`prepare_model.py`), written to `real_metrics.json`.

Nothing is hardcoded or hand-typed.

## ⚠️ Scope and limitations

This is a **demonstrator**, not the full experimental benchmark of the
thesis. In particular:

- Only the WDBC dataset is used here (the thesis evaluates 5 datasets:
  Spambase, WDBC, Adult, Pima Diabetes, Heart Disease).
- The "real FHE latency" shown in the results table is measured on a
  small sample (15 observations, 2 repeats) at deployment time, to fit
  within free-tier compute limits — **not** the thesis's own ≥30-sample,
  95%-CI protocol used in Chapter 5.
- Model hyperparameters (tree depth, number of estimators) are kept
  small for the same reason.

Treat this app as a live illustration of the client-server trust
boundary and the encryption/decryption workflow — refer to the thesis
itself for the actual experimental results and their statistical
validation.

## Project structure

```
.
├── streamlit_app.py    # UI + client-side logic (encryption/decryption)
├── prepare_model.py     # Trains, evaluates and compiles the 3 FHE models
├── fhe_server.py         # Server process -- ciphertext-only, internal port
├── requirements.txt
├── runtime.txt           # Pins Python 3.11 for Concrete-ML compatibility
└── LICENSE
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run will train and compile all three models (this can take a
few minutes) and write `real_metrics.json` and the
`fhe_deployment_wdbc_*` directories, which are regenerated on demand
and not committed to this repository.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new
   app pointing at `streamlit_app.py`.
3. In advanced settings, make sure Python 3.11 is selected.
4. Deploy. The first build installs Concrete-ML and XGBoost and runs
   `prepare_model.py` once expect the first load to be slow.

If the app hits the platform's free-tier resource limits, reduce the
`MODEL_CONFIGS` dictionary in `prepare_model.py` to fewer models (e.g.
Decision Tree only) or lower `n_estimators`.

## Citing Concrete-ML

```bibtex
@Misc{ConcreteML,
  title={Concrete {ML}: a Privacy-Preserving Machine Learning Library
         using Fully Homomorphic Encryption for Data Scientists},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete-ml}},
}
```

## License

The demonstrator code in this repository is released under the
[MIT License](LICENSE). Concrete-ML itself is licensed separately
under BSD-3-Clause-Clear by Zama (free for research, prototyping and
experimentation) -- see [zama-ai/concrete-ml](https://github.com/zama-ai/concrete-ml)
for details.
