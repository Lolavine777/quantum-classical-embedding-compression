"""
Ablation Study Runner: Comprehensive Comparative Study (V2)
============================================================

Runs all compression methods across multiple output dimensions
and generates a unified results table + visualizations.

Methods:
  1. PCA (classical baseline)
  2. Autoencoder (Joint E2E)
  3. PQC (CNOT Ring entanglement)
  4. PQC No-Entanglement (ablation variant)
  5. QiC Cascaded (paper-faithful quantum-inspired)
  6. Fidelity Classifier (metric ablation on PCA embeddings)
"""
import os
import sys
import json
import numpy as np
import torch

import config
from src.utils import set_seed, ensure_dir, get_environment_info, Timer, evaluate_classifier
from src.extract_embeddings import extract_and_cache_embeddings, load_cached_embeddings
from src.pca_compressor import PCACompressor
from src.autoencoder import AutoencoderCompressor
from src.pqc_compressor import PQCCompressor
from src.pqc_no_entanglement import PQCNoEntanglementCompressor
from src.qi_compressor import QiCompressor
from src.fidelity_classifier import FidelityClassifier
from src.classifier import train_and_evaluate_classifier
from src.visualize import (
    plot_tsne, plot_training_curves, plot_comparison_bars, plot_param_efficiency,
)


# ---------------------------------------------------------------------------
# Individual experiment runners
# ---------------------------------------------------------------------------

def run_pca(d_out, train_np, val_np, test_np, train_y, val_y, test_y):
    """Run PCA + linear classifier experiment."""
    print(f"\n{'='*60}\nPCA — d'={d_out}\n{'='*60}")
    comp = PCACompressor(d_out=d_out)
    with Timer() as t:
        comp.fit(train_np)
        tr_c = comp.transform(train_np)
        va_c = comp.transform(val_np)
        te_c = comp.transform(test_np)
        res = train_and_evaluate_classifier(tr_c, train_y, va_c, val_y, te_c, test_y, d_in=d_out)
    res.update(method="PCA", d_out=d_out, projection_params=0, time_seconds=round(t.elapsed, 2))
    res["training_log"] = None
    res["compressed_test"] = te_c
    print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  Time={t.elapsed:.1f}s")
    return res


def run_autoencoder(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y):
    """Run Autoencoder joint E2E experiment."""
    print(f"\n{'='*60}\nAutoencoder (E2E) — d'={d_out}\n{'='*60}")
    comp = AutoencoderCompressor(d_out=d_out)
    with Timer() as t:
        comp.fit(train_t, train_yt, val_t, val_yt)
        preds = comp.predict(test_t)
    te_c = comp.transform(test_t)
    res = evaluate_classifier(test_y, preds, config.LABEL_NAMES)
    res.update(method="Autoencoder", d_out=d_out,
               projection_params=comp.get_param_count(), time_seconds=round(t.elapsed, 2))
    res["training_log"] = comp.get_training_log()
    res["compressed_test"] = te_c
    print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  Params={res['projection_params']}  Time={t.elapsed:.1f}s")
    return res


def run_pqc(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y):
    """Run PQC with CNOT ring entanglement."""
    print(f"\n{'='*60}\nPQC (Ring CNOT) — d'={d_out}\n{'='*60}")
    comp = PQCCompressor(d_out=d_out)
    with Timer() as t:
        comp.fit(train_t, train_yt, val_t, val_yt)
        preds = comp.predict(test_t)
    te_c = comp.transform(test_t)
    res = evaluate_classifier(test_y, preds, config.LABEL_NAMES)
    res.update(method="PQC", d_out=d_out,
               projection_params=comp.get_param_count(), time_seconds=round(t.elapsed, 2))
    res["training_log"] = comp.get_training_log()
    res["anomalies"] = comp.get_anomalies()
    res["compressed_test"] = te_c
    print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  Params={res['projection_params']}  Time={t.elapsed:.1f}s")
    return res


def run_pqc_noent(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y):
    """Run PQC without entanglement (ablation)."""
    print(f"\n{'='*60}\nPQC No-Entanglement — d'={d_out}\n{'='*60}")
    comp = PQCNoEntanglementCompressor(d_out=d_out)
    with Timer() as t:
        comp.fit(train_t, train_yt, val_t, val_yt)
        preds = comp.predict(test_t)
    te_c = comp.transform(test_t)
    res = evaluate_classifier(test_y, preds, config.LABEL_NAMES)
    res.update(method="PQC_NoEnt", d_out=d_out,
               projection_params=comp.get_param_count(), time_seconds=round(t.elapsed, 2))
    res["training_log"] = comp.get_training_log()
    res["anomalies"] = comp.get_anomalies()
    res["compressed_test"] = te_c
    print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  Params={res['projection_params']}  Time={t.elapsed:.1f}s")
    return res


def run_qic(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y):
    """Run Quantum-inspired Cascaded Compression (paper method)."""
    print(f"\n{'='*60}\nQiC Cascaded (Paper) — d'={d_out}\n{'='*60}")
    comp = QiCompressor(d_out=d_out)
    with Timer() as t:
        comp.fit(train_t, train_yt, val_t, val_yt)
        preds = comp.predict(test_t)
    te_c = comp.transform(test_t)
    res = evaluate_classifier(test_y, preds, config.LABEL_NAMES)
    res.update(method="QiC", d_out=d_out,
               projection_params=comp.get_param_count(), time_seconds=round(t.elapsed, 2))
    res["training_log"] = comp.get_training_log()
    res["compressed_test"] = te_c
    print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  Params={res['projection_params']}  Time={t.elapsed:.1f}s")
    return res


def run_fidelity_knn(d_out, train_np, test_np, train_y, test_y):
    """Run Fidelity-based prototype classifier on PCA-compressed embeddings."""
    print(f"\n{'='*60}\nFidelity Classifier (on PCA) — d'={d_out}\n{'='*60}")
    # First compress with PCA
    comp = PCACompressor(d_out=d_out)
    comp.fit(train_np)
    tr_c = comp.transform(train_np)
    te_c = comp.transform(test_np)

    clf = FidelityClassifier()
    clf.fit(tr_c, train_y)

    # Fidelity prediction
    preds_fid = clf.predict(te_c)
    res_fid = evaluate_classifier(test_y, preds_fid, config.LABEL_NAMES)
    res_fid.update(method="Fidelity_KNN", d_out=d_out, projection_params=0, time_seconds=0)
    res_fid["training_log"] = None
    res_fid["compressed_test"] = te_c
    print(f"  [Fidelity] Acc={res_fid['accuracy']:.4f}  F1={res_fid['macro_f1']:.4f}")

    # Cosine prediction for comparison
    preds_cos = clf.predict_cosine(te_c)
    res_cos = evaluate_classifier(test_y, preds_cos, config.LABEL_NAMES)
    print(f"  [Cosine]   Acc={res_cos['accuracy']:.4f}  F1={res_cos['macro_f1']:.4f}")

    return res_fid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(config.SEED)
    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.TRAINING_LOGS_DIR)

    plots_dir = os.path.join(config.RESULTS_DIR, "plots")
    ensure_dir(plots_dir)

    env_info = get_environment_info()
    print("Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # Step 1: Ensure embeddings are cached
    print(f"\n{'='*60}\nStep 1: Loading PhoBERT embeddings\n{'='*60}")
    with Timer() as embed_timer:
        extract_and_cache_embeddings()
    print(f"Embedding load time: {embed_timer.elapsed:.2f}s")

    train_t, train_yt = load_cached_embeddings("train")
    val_t, val_yt = load_cached_embeddings("validation")
    test_t, test_yt = load_cached_embeddings("test")

    train_np = train_t.numpy()
    val_np = val_t.numpy()
    test_np = test_t.numpy()
    train_y = train_yt.numpy()
    val_y = val_yt.numpy()
    test_y = test_yt.numpy()

    print(f"Dataset sizes: train={len(train_t)}, val={len(val_t)}, test={len(test_t)}")
    print(f"Train labels: {torch.bincount(train_yt).tolist()}")

    # Collect all results
    all_results = []
    anomalies = []

    # Determine which dimensions to test
    # For PQC methods (slow), only run d'=8 to save time.
    # For fast methods (PCA, AE, QiC, Fidelity), run all D_OUT_LIST.
    d_out_list = config.D_OUT_LIST
    pqc_d_out = config.D_OUT  # Only run PQC at default dimension

    # --- Fast methods: run across all dimensions ---
    for d_out in d_out_list:
        print(f"\n{'#'*60}")
        print(f"# DIMENSION d' = {d_out}")
        print(f"{'#'*60}")

        # PCA
        try:
            r = run_pca(d_out, train_np, val_np, test_np, train_y, val_y, test_y)
            all_results.append(r)
        except Exception as e:
            anomalies.append(f"PCA d'={d_out}: {e}")
            print(f"  ERROR: {e}")

        # Autoencoder
        try:
            r = run_autoencoder(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y)
            all_results.append(r)
        except Exception as e:
            anomalies.append(f"AE d'={d_out}: {e}")
            print(f"  ERROR: {e}")

        # QiC Cascaded
        try:
            r = run_qic(d_out, train_t, val_t, test_t, train_yt, val_yt, test_y)
            all_results.append(r)
        except Exception as e:
            anomalies.append(f"QiC d'={d_out}: {e}")
            print(f"  ERROR: {e}")

        # Fidelity Classifier
        try:
            r = run_fidelity_knn(d_out, train_np, test_np, train_y, test_y)
            all_results.append(r)
        except Exception as e:
            anomalies.append(f"Fidelity d'={d_out}: {e}")
            print(f"  ERROR: {e}")

    # --- Slow PQC methods: run only at default dimension ---
    # PQC with entanglement
    try:
        r = run_pqc(pqc_d_out, train_t, val_t, test_t, train_yt, val_yt, test_y)
        all_results.append(r)
        if r.get("anomalies"):
            anomalies.extend(r["anomalies"])
    except Exception as e:
        anomalies.append(f"PQC d'={pqc_d_out}: {e}")
        print(f"  ERROR: {e}")

    # PQC without entanglement
    try:
        r = run_pqc_noent(pqc_d_out, train_t, val_t, test_t, train_yt, val_yt, test_y)
        all_results.append(r)
        if r.get("anomalies"):
            anomalies.extend(r["anomalies"])
    except Exception as e:
        anomalies.append(f"PQC_NoEnt d'={pqc_d_out}: {e}")
        print(f"  ERROR: {e}")

    # ------------------------------------------------------------------
    # Step 2: Save results
    # ------------------------------------------------------------------
    results_table = []
    for r in all_results:
        entry = {
            "method": r["method"],
            "d_out": r["d_out"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "projection_params": r.get("projection_params", 0),
            "time_seconds": r.get("time_seconds", 0),
            "classification_report": r.get("classification_report", {}),
        }
        results_table.append(entry)

    output = {
        "environment": env_info,
        "embedding_extraction_time_seconds": round(embed_timer.elapsed, 2),
        "results": results_table,
        "anomalies": anomalies,
    }

    metrics_path = os.path.join(config.RESULTS_DIR, "ablation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {metrics_path}")

    # Save training logs
    for r in all_results:
        if r.get("training_log"):
            log_name = f"{r['method'].lower()}_d{r['d_out']}.json"
            log_path = os.path.join(config.TRAINING_LOGS_DIR, log_name)
            with open(log_path, "w") as f:
                json.dump(r["training_log"], f, indent=2)

    # ------------------------------------------------------------------
    # Step 3: Visualizations
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nGenerating Visualizations\n{'='*60}")

    # t-SNE for each method at d'=8
    for r in all_results:
        if r["d_out"] == config.D_OUT and "compressed_test" in r:
            try:
                plot_tsne(r["compressed_test"], test_y, r["method"], r["d_out"], plots_dir)
            except Exception as e:
                print(f"    t-SNE error for {r['method']}: {e}")

    # Training curves at d'=8
    logs_d8 = {}
    for r in all_results:
        if r["d_out"] == config.D_OUT and r.get("training_log"):
            logs_d8[r["method"]] = r["training_log"]
    if logs_d8:
        plot_training_curves(logs_d8, plots_dir, config.D_OUT)

    # Comparison bars
    try:
        plot_comparison_bars(results_table, plots_dir)
    except Exception as e:
        print(f"    Bar chart error: {e}")

    # Parameter efficiency
    try:
        plot_param_efficiency(results_table, plots_dir)
    except Exception as e:
        print(f"    Param efficiency error: {e}")

    # ------------------------------------------------------------------
    # Step 4: Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FULL ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<18} {'d_out':>5} {'Accuracy':>10} {'Macro-F1':>10} {'Params':>10} {'Time(s)':>10}")
    print("-" * 70)
    for r in sorted(results_table, key=lambda x: (x["d_out"], x["method"])):
        print(f"{r['method']:<18} {r['d_out']:>5} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['projection_params']:>10} {r['time_seconds']:>10.1f}")

    if anomalies:
        print(f"\n[WARNING] {len(anomalies)} anomalies:")
        for a in anomalies:
            print(f"  - {a}")

    print("\nDone.")


if __name__ == "__main__":
    main()
