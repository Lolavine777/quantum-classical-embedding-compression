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
from src.classifier import train_and_evaluate_classifier


def run_pca_experiment(d_out: int, train_emb: np.ndarray, val_emb: np.ndarray,
                       test_emb: np.ndarray, train_labels: np.ndarray,
                       val_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    print(f"\n{'='*60}")
    print(f"PCA - d_out={d_out}")
    print(f"{'='*60}")

    compressor = PCACompressor(d_out=d_out)

    with Timer() as t:
        compressor.fit(train_emb)
        train_compressed = compressor.transform(train_emb)
        val_compressed = compressor.transform(val_emb)
        test_compressed = compressor.transform(test_emb)

        results = train_and_evaluate_classifier(
            train_compressed, train_labels,
            val_compressed, val_labels,
            test_compressed, test_labels,
            d_in=d_out,
        )

    results["method"] = "PCA"
    results["d_out"] = d_out
    results["projection_params"] = compressor.get_param_count()
    results["training_log"] = compressor.get_training_log()
    results["time_seconds"] = round(t.elapsed, 2)

    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro-F1: {results['macro_f1']:.4f}")
    print(f"  Time: {t.elapsed:.2f}s")

    return results


def run_autoencoder_experiment(d_out: int, train_emb_t: torch.Tensor, val_emb_t: torch.Tensor,
                                test_emb_t: torch.Tensor, train_labels_t: torch.Tensor,
                                val_labels_t: torch.Tensor, test_labels: np.ndarray) -> dict:
    print(f"\n{'='*60}")
    print(f"Autoencoder (Joint End-to-End) - d_out={d_out}")
    print(f"{'='*60}")

    compressor = AutoencoderCompressor(d_out=d_out)

    with Timer() as t:
        compressor.fit(train_emb_t, train_labels_t, val_emb_t, val_labels_t)
        
        # Evaluated end-to-end using predict()
        test_preds = compressor.predict(test_emb_t)

    results = evaluate_classifier(test_labels, test_preds, config.LABEL_NAMES)
    results["method"] = "Autoencoder"
    results["d_out"] = d_out
    results["projection_params"] = compressor.get_param_count()
    results["training_log"] = compressor.get_training_log()
    results["time_seconds"] = round(t.elapsed, 2)

    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro-F1: {results['macro_f1']:.4f}")
    print(f"  Projection params: {results['projection_params']}")
    print(f"  Time: {t.elapsed:.2f}s")

    return results


def run_pqc_experiment(d_out: int, train_emb_t: torch.Tensor, val_emb_t: torch.Tensor,
                       test_emb_t: torch.Tensor, train_labels_t: torch.Tensor,
                       val_labels_t: torch.Tensor, test_labels: np.ndarray) -> tuple[dict, list[str]]:
    """Returns (results_dict, anomalies_list)."""
    print(f"\n{'='*60}")
    print(f"PQC (Joint End-to-End) - d_out={d_out}")
    print(f"{'='*60}")

    compressor = PQCCompressor(d_out=d_out)

    with Timer() as t:
        compressor.fit(train_emb_t, train_labels_t, val_emb_t, val_labels_t)
        
        # Evaluated end-to-end using predict()
        test_preds = compressor.predict(test_emb_t)

    results = evaluate_classifier(test_labels, test_preds, config.LABEL_NAMES)
    results["method"] = "PQC"
    results["d_out"] = d_out
    results["projection_params"] = compressor.get_param_count()
    results["training_log"] = compressor.get_training_log()
    results["time_seconds"] = round(t.elapsed, 2)

    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro-F1: {results['macro_f1']:.4f}")
    print(f"  Projection params: {results['projection_params']}")
    print(f"  Time: {t.elapsed:.2f}s")

    return results, compressor.get_anomalies()


def main():
    set_seed(config.SEED)
    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.TRAINING_LOGS_DIR)

    # Collect environment info
    env_info = get_environment_info()
    print("Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # Step 1: Extract and cache embeddings
    print("\n" + "=" * 60)
    print("Step 1: Extracting PhoBERT embeddings")
    print("=" * 60)
    with Timer() as embed_timer:
        extract_and_cache_embeddings()
    print(f"Embedding extraction time: {embed_timer.elapsed:.2f}s")

    # Load cached embeddings
    train_emb_t, train_labels_t = load_cached_embeddings("train")
    val_emb_t, val_labels_t = load_cached_embeddings("validation")
    test_emb_t, test_labels_t = load_cached_embeddings("test")

    # Numpy versions for PCA and evaluation
    train_emb_np = train_emb_t.numpy()
    val_emb_np = val_emb_t.numpy()
    test_emb_np = test_emb_t.numpy()
    train_labels_np = train_labels_t.numpy()
    val_labels_np = val_labels_t.numpy()
    test_labels_np = test_labels_t.numpy()

    print(f"\nDataset sizes: train={len(train_emb_t)}, val={len(val_emb_t)}, test={len(test_emb_t)}")
    print(f"Train label distribution: {torch.bincount(train_labels_t).tolist()}")

    all_results = []
    experiment_times = {}
    anomalies = []

    # --- PCA experiment ---
    try:
        result = run_pca_experiment(
            config.D_OUT, train_emb_np, val_emb_np, test_emb_np,
            train_labels_np, val_labels_np, test_labels_np
        )
        all_results.append(result)
        experiment_times[f"PCA_d{config.D_OUT}"] = result["time_seconds"]
    except Exception as e:
        msg = f"PCA d_out={config.D_OUT} FAILED: {e}"
        print(f"  ERROR: {msg}")
        anomalies.append(msg)

    # --- Autoencoder experiment ---
    try:
        result = run_autoencoder_experiment(
            config.D_OUT, train_emb_t, val_emb_t, test_emb_t,
            train_labels_t, val_labels_t, test_labels_np
        )
        all_results.append(result)
        experiment_times[f"AE_d{config.D_OUT}"] = result["time_seconds"]

        # Save training log
        log_path = os.path.join(config.TRAINING_LOGS_DIR, f"autoencoder_d{config.D_OUT}.json")
        with open(log_path, "w") as f:
            json.dump(result["training_log"], f, indent=2)
    except Exception as e:
        msg = f"Autoencoder d_out={config.D_OUT} FAILED: {e}"
        print(f"  ERROR: {msg}")
        anomalies.append(msg)

    # --- PQC experiment ---
    try:
        result, pqc_anomalies = run_pqc_experiment(
            config.D_OUT, train_emb_t, val_emb_t, test_emb_t,
            train_labels_t, val_labels_t, test_labels_np
        )
        all_results.append(result)
        experiment_times[f"PQC_d{config.D_OUT}"] = result["time_seconds"]
        anomalies.extend(pqc_anomalies)  # Propagate PQC stagnation warnings

        # Save training log
        log_path = os.path.join(config.TRAINING_LOGS_DIR, f"pqc_d{config.D_OUT}.json")
        with open(log_path, "w") as f:
            json.dump(result["training_log"], f, indent=2)
    except Exception as e:
        msg = f"PQC d_out={config.D_OUT} FAILED: {e}"
        print(f"  ERROR: {msg}")
        anomalies.append(msg)

    # Compile final outputs
    output = {
        "environment": env_info,
        "embedding_extraction_time_seconds": round(embed_timer.elapsed, 2),
        "experiment_times": experiment_times,
        "results": [],
        "anomalies": anomalies,
    }

    for result in all_results:
        entry = {
            "method": result["method"],
            "d_out": result["d_out"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "projection_params": result["projection_params"],
            "time_seconds": result["time_seconds"],
            "classification_report": result["classification_report"]
        }
        output["results"].append(entry)

    # Save final metrics
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'d_out':>5} {'Accuracy':>10} {'Macro-F1':>10} {'Params':>10} {'Time(s)':>10}")
    print("-" * 60)
    for r in output["results"]:
        print(f"{r['method']:<15} {r['d_out']:>5} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['projection_params']:>10} {r['time_seconds']:>10.1f}")

    print(f"\nResults saved to: {metrics_path}")

    if anomalies:
        print(f"\n[WARNING] ANOMALIES ({len(anomalies)}):")
        for a in anomalies:
            print(f"  - {a}")

    print("\nDone.")


if __name__ == "__main__":
    main()
