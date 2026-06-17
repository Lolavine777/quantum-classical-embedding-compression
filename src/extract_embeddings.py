import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from underthesea import word_tokenize

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, ensure_dir


def segment_vietnamese(text: str) -> str:
    """Word-segment Vietnamese text for PhoBERT tokenization using underthesea."""
    return word_tokenize(text, format="text")


def extract_and_cache_embeddings() -> None:
    """
    Load UIT-VSFC dataset, extract frozen PhoBERT [CLS] embeddings
    for all splits, and save to disk.
    """
    set_seed(config.SEED)
    ensure_dir(config.EMBEDDING_DIR)

    # Check if embeddings already cached
    all_cached = all(
        os.path.exists(os.path.join(config.EMBEDDING_DIR, f"{split}_embeddings.pt"))
        for split in ["train", "validation", "test"]
    )
    if all_cached:
        print("Embeddings already cached. Skipping extraction.")
        return

    print("Loading dataset...")
    dataset = load_dataset(config.DATASET_NAME)

    print("Loading PhoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_MODEL)
    model = AutoModel.from_pretrained(config.PHOBERT_MODEL)
    model = model.to(config.DEVICE)
    model.eval()

    split_map = {"train": "train", "validation": "validation", "test": "test"}

    for split_name, dataset_split_key in split_map.items():
        print(f"\nExtracting embeddings for {split_name} split...")
        data = dataset[dataset_split_key]

        sentences = data["sentence"]
        labels = data["sentiment"]

        print("  Word segmenting Vietnamese text...")
        segmented = [segment_vietnamese(s) for s in sentences]

        print("  Sorting by length to optimize batching...")
        sorted_data = sorted(enumerate(segmented), key=lambda x: len(x[1]))
        orig_indices, sorted_segmented = zip(*sorted_data)

        all_embeddings_sorted = []
        batch_size = 64

        print("  Running PhoBERT inference...")
        for start_idx in range(0, len(sorted_segmented), batch_size):
            end_idx = min(start_idx + batch_size, len(sorted_segmented))
            batch_sentences = sorted_segmented[start_idx:end_idx]

            # Tokenize
            encoded = tokenizer(
                list(batch_sentences),
                padding=True,
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(config.DEVICE) for k, v in encoded.items()}

            # Extract [CLS] embeddings (the <s> token at index 0)
            with torch.inference_mode():
                outputs = model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

            all_embeddings_sorted.append(cls_embeddings.cpu())

            if (start_idx // batch_size) % 20 == 0:
                print(f"    Processed {end_idx}/{len(sorted_segmented)} samples")

        embeddings_tensor_sorted = torch.cat(all_embeddings_sorted, dim=0)  # (N, 768)

        # Restore original order
        embeddings_tensor = torch.zeros_like(embeddings_tensor_sorted)
        for sorted_idx, orig_idx in enumerate(orig_indices):
            embeddings_tensor[orig_idx] = embeddings_tensor_sorted[sorted_idx]

        labels_tensor = torch.tensor(labels, dtype=torch.long)  # (N,)

        # Save
        torch.save(embeddings_tensor, os.path.join(config.EMBEDDING_DIR, f"{split_name}_embeddings.pt"))
        torch.save(labels_tensor, os.path.join(config.EMBEDDING_DIR, f"{split_name}_labels.pt"))
        print(f"  Saved {split_name}: embeddings {embeddings_tensor.shape}, labels {labels_tensor.shape}")

    print("\nEmbedding extraction complete.")


def load_cached_embeddings(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load cached embeddings and labels for a given split.

    Args:
        split: One of 'train', 'validation', 'test'

    Returns:
        Tuple of (embeddings [N, 768], labels [N])
    """
    embeddings = torch.load(
        os.path.join(config.EMBEDDING_DIR, f"{split}_embeddings.pt"),
        weights_only=True,
    )
    labels = torch.load(
        os.path.join(config.EMBEDDING_DIR, f"{split}_labels.pt"),
        weights_only=True,
    )
    return embeddings, labels


if __name__ == "__main__":
    extract_and_cache_embeddings()

    # Verify
    for split in ["train", "validation", "test"]:
        emb, lab = load_cached_embeddings(split)
        print(f"{split}: embeddings={emb.shape}, labels={lab.shape}, "
              f"label distribution={torch.bincount(lab).tolist()}")
