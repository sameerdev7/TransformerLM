#!/usr/bin/env python3
"""
Prepare TinyStories dataset from HuggingFace.
Usage:
    python scripts/prepare_tinystories.py
    python scripts/prepare_tinystories.py --max-samples 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tokenization.bpe_trainer import train_bpe_tokenizer, save_tokenizer
from tokenization.tokenizer import Tokenizer


def prepare_tinystories_data(
    dataset_name: str,
    train_split: str,
    val_split: str,
    tokenizer,
    output_dir: str,
    max_samples: int | None = None,
    verbose: bool = True,
):
    if verbose:
        print("=" * 70)
        print("LOADING TINYSTORIES FROM HuggingFace")
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Train split: {train_split}")
        print(f"Val split: {val_split}")

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Use correct split names
    train_data = dataset[train_split]
    val_data = dataset[val_split]

    if max_samples is not None:
        train_data = train_data.select(range(min(max_samples, len(train_data))))

    # Get the token ID for <|endoftext|>
    eot_token = tokenizer.encode("<|endoftext|>")[0]  # Since it's a special token, encode returns a single ID

    def encode_split(split):
        ids = []
        for ex in split:
            ids.extend(tokenizer.encode(ex["text"]))
            ids.append(eot_token)
        return np.array(ids, dtype=np.uint16)

    train_ids = encode_split(train_data)
    val_ids = encode_split(val_data)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    if verbose:
        print(f"Saved train.bin ({len(train_ids):,} tokens)")
        print(f"Saved val.bin ({len(val_ids):,} tokens)")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories data")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    args = parser.parse_args()

    print("=" * 70)
    print("TINYSTORIES DATA PREPARATION")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Vocab size: {args.vocab_size:,}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,}")
    print("=" * 70)
    print()

    # Create directories
    Path("data/tokenizer").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Load dataset for tokenizer training
    print("Loading dataset for tokenizer training...")
    dataset = load_dataset(args.dataset, split="train")

    # Sample text for tokenizer (first 10k stories)
    sample_size = min(10000, len(dataset))
    sample_texts = [dataset[i]["text"] for i in range(sample_size)]
    sample_text = "\n".join(sample_texts)

    # Save sample text
    sample_path = Path("data/raw/tokenizer_sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"Saved {sample_size:,} stories to {sample_path}")

    # Train tokenizer
    print("\nTraining BPE tokenizer...")
    vocab, merges = train_bpe_tokenizer(
        input_path=str(sample_path),
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        verbose=True,
        use_multiprocessing=False,
    )

    # Save tokenizer
    vocab_path = "data/tokenizer/vocab.pkl"
    merges_path = "data/tokenizer/merges.pkl"
    save_tokenizer(vocab, merges, vocab_path, merges_path)
    print("Saved tokenizer to data/tokenizer/")

    # Create tokenizer instance
    tokenizer = Tokenizer(vocab, merges)

    # Tokenize full dataset
    print("\nTokenizing full dataset...")
    prepare_tinystories_data(
        dataset_name=args.dataset,
        train_split="train",
        val_split="validation",
        tokenizer=tokenizer,
        output_dir="data/processed",
        max_samples=args.max_samples,
        verbose=True,
    )


if __name__ == "__main__":
    main()
