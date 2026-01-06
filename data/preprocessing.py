import numpy as np 
from pathlib import Path 
from datasets import load_dataset 
from tqdm import tqdm 


def prepare_tinystories_data(
    dataset_name: str, 
    train_split: str, 
    val_split: str, 
    tokenizer, 
    output_dir: str, 
    max_samples: int = None, 
    verbose: bool = True
):

    """
    Download and tokenize TinyStories from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name
        train_split: Training split name
        val_split: Validation split name  
        tokenizer: Tokenizer instance
        output_dir: Directory to save tokenized data
        max_samples: Max samples to use (None = all)
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose: 
        print("="*70)
        print("LOADING TINYSTORIES FROM HuggingFace")
        print("="*70)
        print(f"Dataset: {dataset_name}")
        print(f"Train split: {train_split}")
        print(f"Val split: {val_split}")


    dataset = load_dataset(dataset_name)

    train_data = dataset[dataset_name]
    val_data = dataset[val_split]


    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_samples = min(max_samples // 10, len(val_data))
        val_data = val_data.select(range(val_samples))

    if verbose: 
        print(f"Train samples: {len(train_data):,}")
        print(f"Val samples: {len(val_data):,}")

    if verbose: 
        print(f"\nTokenizing training data....")


    train_tokens = [] 
    for example in tqdm(train_data, disable=not verbose):
        text = example['text']
        tokens = tokenizer.encode(text)
        train_tokens.extend(tokens)


    train_tokens = np.array(train_tokens, dtype=np.uint16)
    train_path = output_path / "train_tokens.npy"
    np.save(train_path, train_tokens)


    if verbose: 
        print(f"Train tokens: {len(train_tokens):,}")
        print(f"Saved to: {train_path}")

     # Tokenize validation data
    if verbose:
        print(f"\nTokenizing validation data...")
    
    val_tokens = []
    for example in tqdm(val_data, disable=not verbose):
        text = example['text']
        tokens = tokenizer.encode(text)
        val_tokens.extend(tokens)
    
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    val_path = output_path / "val_tokens.npy"
    np.save(val_path, val_tokens)
    
    if verbose:
        print(f"Val tokens: {len(val_tokens):,}")
        print(f"Saved to: {val_path}")
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETE")
        print("="*70)
    
    return train_tokens, val_tokens


