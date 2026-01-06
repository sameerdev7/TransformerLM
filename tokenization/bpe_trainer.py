"""Simplified BPE tokenizer training."""

import pickle
from collections import defaultdict
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe_tokenizer(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str], 
    verbose: bool = True,
    use_multiprocessing: bool = False
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer."""
    
    if verbose:
        print(f"Training BPE tokenizer (vocab_size={vocab_size})")
    
    # Initialize vocab with bytes
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # Add special tokens
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # Read text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Count word frequencies
    word_counts = defaultdict(int)
    for match in re.finditer(GPT2_SPLIT_PATTERN, text):
        token_bytes = tuple(match.group().encode('utf-8'))
        word_counts[token_bytes] += 1
    
    if verbose:
        print(f"Found {len(word_counts):,} unique words")
    
    # Convert to working format
    word_freq = {tuple(list(w)): c for w, c in word_counts.items()}
    
    merges = []
    target_merges = vocab_size - len(vocab)
    
    if verbose:
        print(f"Performing {target_merges} merges...")
    
    for merge_num in range(target_merges):
        # Count pairs
        pair_counts = defaultdict(int)
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        
        if not pair_counts:
            break
        
        # Find best pair
        best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
        
        # Create new token
        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]
        vocab[next_id] = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))
        
        # Update word frequencies
        new_word_freq = {}
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == best_pair[0] and 
                    word[i + 1] == best_pair[1]):
                    new_word.append(next_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            new_word_freq[new_word_tuple] = new_word_freq.get(new_word_tuple, 0) + freq
        
        word_freq = new_word_freq
        next_id += 1
        
        if verbose and (merge_num + 1) % 500 == 0:
            print(f"  Merge {merge_num + 1}/{target_merges}")
    
    if verbose:
        print(f"Training complete! Vocab size: {len(vocab)}")
    
    return vocab, merges


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    """Save tokenizer to files."""
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)


def load_tokenizer(vocab_path, merges_path):
    """Load tokenizer from files."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    return vocab, merges
