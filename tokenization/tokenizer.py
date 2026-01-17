"""BPE Tokenizer for encoding/decoding."""

import regex as re

# Splits text into linguistically meaningful chunks 
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """BPE Tokenizer."""
    
    def __init__(self, vocab, merges, special_tokens=None):
        # Used copy to prevent accidental mutation
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []
        
        # Reverse vocab
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        
        # Build merge rules
        self.merge_rules = {}
        for merge_idx, (left_bytes, right_bytes) in enumerate(self.merges):
            left_id = self.byte_to_id[left_bytes]
            right_id = self.byte_to_id[right_bytes]
            merged_bytes = left_bytes + right_bytes
            merged_id = self.byte_to_id[merged_bytes]
            self.merge_rules[(left_id, right_id)] = (merged_id, merge_idx)
    
    def _apply_bpe(self, word_tokens):
        """Apply BPE merges."""
        if len(word_tokens) < 2:
            return word_tokens
        
        while True:
            pairs = []
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                if pair in self.merge_rules:
                    merged_id, merge_idx = self.merge_rules[pair]
                    pairs.append((i, merged_id, merge_idx))
            
            if not pairs:
                break
            
            # Apply earliest merge
            pairs.sort(key=lambda x: x[2])
            pos, merged_id, _ = pairs[0]
            
            word_tokens = word_tokens[:pos] + [merged_id] + word_tokens[pos + 2:]
        
        return word_tokens
    
    def encode(self, text):
        """Encode text to token IDs."""
        token_ids = []
        
        for match in re.finditer(GPT2_SPLIT_PATTERN, text):
            token_bytes = match.group().encode('utf-8')
            word_tokens = [self.byte_to_id.get(bytes([b]), b) for b in token_bytes]
            merged = self._apply_bpe(word_tokens)
            token_ids.extend(merged)
        
        return token_ids
    
    def decode(self, ids):
        """Decode token IDs to text."""
        byte_sequences = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
            else:
                byte_sequences.append(bytes([token_id]))
        
        all_bytes = b''.join(byte_sequences)
        return all_bytes.decode('utf-8', errors='replace')
