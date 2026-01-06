#!/usr/bin/env python3
"""
Train GPT on TinyStories.

Usage:
    python scripts/train_gpt.py --config config/gpt_tinystories.yaml
"""

import argparse
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.language_model import TransformerLM
from training.optimizer import AdamW
from training.loss import cross_entropy
from training.utils import get_batch, gradient_clipping, get_lr_cosine_schedule


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """Estimate train and val loss."""
    model.eval()
    losses = {}
    
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(config['training']['eval_iters']):
            x, y = get_batch(
                data,
                config['training']['batch_size'],
                config['model']['context_length'],
                device
            )
            logits = model(x, apply_softmax=False)
            loss = cross_entropy(logits, y)
            split_losses.append(loss.item())
        
        losses[split] = np.mean(split_losses)
    
    model.train()
    return losses


def train(config_path):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print("="*70)
    print(f"Training: {config['experiment_name']}")
    print("="*70)
    
    # Set device
    device = get_device()
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Load data
    print("Loading data...")
    processed_dir = config['data']['processed_dir']  # probably "data/processed"

    train_data = np.memmap(
        f"{processed_dir}/train.bin",
        dtype='uint16',
        mode='r'
    )
    val_data = np.memmap(
        f"{processed_dir}/val.bin",
        dtype='uint16',
        mode='r'
    )     
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        rope_theta=config['model']['rope_theta'],
        device=device,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create checkpoint dir
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    model.train()
    
    pbar = tqdm(range(config['training']['max_iters']), desc="Training")
    
    for iter_num in pbar:
        # Get learning rate
        lr = get_lr_cosine_schedule(
            iter_num,
            config['training']['learning_rate'],
            config['training']['min_lr'],
            config['training']['warmup_iters'],
            config['training']['max_iters'],
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % config['training']['eval_interval'] == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(f"\n[{iter_num:5d}] train: {losses['train']:.4f}, val: {losses['val']:.4f}, "
                  f"ppl: {np.exp(losses['val']):.2f}")
        
        # Training step
        x, y = get_batch(
            train_data,
            config['training']['batch_size'],
            config['model']['context_length'],
            device
        )
        
        logits = model(x, apply_softmax=False)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        if config['training']['grad_clip'] > 0:
            gradient_clipping(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
        
        # Checkpoint
        if iter_num > 0 and iter_num % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iter_num,
                'config': config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': config['training']['max_iters'],
        'config': config,
    }, final_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    train(args.config)


if __name__ == "__main__":
    main()
