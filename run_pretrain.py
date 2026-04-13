# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on Chinese Wikipedia text data.

Before running this script, make sure you:
1. Extracted and preprocessed the text data
2. Trained a BPE tokenizer on the text data
"""

import argparse
import math
import time
from pathlib import Path

from tokenizers import Tokenizer
import torch
from utils import (
    GPTModel,
    calc_loss_batch,
    create_dataloader_v1,
    evaluate_model,
    plot_losses,
    read_data_from_path,
)


def create_dataloaders(
    text_data,
    tokenizer,
    train_ratio,
    batch_size,
    max_length,
    stride,
    num_workers=0,
):
    """Create training and validation dataloaders from text data."""
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def convert_time(seconds):
    """Convert seconds to hours, minutes, seconds."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def train_model_simple(
    model,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    output_dir,
    save_ckpt_freq,
    tokenizer,
    data_path,
    batch_size=1024,
    train_ratio=0.90,
    max_length=1024,
    stride=None,
    max_chars=None,
    warmup_steps=2000,
    max_train_tokens=None,
    num_workers=0,
):
    """
    Simple training loop for GPT model.
    
    Args:
        model: The GPT model to train
        optimizer: The optimizer
        device: Device to train on
        n_epochs: Number of epochs to train
        eval_freq: Evaluate every N steps
        eval_iter: Number of iterations for evaluation
        output_dir: Directory to save checkpoints
        save_ckpt_freq: Save checkpoint every N steps
        tokenizer: Tokenizer for encoding text
        data_path: Path to the training data file or directory
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training (rest for validation)
        
    Returns:
        Tuple of (train_losses, val_losses, track_tokens_seen, track_steps)
    """
    train_losses, val_losses, track_tokens_seen, track_steps = [], [], [], []
    tokens_seen = 0
    global_step = 0
    start_time = time.time()
    output_dir = Path(output_dir)

    text_data = read_data_from_path(data_path)
    if max_chars is not None and len(text_data) > max_chars:
        text_data = text_data[:max_chars]

    eot = "<|endoftext|>"
    if not text_data.rstrip().endswith(eot):
        text_data = text_data.rstrip() + "\n" + eot

    if stride is None:
        stride = max_length // 2

    train_loader, val_loader = create_dataloaders(
        text_data,
        tokenizer=tokenizer,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        num_workers=num_workers,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training DataLoader is empty (drop_last=True). "
            "Use more text, smaller batch_size, or shorter max_length."
        )

    total_steps = max(1, n_epochs * len(train_loader))
    warmup_steps_eff = min(warmup_steps, total_steps)
    base_lr = optimizer.param_groups[0]["lr"]

    tokens_per_step = batch_size * max_length
    approx_tokens_per_epoch = len(train_loader) * tokens_per_step
    vocab_size = tokenizer.get_vocab_size()
    print(
        f"Train batches/epoch: {len(train_loader)}, val batches: {len(val_loader)}, "
        f"vocab_size={vocab_size}, context_length={max_length}, "
        f"chars={len(text_data):,}, stride={stride}"
    )
    print(
        f"Approx. train tokens per epoch (upper bound): {approx_tokens_per_epoch:,} "
        f"(~{tokens_per_step:,} tokens/step)"
    )
    if max_train_tokens is not None:
        print(f"Will stop after about {max_train_tokens:,} training tokens seen.")

    stop_training = False
    try:
        for epoch in range(n_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                global_step += 1
                if warmup_steps_eff > 0 and global_step <= warmup_steps_eff:
                    lr_mult = float(global_step) / float(warmup_steps_eff)
                else:
                    denom = max(1, total_steps - warmup_steps_eff)
                    progress = float(global_step - warmup_steps_eff) / float(denom)
                    progress = min(1.0, max(0.0, progress))
                    min_ratio = 0.1
                    lr_mult = min_ratio + (1.0 - min_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * lr_mult

                optimizer.zero_grad(set_to_none=True)
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()

                tokens_seen += int(input_batch.numel())

                if max_train_tokens is not None and tokens_seen >= max_train_tokens:
                    stop_training = True

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    track_steps.append(global_step)
                    elapsed = time.time() - start_time
                    h, m, s = convert_time(elapsed)
                    print(
                        f"step {global_step} | epoch {epoch + 1}/{n_epochs} | "
                        f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                        f"tokens {tokens_seen:,} | time {h}h{m}m{s}s"
                    )

                if global_step % save_ckpt_freq == 0 and global_step > 0:
                    ckpt_path = output_dir / f"checkpoint_step_{global_step}.pth"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

                if stop_training:
                    if global_step % eval_freq != 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter
                        )
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        track_steps.append(global_step)
                        elapsed = time.time() - start_time
                        h, m, s = convert_time(elapsed)
                        print(
                            f"step {global_step} | epoch {epoch + 1}/{n_epochs} | "
                            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                            f"tokens {tokens_seen:,} | time {h}h{m}m{s}s "
                            f"(final eval at token budget)"
                        )
                    break

            if stop_training:
                break

    except KeyboardInterrupt:
        ckpt_path = output_dir / "checkpoint_interrupted.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Interrupted. Saved: {ckpt_path}")
        raise

    final_path = output_dir / "model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights: {final_path}")

    if max_train_tokens is not None and tokens_seen < max_train_tokens:
        print(
            f"Note: stopped at {tokens_seen:,} tokens (< {max_train_tokens:,}). "
            "Increase --n_epochs or data size to reach the token budget."
        )

    return train_losses, val_losses, track_tokens_seen, track_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GPT Model Training Configuration",
    )

    parser.add_argument(
        "--data_file", "--data",
        type=str,
        required=True,
        help="Path to the training data file or directory containing .txt files (e.g., data/wiki_zh_2019.txt or data/wiki_zh_2019/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_checkpoints",
        help="Directory where the model checkpoints will be saved",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer JSON file (e.g., wikizh_tokenizer.json)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluations during training (in steps)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        type=int,
        default=100_000,
        help="Frequency of saving model checkpoints during training (in steps)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.90, help="Ratio of data for training (rest for validation)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=52000, help="Vocabulary size (should match tokenizer)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Uses a very small model for debugging purposes",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Optional cap on UTF-8 characters loaded from data (for memory / quick runs).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Linear LR warmup steps (capped by total training steps).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride; default is context_length // 2.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=100_000_000,
        help="Stop after this many training tokens (set to 0 to disable). "
        "Instructor expectation: val loss < 5 after ~100M tokens.",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=20,
        help="Number of batches to average for train/val loss during evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 is safest on Windows).",
    )

    args = parser.parse_args()

    # Set model configuration
    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,
            "context_length": 10,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    else:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,  # Should match tokenizer vocab size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
        }

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Verify vocab size matches
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) doesn't match --vocab_size ({args.vocab_size})")
        print(f"Updating model config to use vocab size: {actual_vocab_size}")
        GPT_CONFIG_124M["vocab_size"] = actual_vocab_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Initialize model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir.absolute()}")

    # Train model
    print("\nStarting training...")
    max_length = GPT_CONFIG_124M["context_length"]
    max_train_tokens = (
        None if args.max_train_tokens == 0 else args.max_train_tokens
    )
    train_losses, val_losses, track_tokens_seen, track_steps = train_model_simple(
        model=model,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        tokenizer=tokenizer,
        data_path=args.data_file,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        max_length=max_length,
        stride=args.stride,
        max_chars=args.max_chars,
        warmup_steps=args.warmup_steps,
        max_train_tokens=max_train_tokens,
        num_workers=args.num_workers,
    )

    if train_losses:
        steps_t = torch.tensor(track_steps, dtype=torch.float)
        tok_t = torch.tensor(track_tokens_seen, dtype=torch.float)
        loss_fig = output_dir / "loss.pdf"
        plot_losses(
            steps_t,
            tok_t,
            train_losses,
            val_losses,
            x_label="Global step",
            save_path=str(loss_fig),
        )
        print(f"Saved loss curve: {loss_fig}")
    
    # Print GPU memory usage if CUDA is available
    if torch.cuda.is_available():
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Training completed!")
