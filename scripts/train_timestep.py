#!/usr/bin/env python3
"""
Minimal training for BertDiffusionLM.

Only trains the timestep embedding (~4.7M params).
BERT's 110M params stay frozen.

This teaches the model to understand "how noisy is this input"
so it can adjust predictions based on diffusion timestep.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BertTokenizer
from datasets import load_dataset

from constrained_diffusion_lm.models.bert_diffusion import BertDiffusionLM
from constrained_diffusion_lm.diffusion import get_schedule


def create_dataloader(tokenizer, max_length=128, batch_size=16, max_samples=50000):
    """Load AG News dataset."""
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    
    # Tokenize
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_step(model, batch, schedule, optimizer, device, tokenizer):
    """Single training step."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    batch_size, seq_len = input_ids.shape
    
    # Sample random timesteps
    t = torch.randint(1, schedule.num_timesteps + 1, (batch_size,), device=device)
    
    # Get masking probability for each timestep
    gamma = schedule(t).unsqueeze(1)  # [B, 1]
    
    # Create noise mask (don't mask special tokens)
    special_mask = (input_ids == tokenizer.pad_token_id) | \
                   (input_ids == tokenizer.cls_token_id) | \
                   (input_ids == tokenizer.sep_token_id)
    
    noise_mask = (torch.rand(batch_size, seq_len, device=device) < gamma) & ~special_mask
    
    # Apply masking
    x_t = input_ids.clone()
    x_t[noise_mask] = tokenizer.mask_token_id
    
    # Forward pass
    logits = model(x_t, t, attention_mask)
    
    # Compute loss only on masked positions
    targets = input_ids.clone()
    targets[~noise_mask] = -100  # Ignore non-masked
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Accuracy on masked tokens
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correct = (predictions == input_ids) & noise_mask
        accuracy = correct.sum().float() / noise_mask.sum().float()
    
    return loss.item(), accuracy.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = BertDiffusionLM(freeze_bert=True, max_timesteps=args.timesteps)
    model.to(device)
    
    # Only optimize timestep embedding
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    
    # Load data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataloader = create_dataloader(
        tokenizer, 
        batch_size=args.batch_size, 
        max_samples=args.max_samples
    )
    
    # Schedule
    schedule = get_schedule("cosine", args.timesteps)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Trainable params: {model.get_num_trainable_params():,}")
    print(f"Batches per epoch: {len(dataloader)}")
    print()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            loss, acc = train_step(model, batch, schedule, optimizer, device, tokenizer)
            total_loss += loss
            total_acc += acc
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")
    
    # Save checkpoint
    save_path = Path("checkpoints/bert_diffusion.pt")
    save_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        "timestep_embedding": model.timestep_embedding.state_dict(),
        "time_norm": model.time_norm.state_dict(),
        "config": {
            "max_timesteps": args.timesteps,
            "bert_model_name": "bert-base-uncased",
        }
    }, save_path)
    print(f"\nSaved to {save_path}")
    
    # Quick test
    print("\n--- Testing ---")
    model.eval()
    
    test_texts = [
        ("Hello my name is suneru", ["suneru"]),
        ("The weather is nice today", ["weather"]),
        ("I love programming in Python", ["Python"]),
    ]
    
    for text, locked in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        x_0 = inputs["input_ids"]
        
        # Create lock mask
        tokens = tokenizer.convert_ids_to_tokens(x_0[0])
        lock_mask = torch.zeros_like(x_0, dtype=torch.bool)
        for i, tok in enumerate(tokens):
            for lw in locked:
                if lw.lower() in tok.lower().replace("##", ""):
                    lock_mask[0, i] = True
        
        # Mask editable tokens
        x_t = x_0.clone()
        edit_mask = ~lock_mask & (x_0 != tokenizer.pad_token_id) & \
                    (x_0 != tokenizer.cls_token_id) & (x_0 != tokenizer.sep_token_id)
        x_t[edit_mask] = tokenizer.mask_token_id
        
        # Simple greedy decode
        with torch.no_grad():
            for step in [50, 25, 10, 5, 1]:
                t = torch.tensor([step], device=device)
                logits = model(x_t, t, inputs["attention_mask"])
                predictions = logits.argmax(dim=-1)
                
                # Unmask highest confidence
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values
                is_masked = (x_t == tokenizer.mask_token_id)
                
                if is_masked.any():
                    conf_at_masked = confidence.clone()
                    conf_at_masked[~is_masked] = -1
                    _, top_pos = conf_at_masked[0].topk(min(3, is_masked.sum().item()))
                    x_t[0, top_pos] = predictions[0, top_pos]
                
                # Preserve locked
                x_t[lock_mask] = x_0[lock_mask]
        
        # Final fill
        x_t = torch.where(x_t == tokenizer.mask_token_id, predictions, x_t)
        x_t[lock_mask] = x_0[lock_mask]
        
        result = tokenizer.decode(x_t[0], skip_special_tokens=True)
        print(f"'{text}' [lock: {locked}] → '{result}'")


if __name__ == "__main__":
    main()

