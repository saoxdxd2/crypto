"""
Offline Pre-Training Harness for FinCast.

Trains the FinCast decoder-only model using Autoregressive Next-Candle Prediction
on historical OHLCV data. 
Utilizes Mixed Precision (fp16) and Gradient Accumulation for maximum GPU throughput.
Automatically pushes best checkpoints to Hugging Face Hub.
"""
import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from huggingface_hub import HfApi

from src.mission_control.forecast import FinCastModel
from src.cloud.data_loaders import FinCastDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "sao/FinCast-crypto-v1"

def apply_smart_weight_reset(model, shrink_factor=0.9, noise_std=0.01):
    """
    Implements Meta-inspired 'Shrink and Perturb' to prevent catastrophic forgetting
    and loss of plasticity during continual learning. Also completely resets the 
    last linear layer to induce transfer shock.
    """
    import torch.nn as nn
    import torch
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'linear' in name or 'head' in name or 'classifier' in name:
                # Reset the last layer (meta-learning effect / transfer shock)
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
            else:
                # Shrink and Perturb for hidden representations
                param.data.mul_(shrink_factor)
                noise = torch.randn_like(param) * noise_std
                param.data.add_(noise)
    
    logger.info("✅ Applied Smart Weight Reset (Shrink & Perturb + Last-Layer Reset) to maintain plasticity.")

def train(args):
    # Set random seeds for deterministic splits and model weights
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting FinCast Pre-Training on device: {device}")
    
    # 1. Initialize Dataset & DataLoader
    dataset = FinCastDataset(args.data_path, seq_len=args.seq_len)
    
    # 80/20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
    
    # 2. Initialize Model, Optimizer, and Scaler
    model = FinCastModel().to(device)
    if hasattr(torch, "compile"):
        logger.info("⚡ Compiling model with Torch 2.0...")
        model = torch.compile(model)
        
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        logger.info("✅ 8-bit AdamW activated (75% VRAM saving)")
    except ImportError:
        logger.warning("⚠️ bitsandbytes not found, falling back to standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    scaler = GradScaler('cuda', enabled=args.fp16)
    
    # Auto-resume to support Continuous Learning Loop
    ckpt_loaded = False
    local_ckpt = Path("checkpoints/fincast_checkpoint.pt")
    
    if local_ckpt.exists():
        logger.info(f"Attempting to resume from local checkpoint {local_ckpt}...")
        try:
            checkpoint = torch.load(local_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"✅ Successfully loaded model and optimizer states from {local_ckpt}")
            ckpt_loaded = True
        except Exception as e:
            logger.warning(f"Error loading local checkpoint: {e}")
            
    elif args.resume_from_hub:
        logger.info(f"Attempting to resume from Hugging Face Hub ({REPO_ID})...")
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="fincast_checkpoint.pt", token=HF_TOKEN)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"✅ Successfully loaded model and optimizer states from {ckpt_path}")
            ckpt_loaded = True
        except Exception as e:
            logger.warning(f"No existing checkpoint found or error downloading: {e}. Starting from scratch.")
            
    # Apply Continuous Learning Plasticity if we are resuming an older checkpoint
    if ckpt_loaded:
        apply_smart_weight_reset(model)
    
    # Autoregressive return prediction (regression task)
    criterion = nn.MSELoss()
    
    best_acc = 0.0
    model_dir = Path("checkpoints")
    model_dir.mkdir(exist_ok=True)
    
    # --- ZERO-SHOT EVALUATION ---
    model.eval()
    val_loss = 0.0
    correct_direction = 0
    total_samples = 0
    
    with torch.no_grad():
        for ohlcv_seq, target_returns in val_loader:
            ohlcv_seq = ohlcv_seq.to(device, non_blocking=True)
            target_returns = target_returns.to(device, non_blocking=True)
            
            with autocast('cuda', enabled=args.fp16):
                predictions = model(ohlcv_seq)
                last_token_preds = predictions[:, -1]
                loss = criterion(last_token_preds, target_returns)
            val_loss += loss.item()
            
            pred_up = last_token_preds > 0
            target_up = target_returns > 0
            correct_direction += (pred_up == target_up).sum().item()
            total_samples += target_returns.size(0)
            
    val_acc = correct_direction / total_samples if total_samples > 0 else 0.0
    val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    logger.info(f"[ZERO-SHOT EVAL] Loss: {val_loss_avg:.6f} | Directional Acc: {val_acc:.4f}")
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for step, (ohlcv_seq, target_returns) in enumerate(loader):
            oom_retries = 0
            while oom_retries < 3:
                try:
                    ohlcv_seq = ohlcv_seq.to(device, non_blocking=True)
                    target_returns = target_returns.to(device, non_blocking=True)
                    
                    with autocast('cuda', enabled=args.fp16):
                        # Model returns (B, SeqLen). We take the prediction for the last token.
                        predictions = model(ohlcv_seq)
                        last_token_preds = predictions[:, -1]
                        
                        loss = criterion(last_token_preds, target_returns)
                        loss = loss / args.accumulate_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % args.accumulate_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                    epoch_loss += loss.item() * args.accumulate_steps
                    break  # Success, break out of retry loop
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_retries += 1
                        logger.warning(f"⚠️ CUDA OOM! (cudf might be extracting data). Clearing cache & waiting 5s (Retry {oom_retries}/3)...")
                        if 'predictions' in locals(): del predictions
                        if 'last_token_preds' in locals(): del last_token_preds
                        if 'loss' in locals(): del loss
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        import time
                        time.sleep(5)
                        if oom_retries == 3:
                            logger.error("❌ Unrecoverable OOM. Skipping this batch.")
                    else:
                        raise e
            
            if step % 50 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{step}/{len(loader)}], Loss: {loss.item()*args.accumulate_steps:.6f}")
                
                # Push to hub or sync to websocket mid-epoch
                if step > 0 and step % args.save_every_steps == 0:
                    ckpt_path = model_dir / "fincast_checkpoint.pt"
                    tmp_path = model_dir / "fincast_checkpoint.pt.tmp"
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict()
                    }
                    torch.save(checkpoint, tmp_path)
                    os.replace(tmp_path, ckpt_path)
                    
                    if args.use_gdrive:
                        import shutil
                        gdrive_path = Path("/content/drive/MyDrive/checkpoints/fincast_checkpoint.pt")
                        gdrive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(ckpt_path, gdrive_path)
                        logger.info(f"✅ Fast-copied checkpoint to Google Drive: {gdrive_path}")
                    
                    if args.push_to_hub:
                        import threading
                        threading.Thread(target=push_model_to_hub, args=(ckpt_path,)).start()
        
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1} training completed. Average Loss: {avg_loss:.6f}")
        
        # --- Validation & Capability Benchmark ---
        model.eval()
        val_loss = 0.0
        correct_direction = 0
        total_samples = 0
        
        with torch.no_grad():
            for ohlcv_seq, target_returns in val_loader:
                ohlcv_seq = ohlcv_seq.to(device, non_blocking=True)
                target_returns = target_returns.to(device, non_blocking=True)
                
                with autocast('cuda', enabled=args.fp16):
                    predictions = model(ohlcv_seq)
                    last_token_preds = predictions[:, -1]
                    loss = criterion(last_token_preds, target_returns)
                val_loss += loss.item()
                
                pred_up = last_token_preds > 0
                target_up = target_returns > 0
                correct_direction += (pred_up == target_up).sum().item()
                total_samples += target_returns.size(0)
                
        val_acc = correct_direction / total_samples if total_samples > 0 else 0.0
        val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        logger.info(f"📊 Validation Benchmarks | Loss: {val_loss_avg:.6f} | Directional Acc: {val_acc:.4f} (Random Baseline: 0.5000)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = model_dir / "fincast_checkpoint.pt"
            tmp_path = model_dir / "fincast_checkpoint.pt.tmp"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, ckpt_path)
            logger.info(f"Saved new best checkpoint: {ckpt_path}")
            
            if args.use_gdrive:
                import shutil
                gdrive_path = Path("/content/drive/MyDrive/checkpoints/fincast_checkpoint.pt")
                gdrive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt_path, gdrive_path)
                logger.info(f"✅ Fast-copied checkpoint to Google Drive: {gdrive_path}")
            
            if args.push_to_hub:
                import threading
                threading.Thread(target=push_model_to_hub, args=(ckpt_path,)).start()
                
        if val_acc >= args.target_acc:
            logger.info(f"🎯 Benchmark threshold met! Directional Accuracy {val_acc:.4f} >= {args.target_acc:.4f}. Stopping training.")
            break
        else:
            logger.info(f"❌ Model accuracy ({val_acc:.4f}) below acceptable threshold ({args.target_acc:.4f}). Continuing training...")
                
        if args.unlimited:
            logger.info("Unlimited mode enabled. Resetting dataset for next pass...")
            pass

    if args.auto_shutdown:
        logger.info("Auto-shutdown enabled. Terminating Colab runtime to save credits...")
        try:
            from google.colab import runtime
            runtime.unassign()
        except ImportError:
            logger.warning("Not running in Google Colab. Skipping auto-shutdown.")
            import sys
            sys.exit(0)

def push_model_to_hub(filepath: Path):
    logger.info(f"Pushing {filepath.name} to Hugging Face Hub ({REPO_ID})...")
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=REPO_ID,
            repo_type="model"
        )
        logger.info("✅ Successfully pushed to Hugging Face Hub!")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinCast Pre-Training Harness")
    parser.add_argument("--data_path", type=str, default="data/ohlcv_history.parquet", help="Path to Parquet data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulate_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Enable Mixed Precision")
    parser.add_argument("--push_to_hub", action="store_true", help="Automatically upload to HF")
    parser.add_argument("--unlimited", action="store_true", help="Run indefinitely, overriding epochs")
    parser.add_argument("--auto_shutdown", action="store_true", help="Terminate Colab session when done to save credits")
    parser.add_argument("--resume_from_hub", action="store_true", help="Auto-download and resume from latest HF checkpoint")
    parser.add_argument("--save_every_steps", type=int, default=5000, help="Push to hub every N steps mid-epoch")
    parser.add_argument("--use_gdrive", action="store_true", help="Instantly copy checkpoints to mounted Google Drive")
    parser.add_argument("--target_acc", type=float, default=0.55, help="Target directional validation accuracy (e.g. 0.55 for 55%) before stopping")
    
    args = parser.parse_args()
    if args.unlimited:
        args.epochs = 999999999 # effectively infinite
        
    train(args)
