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
from torch.cuda.amp import GradScaler, autocast
from huggingface_hub import HfApi

from src.mission_control.forecast import FinCastModel
from src.cloud.data_loaders import FinCastDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "sao/FinCast-crypto-v1"

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting FinCast Pre-Training on device: {device}")
    
    # 1. Initialize Dataset & DataLoader
    dataset = FinCastDataset(args.data_path, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize Model, Optimizer, and Scaler
    model = FinCastModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=args.fp16)
    
    if args.resume_from_hub:
        logger.info(f"Attempting to resume from Hugging Face Hub ({REPO_ID})...")
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="fincast_checkpoint.pt", token=HF_TOKEN)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"✅ Successfully loaded model and optimizer states from {ckpt_path}")
        except Exception as e:
            logger.warning(f"No existing checkpoint found or error downloading: {e}. Starting from scratch.")
    
    # Autoregressive return prediction (regression task)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    model_dir = Path("checkpoints")
    model_dir.mkdir(exist_ok=True)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for step, (ohlcv_seq, target_returns) in enumerate(loader):
            ohlcv_seq = ohlcv_seq.to(device)
            target_returns = target_returns.to(device)
            
            with autocast(enabled=args.fp16):
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
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
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
    
    args = parser.parse_args()
    if args.unlimited:
        args.epochs = 999999999 # effectively infinite
        
    train(args)
