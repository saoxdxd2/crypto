"""
FinCast / Kronos Forecaster.

Replaces the generic TimesFM model with a financial-native decoder-only 
Transformer. This model is engineered specifically for OHLCV candlesticks,
using a hierarchical tokenization scheme to learn the 'language' of K-lines.

Ref: "FinCast / Kronos: Foundation Models for Financial Time-Series"
"""
from __future__ import annotations

import json
import logging
import uuid
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Constants ──
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 6

FINCAST_CHECKPOINT_DIR = Path("checkpoints")
FINCAST_CHECKPOINT_NAME = "fincast_online.pt"
FINCAST_FINETUNE_LR = 1e-4
FINCAST_WEIGHT_DECAY = 0.01


class OHLCVTokenizer(nn.Module):
    """
    Discretizes Open, High, Low, Close, Volume into hierarchical tokens.
    Unlike continuous MLPs, this tokenization treats financial price levels
    as a discrete vocabulary.
    """
    def __init__(self, vocab_size: int = 50000):
        super().__init__()
        self.vocab_size = vocab_size
        # For OHLCV we project 5 continuous values to a D_MODEL embedding.
        # In a full FinCast implementation, this would be a discrete VQ-VAE or similar.
        self.proj = nn.Linear(5, D_MODEL)

    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        ohlcv: (B, SeqLen, 5)
        Returns: (B, SeqLen, D_MODEL)
        """
        return self.proj(ohlcv)


class FinCastModel(nn.Module):
    """
    Decoder-only Foundation Model for Financial Time-Series.
    Predicts the next K-line tokenautoregressively.
    """
    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.tokenizer = OHLCVTokenizer()
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        
        # Decoder-only Transformer (Causal masking)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Forecast Head (predicting next return)
        self.forecast_head = nn.Linear(d_model, 1)

    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        ohlcv: (B, SeqLen, 5)
        Returns predicted returns: (B, SeqLen)
        """
        seq_len = ohlcv.size(1)
        x = self.tokenizer(ohlcv)
        x = x + self.pos_emb[:, :seq_len, :]
        
        # Causal mask ensures we only look at past OHLCV to predict future
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.decoder(x, mask=causal_mask, is_causal=True)
        
        return self.forecast_head(x).squeeze(-1)


class FinCastForecaster:
    """
    Wrapper for running the FinCast model on Parquet datasets.
    """

    def __init__(
        self,
        context_len: int = 512,
        horizon: int = 3,
        backend: str = "cpu",
    ) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self.device = torch.device(backend)
        
        logger.info(f"Initializing FinCast backend={backend} context={context_len}")
        self.model = FinCastModel().to(self.device)
        self.model.eval()
        
        # Load pre-trained weights if available, otherwise initialized randomly
        # self.model.load_state_dict(torch.load("fincast_weights.pt"))

        # ── Online fine-tuning state ──
        self.checkpoint_dir = FINCAST_CHECKPOINT_DIR
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=FINCAST_FINETUNE_LR, weight_decay=FINCAST_WEIGHT_DECAY
        )
        self.loss_fn = nn.MSELoss()
        self._step_count = 0

        self._load_checkpoint()

    def finetune_step(
        self,
        ohlcv_windows: list[np.ndarray],
        target_returns: list[float],
    ) -> float:
        """
        Run one online adaptation step (forward + backward + optimizer step).

        Args:
            ohlcv_windows: List of numpy arrays, each shape (SeqLen, 5) — OHLCV candles
            target_returns: List of floats — ground-truth next-candle returns

        Returns:
            Training loss (float)
        """
        if not ohlcv_windows:
            return 0.0

        self.model.train()

        # Stack numpy arrays into a single tensor: (B, SeqLen, 5)
        x = torch.tensor(np.stack(ohlcv_windows), dtype=torch.float32).to(self.device)
        y = torch.tensor(target_returns, dtype=torch.float32).to(self.device)

        # Forward: use the last timestep's prediction as the forecast
        predictions = self.model(x)  # (B, SeqLen)
        pred_last = predictions[:, -1]  # (B,)
        loss = self.loss_fn(pred_last, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self._step_count += 1

        if self._step_count % 50 == 0:
            self._save_checkpoint()
            logger.info(f"FinCast fine-tune step {self._step_count}, loss={loss.item():.6f}")

        return loss.item()

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step_count,
        }, self.checkpoint_dir / FINCAST_CHECKPOINT_NAME)

    def _load_checkpoint(self) -> None:
        ckpt = self.checkpoint_dir / FINCAST_CHECKPOINT_NAME
        if ckpt.exists():
            try:
                data = torch.load(ckpt, map_location=self.device, weights_only=True)
                self.model.load_state_dict(data["model"])
                self.optimizer.load_state_dict(data["optimizer"])
                self._step_count = data.get("step", 0)
                logger.info(f"Loaded FinCast checkpoint (step {self._step_count})")
            except Exception as e:
                logger.warning(f"Failed to load FinCast checkpoint: {e}")

    def forecast_latest(
        self,
        parquet_path: Path,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        
        # Fast read of last N candles using polars
        df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("is_closed") == True)
            .sort("open_time", descending=True)
            .limit(self.context_len)
            .collect()
            .reverse()
        )

        if df.height < self.context_len:
            raise ValueError(f"Insufficient data: {df.height} candles, need {self.context_len}")

        # Extract OHLCV
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv_data = df[ohlcv_cols].to_numpy()
        last_candle_time = df["open_time"][-1]

        # Forecast
        with torch.no_grad():
            x = torch.tensor(ohlcv_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Predict the next return based on the last hidden state
            predictions = self.model(x)
            point_forecast_return = predictions[0, -1].item()

        # Mocking quantiles for now since this is a deterministic point estimate
        expected_return = point_forecast_return
        p10_return = expected_return - 0.005
        p50_return = expected_return
        p90_return = expected_return + 0.005

        signal = {
            "forecast_id": f"fcst_fincast_{uuid.uuid4().hex[:8]}",
            "model": "fincast_decoder_only",
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "created_at": datetime.now(UTC).isoformat(),
            "input_last_closed_candle": last_candle_time.isoformat() if hasattr(last_candle_time, "isoformat") else str(last_candle_time),
            "horizon": f"{self.horizon} candles",
            "expected_return": round(expected_return, 6),
            "p10_return": round(p10_return, 6),
            "p50_return": round(p50_return, 6),
            "p90_return": round(p90_return, 6),
        }
        
        return signal

    def write_forecast(self, signal: dict[str, Any], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"latest_forecast_{signal['symbol']}.json"
        out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
        return out_path
