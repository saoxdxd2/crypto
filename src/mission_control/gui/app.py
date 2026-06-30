import tkinter as tk
from datetime import datetime, UTC
from typing import Any
import subprocess
import threading
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.core.logger import logger

import customtkinter as ctk

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(base_path, relative_path)

from src.mission_control.gui.worker import DataWorker
from src.mission_control.governor import RiskGovernor

# Configure Dark Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ── Colour Palette ──
BG_ROOT    = "#0a0a0f"
PANEL_BG   = "#12121c"
PANEL_BG2  = "#181825"
BORDER     = "#2a2a3a"
CYAN       = "#00f3ff"
MAGENTA    = "#ff00ea"
GREEN      = "#00ff88"
RED        = "#ff3366"
AMBER      = "#ffaa00"
TEXT_MAIN  = "#e2e8f0"
TEXT_MUTED = "#64748b"
MONO_FONT  = "Consolas"

# Multi-coin symbols
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ────────────────────────────────────────────────────────────────
#  Reusable Metric Card
# ────────────────────────────────────────────────────────────────
class MetricCard(ctk.CTkFrame):
    """A small labelled value card used throughout the dashboard."""

    def __init__(self, parent, label: str, initial: str = "--", colour: str = CYAN, **kw):
        super().__init__(parent, fg_color=PANEL_BG2, corner_radius=10, **kw)
        ctk.CTkLabel(
            self, text=label, font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED
        ).pack(anchor="w", padx=12, pady=(10, 0))
        self._value = ctk.CTkLabel(
            self, text=initial,
            font=ctk.CTkFont(family=MONO_FONT, size=22, weight="bold"),
            text_color=colour
        )
        self._value.pack(anchor="w", padx=12, pady=(2, 10))

    def set(self, text: str, colour: str | None = None):
        self._value.configure(text=text)
        if colour:
            self._value.configure(text_color=colour)


# ────────────────────────────────────────────────────────────────
#  Mini Canvas chart (price sparkline)
# ────────────────────────────────────────────────────────────────
class Sparkline(ctk.CTkFrame):
    """A lightweight canvas-based sparkline for edge history."""

    MAX_POINTS = 60  # keep last 60 ticks

    def __init__(self, parent, chart_width=380, chart_height=100, line_colour=CYAN, **kw):
        super().__init__(parent, fg_color=PANEL_BG2, corner_radius=10, **kw)
        self.line_colour = line_colour
        self.chart_w = chart_width
        self.chart_h = chart_height
        self._points: list[float] = []

        self.canvas = tk.Canvas(
            self, width=chart_width, height=chart_height,
            bg=PANEL_BG2, highlightthickness=0, bd=0
        )
        self.canvas.place(x=8, y=8)

    def push(self, value: float):
        self._points.append(value)
        if len(self._points) > self.MAX_POINTS:
            self._points = self._points[-self.MAX_POINTS:]
        self._redraw()

    def set_points(self, values: list[float]):
        self._points = values[-self.MAX_POINTS:]
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        pts = self._points
        if len(pts) < 2:
            return
        lo, hi = min(pts), max(pts)
        spread = hi - lo if hi != lo else 1.0
        pad = 6
        w = self.chart_w - 2 * pad
        h = self.chart_h - 2 * pad

        # Draw zero-line
        if lo <= 0 <= hi:
            zy = pad + h - ((0 - lo) / spread) * h
            self.canvas.create_line(pad, zy, pad + w, zy, fill=TEXT_MUTED, dash=(4, 4))

        coords = []
        for i, v in enumerate(pts):
            x = pad + (i / (len(pts) - 1)) * w
            y = pad + h - ((v - lo) / spread) * h
            coords.extend([x, y])
        self.canvas.create_line(*coords, fill=self.line_colour, width=2, smooth=True)

        # latest dot
        last_x, last_y = coords[-2], coords[-1]
        r = 4
        self.canvas.create_oval(
            last_x - r, last_y - r, last_x + r, last_y + r,
            fill=self.line_colour, outline=""
        )


# ────────────────────────────────────────────────────────────────
#  Main Dashboard
# ────────────────────────────────────────────────────────────────
class CyberQuantDashboard(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Crypto Mission Control")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_ROOT)

        # Risk Governor reference (for manual override)
        self.governor = RiskGovernor()

        # Trade statistics kept in memory
        self._trades: list[dict[str, Any]] = []
        self._thinking: list[dict[str, Any]] = []
        self._active_symbol: str = SYMBOLS[0]

        self._build_ui()

        # Start Data Worker
        self.worker = DataWorker(
            signal_callback=self._on_signal,
            news_callback=self._on_news,
            thresholds_callback=self._on_thresholds,
            trades_callback=self._on_trades,
            thinking_callback=self._on_thinking,
        )
        self.worker.start()
        
        self.cloud_processes = []

    # ──────────── UI Construction ────────────

    def _build_ui(self) -> None:
        # Master weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)

        # ── HEADER ──
        self._build_header()

        # ── LEFT COLUMN (row=1, col=0) ──
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        left.grid_rowconfigure((0, 1, 2), weight=1)
        left.grid_columnconfigure(0, weight=1)

        self._build_ai_panel(left)
        self._build_chart_panel(left)
        self._build_trades_panel(left)

        # ── RIGHT COLUMN (row=1, col=1) ──
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
        right.grid_rowconfigure((0, 1, 2), weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_news_panel(right)
        self._build_cloud_panel(right)
        self._build_settings_panel(right)
        self._build_thinking_panel(right)

    # ── Header ──
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=56)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            hdr, text="⚡ MISSION CONTROL",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_MAIN
        ).grid(row=0, column=0, padx=20, pady=12, sticky="w")

        # Status indicator
        self.lbl_status = ctk.CTkLabel(
            hdr, text="● SYSTEM ONLINE",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=GREEN
        )
        self.lbl_status.grid(row=0, column=1, padx=20, sticky="w")

        # Win/Loss
        self.lbl_wl = ctk.CTkLabel(
            hdr, text="W/L  0 / 0  (--)",
            font=ctk.CTkFont(family=MONO_FONT, size=14, weight="bold"), text_color=AMBER
        )
        self.lbl_wl.grid(row=0, column=2, padx=20)

        self.kill_btn = ctk.CTkButton(
            hdr, text="🛑 KILL SWITCH", width=140,
            fg_color="transparent", border_width=2,
            border_color=RED, text_color=RED,
            hover_color="#330a15",
            command=self._toggle_kill_switch,
        )
        self.kill_btn.grid(row=0, column=3, padx=(10, 8), pady=8)

        # Coin Selector
        self.coin_selector = ctk.CTkComboBox(
            hdr, values=SYMBOLS, width=130,
            font=ctk.CTkFont(family=MONO_FONT, size=13, weight="bold"),
            fg_color=PANEL_BG2, border_color=CYAN, button_color=CYAN,
            dropdown_fg_color=PANEL_BG2, dropdown_text_color=TEXT_MAIN,
            command=self._on_coin_changed,
        )
        self.coin_selector.set(SYMBOLS[0])
        self.coin_selector.grid(row=0, column=4, padx=(0, 20), pady=8)

        # Chat Button
        self.btn_chat = ctk.CTkButton(
            hdr, text="💬 CHAT AI", width=120,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#8a2be2", text_color="white", hover_color="#6a1b9a",
            command=self._open_chat_window
        )
        self.btn_chat.grid(row=0, column=5, padx=(0, 20), pady=8)

    # ── AI Telemetry ──
    def _build_ai_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(
            frame, text="🧠 TimesFM Signal",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=16, pady=(12, 6), sticky="w")

        self.mc_edge   = MetricCard(frame, "NET EDGE", colour=CYAN)
        self.mc_action = MetricCard(frame, "ACTION", colour=TEXT_MAIN)
        self.mc_size   = MetricCard(frame, "SIZE", colour=AMBER)
        self.mc_reason = MetricCard(frame, "REASON", colour=TEXT_MUTED)

        self.mc_edge.grid(  row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.mc_action.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")
        self.mc_size.grid(  row=1, column=2, padx=8, pady=8, sticky="nsew")
        self.mc_reason.grid(row=1, column=3, padx=8, pady=8, sticky="nsew")

    # ── Portfolio Balance Chart ──
    def _build_chart_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=1, column=0, sticky="nsew", pady=8)

        ctk.CTkLabel(
            frame, text="📈 Portfolio Balance ($)",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 4))

        self.sparkline = Sparkline(frame, chart_width=520, chart_height=110, line_colour=GREEN)
        self.sparkline.pack(padx=12, pady=(0, 12), fill="x")

    # ── Recent Trades ──
    def _build_trades_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

        ctk.CTkLabel(
            frame, text="💰 Trade History",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 4))

        self.trades_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family=MONO_FONT, size=12),
            fg_color=PANEL_BG2, text_color=TEXT_MAIN,
            corner_radius=8, height=120, wrap="none",
            state="disabled"
        )
        self.trades_text.pack(padx=12, pady=(0, 12), fill="both", expand=True)

    # ── News / Gemini Risk ──
    def _build_news_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        ctk.CTkLabel(
            frame, text="📰 Qwen Risk Engine",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        self.mc_risk = MetricCard(frame, "RISK MODIFIER", colour=MAGENTA)
        self.mc_risk.pack(padx=12, fill="x")

        self.lbl_news = ctk.CTkLabel(
            frame, text="Waiting for market events...",
            font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
            wraplength=300, justify="left",
        )
        self.lbl_news.pack(anchor="w", padx=16, pady=(8, 12))

    # ── Interactive Settings ──
    def _build_settings_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=1, column=0, sticky="nsew", pady=8)

        ctk.CTkLabel(
            frame, text="⚙️ Risk Settings",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        inputs = ctk.CTkFrame(frame, fg_color="transparent")
        inputs.pack(fill="x", padx=16, pady=(0, 8))
        inputs.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(inputs, text="Max Spread (bps):", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=2)
        self.set_spread = ctk.CTkEntry(inputs, placeholder_text="e.g. 5.0", height=24)
        self.set_spread.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        
        ctk.CTkLabel(inputs, text="Max Daily Loss ($):", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", pady=2)
        self.set_loss = ctk.CTkEntry(inputs, placeholder_text="e.g. 500", height=24)
        self.set_loss.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)

        self.btn_apply_settings = ctk.CTkButton(
            frame, text="APPLY OVERRIDES", font=ctk.CTkFont(weight="bold"),
            fg_color=CYAN, text_color="black", hover_color="#00aacc",
            command=self._apply_manual_settings
        )
        self.btn_apply_settings.pack(fill="x", padx=16, pady=(4, 12))

    # ── Cloud Orchestration ──
    def _build_cloud_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=3, column=0, sticky="nsew", pady=8)

        ctk.CTkLabel(
            frame, text="☁️ Cloud Orchestrator",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        inputs = ctk.CTkFrame(frame, fg_color="transparent")
        inputs.pack(fill="x", padx=16, pady=(0, 8))
        inputs.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(inputs, text="API Token:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=2)
        self.kaggle_token_entry = ctk.CTkEntry(inputs, placeholder_text="KGAT_...", show="*", height=24)
        self.kaggle_token_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        
        # Set default token provided by user
        self.kaggle_token_entry.insert(0, "KGAT_e7e2b99a1d8a2e862044df2e079f86fa")

        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(0, 12))
        
        self.btn_setup_kaggle = ctk.CTkButton(
            btns, text="1. SETUP KAGGLE API", font=ctk.CTkFont(weight="bold"),
            fg_color="#00aacc", text_color="white", hover_color="#0088aa",
            command=self._setup_kaggle
        )
        self.btn_setup_kaggle.pack(side="top", fill="x", pady=(0, 8))

        self.btn_manual_retrain = ctk.CTkButton(
            btns, text="FORCE KAGGLE RETRAIN", font=ctk.CTkFont(weight="bold"),
            fg_color="#8a2be2", text_color="white", hover_color="#6a1b9a",
            command=self._force_kaggle_retrain
        )
        self.btn_manual_retrain.pack(side="top", fill="x", pady=(0, 8))

        bottom_btns = ctk.CTkFrame(btns, fg_color="transparent")
        bottom_btns.pack(fill="x")

        self.btn_start_cloud = ctk.CTkButton(
            bottom_btns, text="2. START PIPELINE", font=ctk.CTkFont(weight="bold"),
            fg_color=GREEN, text_color="black", hover_color="#00cc6a",
            command=self._start_cloud_pipeline
        )
        self.btn_start_cloud.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_stop_cloud = ctk.CTkButton(
            bottom_btns, text="⏹ STOP", font=ctk.CTkFont(weight="bold"),
            fg_color=RED, text_color="white", hover_color="#cc0044",
            state="disabled", command=self._stop_cloud_pipeline
        )
        self.btn_stop_cloud.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(
            frame, height=10, corner_radius=5,
            progress_color=CYAN, fg_color=PANEL_BG2
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 12))

    # ── Thinking Process / Decision Log ──
    def _build_thinking_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=14)
        frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

        ctk.CTkLabel(
            frame, text="🔍 Decision Reasoning",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=16, pady=(12, 4))

        self.thinking_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family=MONO_FONT, size=11),
            fg_color=PANEL_BG2, text_color=TEXT_MUTED,
            corner_radius=8, height=120, wrap="word",
            state="disabled"
        )
        self.thinking_text.pack(padx=12, pady=(0, 12), fill="both", expand=True)

    # ──────────── Callbacks (thread-safe via .after) ────────────

    def _on_signal(self, sig: dict[str, Any]):
        self.after(0, lambda: self._apply_signal(sig))

    def _apply_signal(self, sig: dict[str, Any]):
        edge = sig.get("net_edge", 0)
        action = str(sig.get("action", "--")).upper()
        size = sig.get("size", "--")
        reason = sig.get("reason_code", "--")

        edge_colour = GREEN if float(edge) > 0 else RED
        action_colour = GREEN if action == "OPEN" else TEXT_MUTED

        self.mc_edge.set(f"{float(edge):.6f}", edge_colour)
        self.mc_action.set(action, action_colour)
        self.mc_size.set(str(size))
        self.mc_reason.set(reason)

    def _on_news(self, news: dict[str, Any]):
        self.after(0, lambda: self._apply_news(news))

    def _apply_news(self, news: dict[str, Any]):
        mod = news.get("risk_modifier", "--")
        self.mc_risk.set(f"{mod}x")
        summary = news.get("summary", "")
        self.lbl_news.configure(text=summary or "No recent events.")

    def _on_thresholds(self, t: dict[str, Any]):
        self.after(0, lambda: self._apply_thresholds(t))

    def _apply_thresholds(self, t: dict[str, Any]):
        # Update the UI inputs if FLAML changed them (and not manual)
        if "max_spread_bps" in t:
            self.set_spread.delete(0, "end")
            self.set_spread.insert(0, str(t["max_spread_bps"]))
        if "max_daily_loss_usd" in t:
            self.set_loss.delete(0, "end")
            self.set_loss.insert(0, str(t["max_daily_loss_usd"]))

    def _on_trades(self, trades: list[dict[str, Any]]):
        self.after(0, lambda: self._apply_trades(trades))

    def _apply_trades(self, trades: list[dict[str, Any]]):
        self._trades = trades

        # -- Calculate Cumulative Portfolio Balance --
        balance = 10000.0
        balance_history = [balance]
        for t in trades:
            balance += float(t.get("pnl", 0))
            balance_history.append(balance)
        
        self.sparkline.set_points(balance_history)

        # -- Win / Loss ratio --
        wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl", 0)) < 0)
        total = wins + losses
        ratio = f"{(wins / total * 100):.0f}%" if total else "--"
        self.lbl_wl.configure(text=f"W/L  {wins} / {losses}  ({ratio})")

        # -- Trade log table --
        self.trades_text.configure(state="normal")
        self.trades_text.delete("1.0", "end")

        header = f"{'TIME':<22s} {'SIDE':<6s} {'SIZE':>8s} {'EDGE':>10s} {'PnL':>10s} {'STATUS':<10s}\n"
        sep    = "─" * 70 + "\n"
        self.trades_text.insert("end", header, "header")
        self.trades_text.insert("end", sep)

        for t in reversed(trades[-30:]):  # most recent 30
            pnl = float(t.get("pnl", 0))
            pnl_colour = GREEN if pnl > 0 else RED
            line = (
                f"{t.get('time', ''):<22s} "
                f"{t.get('side', ''):<6s} "
                f"{str(t.get('size', '')):>8s} "
                f"{str(t.get('net_edge', '')):>10s} "
                f"{pnl:>+10.2f} "
                f"{t.get('status', ''):<10s}\n"
            )
            self.trades_text.insert("end", line)

        self.trades_text.configure(state="disabled")

    def _on_thinking(self, entries: list[dict[str, Any]]):
        self.after(0, lambda: self._apply_thinking(entries))

    def _apply_thinking(self, entries: list[dict[str, Any]]):
        self._thinking = entries

        self.thinking_text.configure(state="normal")
        self.thinking_text.delete("1.0", "end")

        for e in entries[-15:]:  # last 15 decisions
            ts = e.get("time", "")
            verdict = e.get("verdict", "")
            reason = e.get("reason", "")
            progress = e.get("progress", None)

            if progress is not None:
                self.progress_bar.set(float(progress))

            colour_tag = "allow" if verdict == "ALLOW" else ("block" if verdict == "BLOCK" else "reduce")
            line = f"[{ts}] {verdict}: {reason}\n"
            self.thinking_text.insert("end", line)

        self.thinking_text.configure(state="disabled")

    # ──────────── Kill Switch ────────────

    def _toggle_kill_switch(self):
        if not self.governor.kill_switch_active:
            self.governor.activate_kill_switch()
            self.kill_btn.configure(
                text="🛑 KILL ACTIVE", fg_color=RED, text_color="white"
            )
            self.lbl_status.configure(text="● KILL SWITCH ACTIVE", text_color=RED)
        else:
            self.governor.kill_switch_active = False
            self.kill_btn.configure(
                text="🛑 KILL SWITCH", fg_color="transparent", text_color=RED
            )
            self.lbl_status.configure(text="● SYSTEM ONLINE", text_color=GREEN)

    # ──────────── Cloud Pipeline Logic ────────────

    def _setup_kaggle(self):
        token = self.kaggle_token_entry.get().strip()
        
        if not token:
            import webbrowser
            logger.info("Opening Kaggle API Settings...")
            webbrowser.open("https://www.kaggle.com/settings")
            # Show a simple dialog
            dialog = ctk.CTkToplevel(self)
            dialog.title("Kaggle Setup Guide")
            dialog.geometry("400x250")
            dialog.attributes("-topmost", True)
            ctk.CTkLabel(dialog, text="Kaggle API Setup", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 10))
            guide_text = "1. Your browser just opened the Kaggle Settings page.\n2. Scroll down to the 'API' section.\n3. Click 'Create New Token'.\n4. Copy the new KGAT_... token here."
            ctk.CTkLabel(dialog, text=guide_text, justify="left").pack(padx=20, pady=10)
            ctk.CTkButton(dialog, text="Got it!", command=dialog.destroy).pack(pady=10)
            return
            
        try:
            from pathlib import Path
            import os
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_token_file = kaggle_dir / "access_token"
            
            kaggle_token_file.write_text(token.strip())
            os.environ["KAGGLE_API_TOKEN"] = token.strip()
            
            # Secure the file
            if sys.platform != "win32":
                os.chmod(kaggle_token_file, 0o600)
                
            self.btn_setup_kaggle.configure(text="✅ KAGGLE READY", fg_color=GREEN, text_color="black")
            logger.info("Kaggle access_token saved securely.")
        except Exception as e:
            logger.error(f"Failed to save Kaggle token: {e}")

    def _force_kaggle_retrain(self):
        """Manually triggers the Kaggle Training Pipeline from the GUI"""
        self.btn_manual_retrain.configure(text="PUSHING TO KAGGLE...", state="disabled")
        logger.info("Manually triggering Kaggle Cloud Training...")
        
        def run_kaggle():
            try:
                # Kaggle trainer has been removed.
                logger.info("Kaggle training has been permanently removed.")
                self.after(0, lambda: self.btn_manual_retrain.configure(text="✅ CLOUD REMOVED", fg_color=GREEN, text_color="black"))
                self.after(3000, lambda: self.btn_manual_retrain.configure(text="FORCE KAGGLE RETRAIN", fg_color="#8a2be2", text_color="white", state="normal"))
            except Exception as e:
                logger.error(f"Kaggle Push Failed: {e}")
                self.after(0, lambda: self.btn_manual_retrain.configure(text="❌ FAILED", fg_color=RED, text_color="white"))
                self.after(3000, lambda: self.btn_manual_retrain.configure(text="FORCE KAGGLE RETRAIN", fg_color="#8a2be2", text_color="white", state="normal"))
                
        import threading
        threading.Thread(target=run_kaggle, daemon=True).start()

    def _start_cloud_pipeline(self):
        self.btn_start_cloud.configure(state="disabled")
        self.btn_stop_cloud.configure(state="normal")
        logger.info("Initializing Local Execution Pipeline...")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # 1. Start Go Orchestrator
        orchestrator_path = resource_path(os.path.join("src", "go_orchestrator", "orchestrator.exe"))
        
        if os.path.exists(orchestrator_path):
            cwd_orch = os.path.dirname(orchestrator_path)
            cmd_orch = [orchestrator_path, "local", "local"]
            p_orch = subprocess.Popen(
                cmd_orch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                cwd=cwd_orch, env=env, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
        else:
            cmd_orch = ["go", "run", "main.go", "local", "local"]
            cwd_orch = os.path.join(os.getcwd(), "src", "go_orchestrator")
            p_orch = subprocess.Popen(
                cmd_orch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                cwd=cwd_orch, env=env, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
        
        self.cloud_processes.append(p_orch)
        logger.info("Go Orchestrator started.")

        # 1.5 Start TLS Proxy (Binance WSS -> Local TCP 9000)
        import multiprocessing
        from src.data.tls_proxy import run_proxy
        p_proxy = multiprocessing.Process(target=run_proxy, daemon=True)
        p_proxy.start()
        self.cloud_processes.append(p_proxy)
        logger.info("TLS Proxy started on port 9000.")

        # 2. Start Native Execution Engine (Zig/C++)
        engine_path = resource_path(os.path.join("src", "execution", "zig-out", "bin", "engine.exe"))
        if not os.path.exists(engine_path):
            engine_path = resource_path(os.path.join("src", "execution", "engine.exe")) # fallback
            
        if os.path.exists(engine_path):
            cwd_engine = os.path.dirname(engine_path)
            p_live = subprocess.Popen(
                [engine_path], 
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=cwd_engine, env=env, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            self.cloud_processes.append(p_live)
            logger.info("Zig/C++ Native Execution Engine started.")
            
            # Start background thread to read JSON IPC
            threading.Thread(target=self._stream_native_ipc, args=(p_live,), daemon=True).start()
        else:
            logger.error(f"Native engine not found at {engine_path}. Please compile it first.")

    def _stream_native_ipc(self, process):
        """Reads JSON stdout from the native execution engine and updates the UI."""
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("type") == "SIGNAL":
                        self._on_signal(data)
                except json.JSONDecodeError:
                    logger.debug(f"[Native] {line.strip()}")
        logger.info("Native engine IPC stream closed.")

    def _stop_cloud_pipeline(self):
        logger.info("Stopping Cloud Pipeline (Force Cleanup)...")
        for p in self.cloud_processes:
            try:
                p.terminate()
                if hasattr(p, 'wait'):
                    p.wait(timeout=1)
                elif hasattr(p, 'join'):
                    p.join(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {p.pid} hung on terminate. Forcing kill.")
                p.kill()
            except Exception as e:
                logger.error(f"Error terminating {p.pid}: {e}")
        self.cloud_processes = []
        self.btn_start_cloud.configure(state="normal")
        self.btn_stop_cloud.configure(state="disabled")

    # ──────────── Interactive Settings ────────────

    def _apply_manual_settings(self):
        try:
            spread = float(self.set_spread.get() or "5.0")
            loss = float(self.set_loss.get() or "500")
            
            # Write to optimal_thresholds.json which is read by runner & DataWorker
            import json
            from pathlib import Path
            path = Path("data/metadata/optimal_thresholds.json")
            
            config = {}
            if path.exists():
                try:
                    config = json.loads(path.read_text(encoding="utf-8"))
                except:
                    pass
                    
            config["max_spread_bps"] = spread
            config["max_daily_loss_usd"] = loss
            config["manual_override"] = True
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            
            from src.core.sys_events import push_sys_event
            push_sys_event("SETTINGS", f"Manual Override: Max Spread={spread}bps, Max Loss=${loss}")
            
            # Also instantly push it to UI
            self._apply_thresholds(config)
            logger.info("Manual settings applied!")
        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")

    # ──────────── Coin Selector ────────────

    def _on_coin_changed(self, symbol: str):
        self._active_symbol = symbol
        self.sparkline._points.clear()
        self.sparkline._redraw()
        logger.info(f"Switched to {symbol}")

    # ──────────── Chat UI ────────────

    def _open_chat_window(self):
        if hasattr(self, "chat_window") and self.chat_window is not None and self.chat_window.winfo_exists():
            self.chat_window.focus()
            return

        self.chat_window = ctk.CTkToplevel(self)
        self.chat_window.title("💬 Qwen3 AI Chat")
        self.chat_window.geometry("500x600")
        self.chat_window.configure(fg_color=PANEL_BG)

        # Chat History
        self.chat_history = ctk.CTkTextbox(
            self.chat_window, font=ctk.CTkFont(family=MONO_FONT, size=12),
            fg_color=PANEL_BG2, text_color=TEXT_MAIN,
            corner_radius=8, wrap="word", state="disabled"
        )
        self.chat_history.pack(padx=16, pady=(16, 8), fill="both", expand=True)

        # Input Frame
        input_frame = ctk.CTkFrame(self.chat_window, fg_color="transparent")
        input_frame.pack(fill="x", padx=16, pady=(0, 16))
        
        self.chat_input = ctk.CTkEntry(
            input_frame, placeholder_text="Ask about market conditions...", 
            height=36, font=ctk.CTkFont(size=13)
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.chat_input.bind("<Return>", lambda e: self._send_chat_message())

        self.btn_send = ctk.CTkButton(
            input_frame, text="SEND", font=ctk.CTkFont(weight="bold"),
            fg_color="#8a2be2", text_color="white", hover_color="#6a1b9a",
            width=80, height=36, command=self._send_chat_message
        )
        self.btn_send.pack(side="right")

        self._append_chat("SYSTEM", "Connecting to Shared LLM Server...", CYAN)
        
        # Check connection
        import threading
        threading.Thread(target=self._check_chat_server, daemon=True).start()

    def _check_chat_server(self):
        import urllib.request
        import time
        max_retries = 30
        for i in range(max_retries):
            try:
                # We just do a dummy request to see if it responds or 404s
                req = urllib.request.Request("http://localhost:5001/", method="GET")
                urllib.request.urlopen(req, timeout=2)
                # If no exception, server is up but returned 200 (unexpected but okay)
                self.after(0, lambda: self._append_chat("SYSTEM", "Connected to Shared LLM!", GREEN))
                self.chat_ready = True
                return
            except Exception as e:
                # 404 means server is up (since only /generate is handled)
                if hasattr(e, 'code') and e.code == 404:
                    self.after(0, lambda: self._append_chat("SYSTEM", "Connected to Shared LLM!", GREEN))
                    self.chat_ready = True
                    return
                # Otherwise, it's probably booting up. Sleep and retry.
                time.sleep(1)
        
        # If we exhausted retries
        self.after(0, lambda: self._append_chat("SYSTEM", f"Warning: LLM Server not reachable after {max_retries}s. Is it still loading?", RED))
        self.chat_ready = False

    def _append_chat(self, sender: str, msg: str, color: str = TEXT_MAIN):
        if not hasattr(self, "chat_history") or not self.chat_history.winfo_exists():
            return
        
        self.chat_history.configure(state="normal")
        self.chat_history.tag_config(sender, foreground=color)
        self.chat_history.insert("end", f"{sender}: ", sender)
        self.chat_history.insert("end", f"{msg}\n\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def _send_chat_message(self):
        msg = self.chat_input.get().strip()
        if not msg:
            return
        
        self.chat_input.delete(0, "end")
        self._append_chat("YOU", msg, CYAN)

        if not getattr(self, "chat_ready", False):
            self._append_chat("SYSTEM", "LLM Server is offline.", RED)
            return

        self.btn_send.configure(state="disabled")
        import threading
        threading.Thread(target=self._generate_chat_response, args=(msg,), daemon=True).start()

    def _generate_chat_response(self, user_msg: str):
        prompt = f"<|im_start|>system\nYou are a helpful AI trading assistant. Keep responses brief and analytical.<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            import urllib.request
            import json
            data = json.dumps({"prompt": prompt, "max_tokens": 200, "temperature": 0.3}).encode('utf-8')
            req = urllib.request.Request("http://localhost:5001/generate", data=data, headers={'Content-Type': 'application/json'}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result.get("response", "Error: No response")
            self.after(0, lambda: self._append_chat("QWEN3", text, AMBER))
        except Exception as e:
            self.after(0, lambda: self._append_chat("SYSTEM", f"Error generating response: {e}", RED))
        finally:
            self.after(0, lambda: self.btn_send.configure(state="normal"))

    # ──────────── Cleanup ────────────

    def destroy(self):
        self._stop_cloud_pipeline()
        self.worker.stop()
        super().destroy()


if __name__ == "__main__":
    app = CyberQuantDashboard()
    app.mainloop()
