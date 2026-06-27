import logging
import customtkinter as ctk
from typing import Any

from crypto_research.gui.worker import DataWorker
from crypto_research.governor import RiskGovernor

logger = logging.getLogger(__name__)

# Configure Dark Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Neon Colors for UI
CYAN = "#00f3ff"
MAGENTA = "#ff00ea"
GREEN = "#00ff88"
RED = "#ff3366"
PANEL_BG = "#1a1a24"

class CyberQuantDashboard(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Crypto Mission Control")
        self.geometry("1000x700")
        self.configure(fg_color="#0a0a0f")
        
        # Risk Governor reference (for manual override)
        self.governor = RiskGovernor()

        self._build_ui()
        
        # Start Data Worker
        self.worker = DataWorker(
            signal_callback=self.update_signal_ui,
            news_callback=self.update_news_ui,
            thresholds_callback=self.update_thresholds_ui
        )
        self.worker.start()

    def _build_ui(self) -> None:
        # Layout weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure((0, 1, 2), weight=1)

        # ---- HEADER ----
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10), sticky="ew")
        
        title = ctk.CTkLabel(header_frame, text="MISSION CONTROL", font=ctk.CTkFont(family="Inter", size=24, weight="bold"))
        title.pack(side="left")

        self.kill_btn = ctk.CTkButton(
            header_frame, 
            text="ACTIVATE KILL SWITCH", 
            fg_color="transparent", 
            border_width=2, 
            border_color=RED, 
            text_color=RED,
            hover_color="#330a15",
            command=self.toggle_kill_switch
        )
        self.kill_btn.pack(side="right")

        # ---- AI PLANE (TIMESFM) ----
        ai_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=15)
        ai_frame.grid(row=1, column=0, columnspan=2, padx=(20, 10), pady=10, sticky="nsew")
        
        ctk.CTkLabel(ai_frame, text="🧠 TimesFM Telemetry", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=15)
        
        self.lbl_edge = ctk.CTkLabel(ai_frame, text="LATEST EDGE:\n--", font=ctk.CTkFont(size=16), text_color=CYAN)
        self.lbl_edge.pack(anchor="w", padx=20, pady=10)
        
        self.lbl_action = ctk.CTkLabel(ai_frame, text="ACTION:\n--", font=ctk.CTkFont(size=16))
        self.lbl_action.pack(anchor="w", padx=20, pady=10)
        
        self.lbl_size = ctk.CTkLabel(ai_frame, text="TARGET SIZE:\n--", font=ctk.CTkFont(size=16))
        self.lbl_size.pack(anchor="w", padx=20, pady=10)

        # ---- NEWS & RISK ENGINE (GEMINI) ----
        news_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=15)
        news_frame.grid(row=1, column=2, padx=(10, 20), pady=10, sticky="nsew")
        
        ctk.CTkLabel(news_frame, text="📰 Gemini Risk Engine", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=15)
        
        self.lbl_risk = ctk.CTkLabel(news_frame, text="RISK MODIFIER:\n--", font=ctk.CTkFont(size=16), text_color=MAGENTA)
        self.lbl_risk.pack(anchor="w", padx=20, pady=10)
        
        self.lbl_news = ctk.CTkLabel(
            news_frame, 
            text="Waiting for market events...", 
            font=ctk.CTkFont(size=13), 
            text_color="#94a3b8",
            wraplength=250,
            justify="left"
        )
        self.lbl_news.pack(anchor="w", padx=20, pady=10)

        # ---- FLAML OPTIMIZER ----
        flaml_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=15)
        flaml_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=(10, 20), sticky="ew")
        
        ctk.CTkLabel(flaml_frame, text="⚙️ FLAML Thresholds", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=15)
        
        self.lbl_thresholds = ctk.CTkLabel(flaml_frame, text="Polling for optimal constraints...", font=ctk.CTkFont(family="Courier", size=14), justify="left")
        self.lbl_thresholds.pack(anchor="w", padx=20, pady=10)

    # --- Callbacks ---

    def toggle_kill_switch(self) -> None:
        if not self.governor.kill_switch_active:
            self.governor.activate_kill_switch()
            self.kill_btn.configure(text="KILL SWITCH ACTIVE", fg_color=RED, text_color="white")
        else:
            self.governor.kill_switch_active = False
            self.kill_btn.configure(text="ACTIVATE KILL SWITCH", fg_color="transparent", text_color=RED)

    def update_signal_ui(self, signal: dict[str, Any]) -> None:
        # Use .after to thread-safely update UI
        self.after(0, lambda: self._apply_signal(signal))

    def _apply_signal(self, signal: dict[str, Any]) -> None:
        self.lbl_edge.configure(text=f"LATEST EDGE:\n{signal.get('net_edge', '--')}")
        self.lbl_action.configure(text=f"ACTION:\n{signal.get('action', '--').upper()}")
        self.lbl_size.configure(text=f"TARGET SIZE:\n{signal.get('size', '--')}")

    def update_news_ui(self, news: dict[str, Any]) -> None:
        self.after(0, lambda: self._apply_news(news))

    def _apply_news(self, news: dict[str, Any]) -> None:
        self.lbl_risk.configure(text=f"RISK MODIFIER:\n{news.get('risk_modifier', '--')}x")
        self.lbl_news.configure(text=f"{news.get('summary', '')}")

    def update_thresholds_ui(self, thresholds: dict[str, Any]) -> None:
        self.after(0, lambda: self._apply_thresholds(thresholds))

    def _apply_thresholds(self, thresholds: dict[str, Any]) -> None:
        fmt = "\n".join([f"{k}: {v}" for k, v in thresholds.items()])
        self.lbl_thresholds.configure(text=fmt)

    def destroy(self):
        self.worker.stop()
        super().destroy()

if __name__ == "__main__":
    app = CyberQuantDashboard()
    app.mainloop()
