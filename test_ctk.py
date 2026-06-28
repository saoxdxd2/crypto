import customtkinter as ctk
import tkinter as tk

PANEL_BG2  = "#181825"
CYAN       = "#00f3ff"

class Sparkline(ctk.CTkFrame):
    def __init__(self, parent, chart_width=380, chart_height=100, line_colour=CYAN, **kw):
        super().__init__(parent, fg_color=PANEL_BG2, corner_radius=10, **kw)
        self.line_colour = line_colour
        self._w = chart_width
        self._h = chart_height

        self.canvas = tk.Canvas(
            self, width=chart_width, height=chart_height,
            bg=PANEL_BG2, highlightthickness=0, bd=0
        )
        self.canvas.place(x=8, y=8)

app = ctk.CTk()
spark = Sparkline(app, chart_width=520, chart_height=110, line_colour=CYAN)
spark.pack()
app.mainloop()
