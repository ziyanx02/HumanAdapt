
import time
import numpy as np

import genesis as gs
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, List, Callable

class GUI:
    def __init__(self, root: tk.Tk, cfg: dict, values: List[float], 
                 save_callback: Callable = None, reset_callback: Callable = None):
        self.root = root
        self.root.title("Robot Controller")

        self.save_callback = save_callback
        self.reset_callback = reset_callback

        self.label_font = ("Helvetica", 15)
        self.entry_font = ("Arial", 15)

        self.cfg = cfg
        self.labels = self.cfg["label"]
        self.init_values = values.copy()
        self.values = values
        self.sliders = []
        self.entries = []

        # Create all widgets before setting up any bindings
        self.create_scrollable_area()
        self.create_widgets()
        # Set up bindings after all widgets are created
        self.setup_bindings()

        # Calculate content height after widgets are created
        self.root.update_idletasks()  # Force GUI layout calculations
        content_height = self.scrollable_frame.winfo_reqheight()  # Get content height
        screen_height = self.root.winfo_screenheight()  # Get max screen height
        
        # Set window height to content height (capped at 90% of screen height)
        window_height = min(content_height, int(screen_height * 0.9))
        self.root.geometry(f"1000x{window_height}")  # Fixed width, dynamic height
        self.root.resizable(False, False)  # Disable window resizing

    def create_scrollable_area(self):
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows/Linux
        self.canvas.bind("<Button-4>", self.on_mousewheel)   # Linux (scroll up)
        self.canvas.bind("<Button-5>", self.on_mousewheel)   # Linux (scroll down)

    def create_widgets(self):
        for i, name in enumerate(self.labels):
            # Get the min and max limits for the slider
            min_limit, max_limit = self.cfg["range"][name][:2]
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(pady=5, padx=10, fill=tk.X)

            # Label for the control
            tk.Label(frame, text=f"{name}", font=self.label_font, width=max([len(name) for name in self.labels]) + 5).pack(side=tk.LEFT)

            # Slider
            slider = ttk.Scale(
                frame,
                from_=float(min_limit),
                to=float(max_limit),
                orient=tk.HORIZONTAL,
                length=300
            )
            slider.pack(side=tk.LEFT, padx=10)
            slider.set(self.values[i])
            self.sliders.append(slider)

            # Entry field with modified validation to allow negative numbers
            vcmd = (self.root.register(self.validate_entry), '%P')
            entry = tk.Entry(
                frame,
                width=8,
                validate='key',
                font=self.entry_font,
                validatecommand=vcmd
            )
            entry.insert(0, f"{self.values[i]:.2f}")
            entry.pack(side=tk.LEFT, padx=5)
            self.entries.append(entry)

        if self.save_callback is not None or self.reset_callback is not None:
            button_frame = tk.Frame(self.scrollable_frame)
            button_frame.pack(pady=10, padx=10, fill=tk.X)

        if self.save_callback is not None:
            tk.Button(button_frame, text="Save", font=("Arial", 12), 
                     command=self.save, width=10).pack(side=tk.RIGHT, pady=10)

        if self.reset_callback is not None:
            tk.Button(button_frame, text="Reset", font=("Arial", 12), 
                     command=self.reset, width=10).pack(side=tk.RIGHT, pady=10)

    def setup_bindings(self):
        # Set up bindings after all widgets are created
        for i, slider in enumerate(self.sliders):
            slider.configure(command=lambda val, idx=i: self.update_from_slider(idx, val))
            self.entries[i].bind('<Return>', lambda e, idx=i: self.update_from_entry(idx))
            self.entries[i].bind('<FocusOut>', lambda e, idx=i: self.update_from_entry(idx))

    def validate_entry(self, value):
        if value == "" or value == "-":  # Allow empty string and minus sign
            return True
        try:
            float(value)  # Try converting to float to validate
            return True
        except ValueError:
            return False

    def update_from_slider(self, idx: int, val: str):
        """Update values when slider is moved"""
        value = float(val)
        self.values[idx] = value
        self.entries[idx].delete(0, tk.END)
        self.entries[idx].insert(0, f"{value:.2f}")

    def update_from_entry(self, idx: int):
        """Update values when entry is changed"""
        try:
            value = float(self.entries[idx].get())
            min_limit, max_limit = self.cfg["range"][self.labels[idx]][:2]
            
            # Clamp value to slider range
            value = max(min_limit, min(value, max_limit))
            
            self.values[idx] = value
            self.sliders[idx].set(value)
            self.entries[idx].delete(0, tk.END)
            self.entries[idx].insert(0, f"{value:.2f}")
        except ValueError:
            # Restore previous value if entry is invalid
            self.entries[idx].delete(0, tk.END)
            self.entries[idx].insert(0, f"{self.values[idx]:.2f}")

    def save(self):
        self.save_callback()

    def reset(self):
        for i, value in enumerate(self.init_values):
            self.values[i] = value
            self.sliders[i].set(value)
            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, f"{value:.2f}")
        if self.reset_callback is not None:
            self.reset_callback()

    def on_mousewheel(self, event):
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/macOS
            self.canvas.yview_scroll(-1 * (event.delta // 60), "units")  # Adjust speed


def start_gui(cfg: Dict, values: List[float], 
              save_callback: Callable = None, 
              reset_callback: Callable = None) -> threading.Event:
    """
    Start GUI in a separate thread
    Returns a threading.Event that is set when the GUI is closed
    """
    stop_event = threading.Event()
    is_gui_closed = [False]

    def close(root):
        def on_close():
            is_gui_closed[0] = True
            stop_event.set()
            root.destroy()
        return on_close

    def start_gui():
        root = tk.Tk()
        app = GUI(root, cfg, values, 
                 save_callback=save_callback, 
                 reset_callback=reset_callback)
        root.protocol("WM_DELETE_WINDOW", close(root))
        root.mainloop()

    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()
    
    return stop_event

if __name__ == "__main__":
    values = [0, 0, 0]
    start_gui(
        cfg={"label": ["joint1", "joint2", "joint3"], "range": {"joint1": [-1, 1], "joint2": [-1, 1], "joint3": [-1, 1]}},
        values=values,
        save_callback=lambda x: print(x),
        reset_callback=lambda: print("reset")
    )
    while True:
        print(values)
        time.sleep(0.5)
