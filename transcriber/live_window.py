from __future__ import annotations

import queue
from typing import Any

from transcriber.live_wlk import CaptionState


def format_caption_text(state: CaptionState, *, max_committed_lines: int = 6) -> str:
    lines = list(state.committed_lines[-max(1, max_committed_lines) :])
    if state.partial_line:
        lines.append(state.partial_line)
    return "\n".join(lines)


class CaptionWindow:
    def __init__(
        self,
        *,
        state_queue: queue.Queue[CaptionState],
        stop_event: Any,
        max_committed_lines: int = 6,
    ) -> None:
        self.state_queue = state_queue
        self.stop_event = stop_event
        self.max_committed_lines = max_committed_lines
        self.state = CaptionState(committed_lines=(), partial_line="")

    def run(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Live Captions")
        root.geometry("820x240")
        root.minsize(420, 120)
        root.configure(background="#111111")
        root.attributes("-topmost", True)

        style = ttk.Style(root)
        style.configure("Caption.TFrame", background="#111111")
        style.configure(
            "Caption.TLabel",
            background="#111111",
            foreground="#f5f5f5",
            font=("Segoe UI", 18),
            justify="left",
        )

        frame = ttk.Frame(root, style="Caption.TFrame", padding=16)
        frame.pack(fill="both", expand=True)
        label = ttk.Label(frame, text="", style="Caption.TLabel", anchor="sw", justify="left")
        label.pack(fill="both", expand=True)

        def close() -> None:
            self.stop_event.set()
            root.destroy()

        def poll() -> None:
            try:
                while True:
                    self.state = self.state_queue.get_nowait()
            except queue.Empty:
                pass
            label.configure(text=format_caption_text(self.state, max_committed_lines=self.max_committed_lines))
            if self.stop_event.is_set():
                root.destroy()
                return
            root.after(50, poll)

        root.protocol("WM_DELETE_WINDOW", close)
        poll()
        root.mainloop()
