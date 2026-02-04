"""
Chat window GUI — separate window for chat interface.

Uses tkinter for a clean, color-coded chat window.
Terminal remains available for logging output.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext, font as tkfont
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.service.stt.engine import STTEngine
    from src.db.db_manager import SessionInfo


class SessionDialog:
    """Dialog for selecting a session to continue or creating a new one."""

    # Color scheme — dark mode (matches ChatWindow)
    BG_COLOR = "#0d0d0d"
    TEXT_BG = "#1a1a1a"
    TEXT_COLOR = "#e0e0e0"
    DIM_COLOR = "#666666"
    ACCENT_COLOR = "#4fc3f7"
    BUTTON_BG = "#333333"
    BUTTON_FG = "#ffffff"
    HOVER_BG = "#2a2a2a"

    def __init__(self, sessions: list[SessionInfo]):
        self.sessions = sessions
        self.selected_session_id: int | None = None
        self._result: str = "new"  # "new" or "continue"

        self.root = tk.Tk()
        self.root.title("Project Rin — Session")
        self.root.geometry("500x400")
        self.root.configure(bg=self.BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_ui()
        self.root.grab_set()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        # Header
        header = tk.Label(
            self.root,
            text="Select Session",
            font=("Consolas", 16, "bold"),
            fg=self.ACCENT_COLOR,
            bg=self.BG_COLOR,
            pady=15,
        )
        header.pack()

        # New session button
        new_btn = tk.Button(
            self.root,
            text="+ New Session",
            font=("Consolas", 11, "bold"),
            bg=self.ACCENT_COLOR,
            fg="#000000",
            activebackground="#3aa3d7",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._new_session,
        )
        new_btn.pack(pady=(0, 15))

        # Separator
        sep_label = tk.Label(
            self.root,
            text="— or continue previous session —",
            font=("Consolas", 9),
            fg=self.DIM_COLOR,
            bg=self.BG_COLOR,
        )
        sep_label.pack(pady=(0, 10))

        # Session list frame
        list_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=20)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollable canvas for session list
        canvas = tk.Canvas(list_frame, bg=self.TEXT_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.session_frame = tk.Frame(canvas, bg=self.TEXT_BG)

        self.session_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.session_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate sessions
        if not self.sessions:
            no_sessions = tk.Label(
                self.session_frame,
                text="No previous sessions",
                font=("Consolas", 10),
                fg=self.DIM_COLOR,
                bg=self.TEXT_BG,
                pady=20,
            )
            no_sessions.pack()
        else:
            for session in self.sessions:
                self._add_session_item(session)

    def _add_session_item(self, session: SessionInfo) -> None:
        """Add a session item to the list."""
        frame = tk.Frame(self.session_frame, bg=self.TEXT_BG, pady=5, padx=10)
        frame.pack(fill=tk.X, pady=2)

        # Format time
        created = datetime.fromtimestamp(session["created_at"])
        time_str = created.strftime("%Y-%m-%d %H:%M")

        # Session info
        name_label = tk.Label(
            frame,
            text=session["name"],
            font=("Consolas", 10, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.TEXT_BG,
            anchor="w",
        )
        name_label.pack(fill=tk.X)

        info_text = f"{session['message_count']} messages"
        info_label = tk.Label(
            frame,
            text=info_text,
            font=("Consolas", 9),
            fg=self.DIM_COLOR,
            bg=self.TEXT_BG,
            anchor="w",
        )
        info_label.pack(fill=tk.X)

        # Make the whole frame clickable
        for widget in [frame, name_label, info_label]:
            widget.bind("<Enter>", lambda e, f=frame: f.configure(bg=self.HOVER_BG))
            widget.bind("<Leave>", lambda e, f=frame: f.configure(bg=self.TEXT_BG))
            widget.bind("<Button-1>", lambda e, s=session: self._select_session(s))

        # Update child backgrounds on hover
        for widget in [frame, name_label, info_label]:
            widget.bind("<Enter>", lambda e, f=frame, n=name_label, i=info_label: (
                f.configure(bg=self.HOVER_BG),
                n.configure(bg=self.HOVER_BG),
                i.configure(bg=self.HOVER_BG),
            ))
            widget.bind("<Leave>", lambda e, f=frame, n=name_label, i=info_label: (
                f.configure(bg=self.TEXT_BG),
                n.configure(bg=self.TEXT_BG),
                i.configure(bg=self.TEXT_BG),
            ))

    def _select_session(self, session: SessionInfo) -> None:
        """Handle session selection."""
        self.selected_session_id = session["id"]
        self._result = "continue"
        self.root.quit()
        self.root.destroy()

    def _new_session(self) -> None:
        """Handle new session button."""
        self._result = "new"
        self.root.quit()
        self.root.destroy()

    def _on_close(self) -> None:
        """Handle window close — defaults to new session."""
        self._result = "new"
        self.root.quit()
        self.root.destroy()

    def run(self) -> tuple[str, int | None]:
        """Run the dialog and return (action, session_id)."""
        self.root.mainloop()
        return self._result, self.selected_session_id


class ChatWindow:
    """Standalone chat window with async integration."""

    # Color scheme — dark mode
    BG_COLOR = "#0d0d0d"
    TEXT_BG = "#1a1a1a"
    USER_COLOR = "#4fc3f7"       # Light blue for user
    AI_COLOR = "#81c784"         # Soft green for AI name
    TEXT_COLOR = "#e0e0e0"       # Light gray for text
    DIM_COLOR = "#666666"        # Dimmed gray
    INPUT_BG = "#252525"
    BUTTON_BG = "#333333"
    BUTTON_FG = "#ffffff"
    ACCENT_COLOR = "#4fc3f7"     # Accent for highlights

    def __init__(self, on_submit: Callable[[str], None], ai_name: str = "Rin"):
        self.on_submit = on_submit
        self.ai_name = ai_name
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._closed = False

        # Create window in main thread
        self.root = tk.Tk()
        self.root.title(f"Project Rin — Chat")
        self.root.geometry("700x600")
        self.root.configure(bg=self.BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set window icon (optional, won't crash if missing)
        try:
            self.root.iconbitmap(default="")
        except tk.TclError:
            pass

        self._setup_fonts()
        self._setup_ui()

    def _setup_fonts(self) -> None:
        """Configure fonts."""
        self.chat_font = tkfont.Font(family="Consolas", size=11)
        self.input_font = tkfont.Font(family="Consolas", size=11)
        self.header_font = tkfont.Font(family="Consolas", size=14, weight="bold")

    def _setup_ui(self) -> None:
        """Build the UI components."""
        # Header
        header_frame = tk.Frame(self.root, bg=self.BG_COLOR, pady=10)
        header_frame.pack(fill=tk.X)

        header_label = tk.Label(
            header_frame,
            text=f"✨ Project {self.ai_name} ✨",
            font=self.header_font,
            fg=self.AI_COLOR,
            bg=self.BG_COLOR,
        )
        header_label.pack()

        # Status label
        self.status_label = tk.Label(
            header_frame,
            text="Type your message and press Enter",
            font=("Consolas", 9),
            fg=self.DIM_COLOR,
            bg=self.BG_COLOR,
        )
        self.status_label.pack()

        # Chat display area
        chat_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=10, pady=5)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=self.chat_font,
            bg=self.TEXT_BG,
            fg=self.TEXT_COLOR,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED,
            cursor="arrow",
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for colors
        self.chat_display.tag_configure("user", foreground=self.USER_COLOR)
        self.chat_display.tag_configure("user_name", foreground=self.USER_COLOR, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("ai", foreground=self.TEXT_COLOR)
        self.chat_display.tag_configure("ai_name", foreground=self.AI_COLOR, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("dim", foreground=self.DIM_COLOR)
        self.chat_display.tag_configure("error", foreground="#ff6b6b")

        # Input area
        input_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=10, pady=10)
        input_frame.pack(fill=tk.X)

        self.input_field = tk.Entry(
            input_frame,
            font=self.input_font,
            bg=self.INPUT_BG,
            fg=self.TEXT_COLOR,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=self.ACCENT_COLOR,
            highlightbackground=self.DIM_COLOR,
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.input_field.bind("<Return>", self._handle_submit)
        self.input_field.focus_set()

        send_button = tk.Button(
            input_frame,
            text="Send",
            font=("Consolas", 10, "bold"),
            bg=self.BUTTON_BG,
            fg=self.BUTTON_FG,
            activebackground="#444444",
            activeforeground=self.BUTTON_FG,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._submit_from_button,
        )
        send_button.pack(side=tk.RIGHT, padx=(10, 0))

    def _handle_submit(self, event: tk.Event = None) -> None:
        """Handle Enter key press."""
        text = self.input_field.get().strip()
        if text:
            self.input_field.delete(0, tk.END)
            self._input_queue.put(text)
            self.on_submit(text)

    def _submit_from_button(self) -> None:
        """Handle button click."""
        self._handle_submit()

    def _on_close(self) -> None:
        """Handle window close."""
        self._closed = True
        self._input_queue.put("")  # Unblock any waiting get()
        self.root.quit()
        self.root.destroy()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def set_status(self, text: str) -> None:
        """Update status label."""
        if not self._closed:
            self.status_label.config(text=text)

    def append_user_message(self, text: str) -> None:
        """Add a user message to the chat."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "You: ", "user_name")
        self.chat_display.insert(tk.END, f"{text}\n\n", "user")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def start_ai_message(self) -> None:
        """Start an AI message (shows name prefix)."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{self.ai_name}: ", "ai_name")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_ai_chunk(self, chunk: str) -> None:
        """Append a chunk to the current AI message (streaming)."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, chunk, "ai")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def end_ai_message(self) -> None:
        """End the current AI message."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def show_error(self, message: str) -> None:
        """Display an error message."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[ERROR] {message}\n\n", "error")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def show_info(self, message: str) -> None:
        """Display an info message."""
        if self._closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{message}\n", "dim")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def get_input(self) -> str:
        """Get next user input (blocking)."""
        return self._input_queue.get()

    def update(self) -> None:
        """Process pending GUI events (call from async loop)."""
        if not self._closed:
            try:
                self.root.update()
            except tk.TclError:
                self._closed = True

    def run_mainloop(self) -> None:
        """Run the tkinter mainloop (blocking)."""
        self.root.mainloop()


class AsyncChatWindow:
    """Async wrapper for ChatWindow that integrates with asyncio."""

    def __init__(self, ai_name: str = "Rin"):
        self.ai_name = ai_name
        self._window: ChatWindow | None = None
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending_inputs: list[str] = []

    def _on_submit(self, text: str) -> None:
        """Callback when user submits text."""
        self._pending_inputs.append(text)

    async def start(self) -> None:
        """Create and show the window."""
        self._window = ChatWindow(self._on_submit, self.ai_name)

    @property
    def is_closed(self) -> bool:
        return self._window is None or self._window.is_closed

    def set_status(self, text: str) -> None:
        if self._window:
            self._window.set_status(text)

    def append_user_message(self, text: str) -> None:
        if self._window:
            self._window.append_user_message(text)

    def start_ai_message(self) -> None:
        if self._window:
            self._window.start_ai_message()

    def append_ai_chunk(self, chunk: str) -> None:
        if self._window:
            self._window.append_ai_chunk(chunk)

    def end_ai_message(self) -> None:
        if self._window:
            self._window.end_ai_message()

    def show_error(self, message: str) -> None:
        if self._window:
            self._window.show_error(message)

    def show_info(self, message: str) -> None:
        if self._window:
            self._window.show_info(message)

    async def get_input(self, stt: STTEngine | None = None) -> str:
        """
        Get user input asynchronously.

        Polls the GUI and checks for voice input if STT is provided.
        Returns empty string if window is closed.
        """
        while not self.is_closed:
            # Update GUI
            if self._window:
                self._window.update()

            # Check for pending text input
            if self._pending_inputs:
                return self._pending_inputs.pop(0)

            # Check for voice input
            if stt is not None:
                try:
                    voice_text = stt.input_queue.get_nowait()
                    if voice_text:
                        return voice_text
                except asyncio.QueueEmpty:
                    pass

            # Small delay to prevent busy-waiting
            await asyncio.sleep(0.016)  # ~60fps

        return ""

    def close(self) -> None:
        """Close the window."""
        if self._window and not self._window.is_closed:
            self._window._on_close()
