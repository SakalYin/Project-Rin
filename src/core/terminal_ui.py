"""
Terminal UI — colored, clean chat interface.

Displays only user/AI messages with color coding.
Hides all logging output from the terminal.
"""

from __future__ import annotations

import sys

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Bright variants
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_GREEN = "\033[92m"


class TerminalUI:
    """Clean terminal interface for chat."""

    def __init__(self, user_name: str = "You", ai_name: str = "Rin"):
        self.user_name = user_name
        self.ai_name = ai_name
        self._enable_colors()

    def _enable_colors(self) -> None:
        """Enable ANSI colors on Windows."""
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="", flush=True)

    def print_header(self) -> None:
        """Print the app header."""
        print()
        print(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║{Colors.RESET}          {Colors.BRIGHT_CYAN}✨ Project Rin — Local AI Voice Agent ✨{Colors.RESET}          {Colors.BRIGHT_MAGENTA}{Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()

    def print_status(self, stt_enabled: bool) -> None:
        """Print status info."""
        if stt_enabled:
            print(f"  {Colors.GREEN}●{Colors.RESET} {Colors.DIM}Voice input enabled — speak or type{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}○{Colors.RESET} {Colors.DIM}Type your message and press Enter{Colors.RESET}")
        print(f"  {Colors.DIM}Type {Colors.WHITE}\"quit\"{Colors.DIM} or press {Colors.WHITE}Ctrl+C{Colors.DIM} to exit{Colors.RESET}")
        print()
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        print()

    def print_user_prompt(self) -> str:
        """Print user prompt and get input."""
        try:
            return input(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{self.user_name}:{Colors.RESET} ")
        except EOFError:
            return ""

    def print_user_message(self, text: str) -> None:
        """Print a user message (for STT transcriptions)."""
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{self.user_name}:{Colors.RESET} {text}")

    def start_ai_response(self) -> None:
        """Print AI name prefix before streaming."""
        print(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}{self.ai_name}:{Colors.RESET} ", end="", flush=True)

    def print_ai_chunk(self, chunk: str) -> None:
        """Print a chunk of AI response (streaming)."""
        print(f"{Colors.WHITE}{chunk}{Colors.RESET}", end="", flush=True)

    def end_ai_response(self) -> None:
        """End AI response line."""
        print()
        print()

    def print_error(self, message: str) -> None:
        """Print an error message."""
        print(f"\n{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} {Colors.RED}{message}{Colors.RESET}\n")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        print(f"{Colors.DIM}{message}{Colors.RESET}")

    def print_goodbye(self) -> None:
        """Print goodbye message."""
        print(f"\n{Colors.BRIGHT_MAGENTA}Bye bye~ 👋{Colors.RESET}\n")

    def print_interrupted(self) -> None:
        """Print interrupted message."""
        print(f"\n\n{Colors.YELLOW}Interrupted — shutting down...{Colors.RESET}\n")


# Global instance for easy access
ui = TerminalUI()
