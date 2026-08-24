"""On-screen status overlay for robot state visualization."""

import logging
import threading
import time

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

logger = logging.getLogger('agent.status_overlay')


class StatusOverlay:
    """Displays robot state (running/paused/stopped) in a corner of the screen."""
    
    def __init__(self):
        self.root = None
        self.label = None
        self.is_running = True
        self._lock = threading.Lock()
        self._thread = None
        self._stop_flag = False
        
    def start(self):
        """Start the overlay display in a background thread."""
        if not TKINTER_AVAILABLE:
            logger.warning("tkinter not available; status overlay disabled")
            return False
            
        if self._thread is not None:
            return True
            
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Status overlay started")
        return True
    
    def _run(self):
        """Run the tkinter mainloop in a dedicated thread."""
        self.root = tk.Tk()
        self.root.title("Agent Status")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.85)
        self.root.overrideredirect(True)  # No window decorations
        
        # Position in bottom-right corner
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width, height = 200, 60
        x = screen_width - width - 20
        y = screen_height - height - 20
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        self.label = tk.Label(
            self.root,
            text="🟢 RUNNING",
            font=("Segoe UI", 14, "bold"),
            bg="#2E7D32",
            fg="white",
            width=20,
            height=2
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error("Overlay mainloop error: %s", e)
    
    def _on_close(self):
        """Handle window close button."""
        self._stop_flag = True
        if self.root:
            self.root.destroy()
    
    def set_paused(self, paused: bool):
        """Update overlay to show paused or running state."""
        with self._lock:
            self.is_running = not paused
            
        if self.label is None:
            return
            
        try:
            if paused:
                self.label.config(text="⏸️ PAUSED", bg="#F57C00")
                self.label.update()
            else:
                self.label.config(text="🟢 RUNNING", bg="#2E7D32")
                self.label.update()
        except Exception as e:
            logger.debug("Failed to update overlay: %s", e)
    
    def set_stopped(self):
        """Update overlay to show stopped state."""
        with self._lock:
            self.is_running = False
            
        if self.label is None:
            return
            
        try:
            self.label.config(text="🔴 STOPPED", bg="#C62828")
            self.label.update()
        except Exception as e:
            logger.debug("Failed to update overlay: %s", e)
    
    def stop(self):
        """Stop and destroy the overlay."""
        self._stop_flag = True
        if self.root:
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("Status overlay stopped")
    
    def is_active(self) -> bool:
        """Check if overlay is currently active."""
        return self._thread is not None and not self._stop_flag


# Singleton instance
_overlay = None


def get_overlay() -> StatusOverlay:
    """Get or create the status overlay singleton."""
    global _overlay
    if _overlay is None:
        _overlay = StatusOverlay()
    return _overlay


def start_overlay():
    """Start the status overlay."""
    return get_overlay().start()


def update_overlay_state(paused: bool):
    """Update the overlay to show paused or running state."""
    if _overlay is not None:
        _overlay.set_paused(paused)


def stop_overlay():
    """Stop the status overlay."""
    if _overlay is not None:
        _overlay.stop()
