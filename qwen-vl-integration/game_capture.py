"""Game-window capture: grabs the Hades client area directly via PrintWindow.

Unlike pyautogui (which screenshots the whole desktop), this captures the
game's own surface even when other windows overlap it, and returns frames
in client coordinates so HUD calibration stays valid regardless of where
the window sits on screen. Falls back to pyautogui full-screen capture if
pywin32 or the game window is unavailable.
"""

import ctypes

from PIL import Image

_hwnd_cache: dict = {"hwnd": None}
GAME_WINDOW_TITLE = "Hades"


def find_game_hwnd(title=GAME_WINDOW_TITLE):
    """Locate the game window handle (cached)."""
    if _hwnd_cache["hwnd"] and win32_is_window(_hwnd_cache["hwnd"]):
        return _hwnd_cache["hwnd"]
    try:
        import win32gui
    except ImportError:
        return None
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        _hwnd_cache["hwnd"] = hwnd
    return hwnd


def win32_is_window(hwnd):
    try:
        import win32gui
        return bool(win32gui.IsWindow(hwnd))
    except Exception:
        return False


def capture_game_window():
    """Return a PIL RGB image of the game client area, or None on failure."""
    hwnd = find_game_hwnd()
    if not hwnd:
        return None
    try:
        import win32gui
        import win32ui
        import win32con
    except ImportError:
        return None

    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        w, h = r - l, b - t
        if w <= 0 or h <= 0:
            return None
        if win32gui.IsIconic(hwnd):
            return None  # minimized

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        # PW_RENDERFULLCONTENT: works for GPU-composited/DX windows
        ok = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        info = bmp.GetInfo()
        data = bmp.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                               data, "raw", "BGRX", 0, 1)
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        if not ok:
            return None
        return img.copy()
    except Exception:
        return None


def capture_screen_fallback():
    """Full-desktop capture fallback (coordinates NOT client-relative)."""
    import pyautogui
    return pyautogui.screenshot()


if __name__ == "__main__":
    img = capture_game_window() or capture_screen_fallback()
    print("captured:", img.size)
