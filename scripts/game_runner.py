import json
import logging
import subprocess
import time
from pathlib import Path

try:
    from action_space import get_action_space
except ImportError:  # imported as part of the 'scripts' package
    from .action_space import get_action_space

logger = logging.getLogger('agent.game_runner')


class GameRunner:
    """Manages game launch, input execution, and shutdown."""

    def __init__(self, game_exe, config_path=None, action_space=None,
                 move_hold_ms=120):
        self.game_exe = Path(game_exe)
        self.config = self._load_config(config_path)
        self.process = None
        self.pid = None
        self.is_running = False
        self.action_space = action_space or get_action_space()
        # How long continuous (movement) keys are held per action, ms.
        # Hades movement is continuous input; single taps barely move.
        self.move_hold_ms = float(move_hold_ms)
        self._last_exec = {}  # action name -> monotonic timestamp
        self._attached = False  # True when we adopted an already-running game

    def _load_config(self, config_path):
        """Load game configuration"""
        default_config = {
            "executable": str(self.game_exe),
            "window_mode": "fullscreen",
            "resolution": "1920x1080"
        }

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not read runner config: %s", e)

        return default_config

    def _game_already_running(self):
        """True if the game executable is already running (Windows tasklist).
        Used to attach to the existing instance instead of spawning a dupe."""
        try:
            out = subprocess.run(
                ['tasklist', '/FI', f'IMAGENAME eq {self.game_exe.name}'],
                capture_output=True, text=True, timeout=10)
            return self.game_exe.name.lower() in (out.stdout or '').lower()
        except Exception as e:
            logger.debug("tasklist check failed: %s", e)
            return False

    def launch(self):
        """Launch the game, or attach if it is already running."""
        logger.info("Launching: %s", self.game_exe)

        if self._game_already_running():
            logger.info("Game already running - attaching instead of "
                        "launching a second instance")
            self._attached = True
            self.is_running = True
            self.pid = None
            self.focus_window()
            return True

        try:
            self.process = subprocess.Popen(
                [str(self.game_exe)],
                cwd=str(self.game_exe.parent)
            )
            self.pid = self.process.pid
            self.is_running = True

            logger.info("Game launched with PID: %d", self.pid)

            # Wait for game to initialize
            time.sleep(5)

            if self._check_running():
                logger.info("Game initialized successfully")
                self.focus_window()
                return True

            logger.error("Failed to initialize game")
            return False

        except Exception as e:
            logger.error("Failed to launch game: %s", e)
            return False

    def focus_window(self, title="Hades"):
        """Bring the game window to the foreground so keys land in it."""
        try:
            import win32gui
        except ImportError:
            logger.debug("pywin32 unavailable; cannot focus game window")
            return False
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            logger.debug("Game window %r not found", title)
            return False
        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            logger.debug("SetForegroundWindow failed: %s", e)
            return False

    def _check_running(self):
        """Check if game process is running"""
        return self.process is not None and self.process.poll() is None

    def capture_frame(self):
        """Capture and analyze current game state"""
        from visual_state import capture_and_analyze

        state = capture_and_analyze()
        state['capture_time'] = time.time()
        state['game_pid'] = self.pid
        return state

    def execute_action(self, action_name):
        """Execute a named action with hold support and cooldown enforcement.

        Continuous actions (movement) are keyDown/hold/keyUp so Hades
        registers real movement; discrete actions are single presses.
        Actions still inside their declared cooldown are skipped.
        """
        try:
            import pyautogui
        except Exception as e:
            return {'action': action_name, 'status': 'error',
                    'error': f'pyautogui unavailable: {e}'}

        meta = self.action_space.actions.get(action_name, {})
        key = meta.get('key', str(action_name).lower())
        input_type = meta.get('input_type', 'discrete')
        cooldown = float(meta.get('cooldown', 0) or 0)

        now = time.monotonic()
        last = self._last_exec.get(action_name)
        if cooldown > 0 and last is not None and now - last < cooldown:
            return {'action': action_name, 'status': 'cooldown'}

        try:
            if input_type == 'continuous':
                hold_s = max(0.03, self.move_hold_ms / 1000.0)
                pyautogui.keyDown(key)
                time.sleep(hold_s)
                pyautogui.keyUp(key)
            else:
                pyautogui.press(key)
            self._last_exec[action_name] = now
            return {'action': action_name, 'status': 'success'}
        except Exception as e:
            return {'action': action_name, 'status': 'error', 'error': str(e)}

    def emergency_stop(self):
        """Release all held keys immediately - breaks keyboard control lock.
        
        Call this when Windows Defender or other security software interferes,
        or when the agent gets stuck sending continuous input.
        """
        try:
            import pyautogui
            # Release all possible held keys
            for key in ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl', 'alt', 'g', 'r', 'q', 'e', 'f']:
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass
            logger.info("Emergency stop: all keys released")
            return True
        except Exception as e:
            logger.error("Emergency stop failed: %s", e)
            return False

    def shutdown(self):
        """Shutdown the game"""
        # First release any held keys
        self.emergency_stop()
        if self._attached:
            logger.info("Attached to a pre-existing game instance - "
                        "leaving it running")
            self.process = None
            self.pid = None
            self.is_running = False
            self._attached = False
            return
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Game terminated")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("Game killed (did not exit gracefully)")
            except Exception as e:
                logger.error("Failed to terminate game: %s", e)
            self.process = None
            self.pid = None
            self.is_running = False


# Singleton instance
_game_runner = None


def get_game_runner(game_exe=None, config_path=None, **kwargs):
    """Get or create game runner instance"""
    global _game_runner

    if _game_runner is None or (game_exe and _game_runner.game_exe != Path(game_exe)):
        _game_runner = GameRunner(game_exe, config_path, **kwargs)

    return _game_runner
