import logging
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

logger = logging.getLogger('agent.visual_state')

# VL readings older than this are treated as unavailable for gap-filling.
# The VL refresher runs every ~2.5 s (5 s on error), so anything older than
# 6 s means the server has stopped producing fresh analysis.
VL_MAX_AGE_S = 6.0

class VisualStateAnalyzer:
    """Analyzes game screen state for AI agent"""
    
    def __init__(self):
        self.screen_capture = None
        self.state_description = None
        self.last_update = None
        
    def capture_screen(self):
        """Capture current game screen (game window client area if possible)"""
        source = 'screen'
        shot = None
        try:
            vl_dir = Path(__file__).resolve().parent.parent / 'qwen-vl-integration'
            if str(vl_dir) not in sys.path:
                sys.path.insert(0, str(vl_dir))
            from game_capture import capture_game_window
            shot = capture_game_window()
            if shot is not None:
                source = 'game_window'
        except Exception:
            shot = None
        if shot is None:
            try:
                import pyautogui
                shot = pyautogui.screenshot()
            except Exception as e:
                logger.warning("Screen capture failed: %s", e)
                return None

        # Store capture
        self.screen_capture = shot
        self.last_update = time.time()

        return {
            'image': shot,
            'width': shot.width,
            'height': shot.height,
            'timestamp': time.time(),
            'source': source
        }
            
    def describe_state(self):
        """Describe current game state"""
        if self.screen_capture is None:
            return {'error': 'No screen captured'}
            
        image = self.screen_capture
        width, height = image.width, image.height
        
        # Analyze image
        state = {
            'resolution': f'{width}x{height}',
            'analysis': self._analyze_game_state(image),
            'dominant_colors': self._get_dominant_colors(image),
            'ui_elements': self.detect_ui_elements(image)
        }
        
        self.state_description = state
        return state
        
    def _analyze_game_state(self, image):
        """Analyze game state from image"""
        # Convert to numpy array for analysis
        rgb = np.array(image)
        
        # Calculate average brightness
        avg_brightness = float(np.mean(rgb) / 255.0)
        
        # Simple state detection based on visual cues
        state = {
            'brightness': avg_brightness,
            'is_dark': avg_brightness < 0.3,
            'is_bright': avg_brightness > 0.6,
            'analysis_confidence': 0.7  # Placeholder - would improve with ML
        }
        
        # Check for UI elements (simplified)
        state['has_hud'] = self._detect_hud(image)
        state['has_health_bar'] = self._detect_health_bar(image)
        state['has_enemies'] = self._detect_enemies(image)
        
        return state
        
    def _detect_hud(self, image):
        """Detect HUD elements (placeholder - would need proper detection)"""
        return True
            
    def _detect_health_bar(self, image):
        """Detect health bar"""
        try:
            import cv2
            
            # Check for red/orange colors in typical health bar regions
            hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
            
            # Red color range for health
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([15, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # Check if health bar is present (not empty)
            red_pixels = np.sum(mask > 0)
            return red_pixels > 100  # Threshold
        except Exception:
            return False
            
    def _detect_enemies(self, image):
        """Detect enemy presence"""
        try:
            # Look for red/orange blobs that aren't health bars
            import cv2
            hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
            
            # Red color range
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([15, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # Orange/yellow for enemies
            lower_orange = np.array([15, 100, 100])
            upper_orange = np.array([25, 255, 255])
            mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(mask, mask2)
            
            # Count enemy-like regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
            
            # Filter by size (enemies should be reasonably sized)
            enemy_regions = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 1000 < area < 50000:  # Reasonable size for enemies
                    enemy_regions.append(centroids[i])
            
            return len(enemy_regions) > 0
        except Exception:
            return False
            
    def _get_dominant_colors(self, image):
        """Get dominant colors in image"""
        try:
            import cv2
            
            hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
            
            # Color histogram
            color_hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            
            # Find peaks (dominant colors)
            dominant_colors = []
            for i in range(color_hist.shape[0]):
                for j in range(color_hist.shape[1]):
                    if color_hist[i, j] > 1000:
                        hue = i / 32 * 180
                        saturation = j / 32 * 256
                        dominant_colors.append(f"H{hue:.0f},S{saturation:.0f}")
                        
            return dominant_colors[:10]  # Top 10 colors
        except Exception:
            return []
            
    def detect_ui_elements(self, image):
        """Detect UI elements"""
        # Simplified UI detection
        return {
            'menu_open': False,
            'dialog_box': False,
            'boss_health': False,
            'item_drop': False,
            'health_bar': self._detect_health_bar(image)
        }

# Global analyzer instance
_analyzer = VisualStateAnalyzer()

def _health_bar_in_hud(image):
    """Deterministic check: does a red health-bar-like blob exist bottom-left?

    Hades draws the player health bar at the bottom-left of the screen.
    Used to ground (or veto) VL-model claims about health.
    """
    try:
        import cv2
        import numpy as np

        w, h = image.width, image.height
        # Bottom-left region where Hades renders the player health bar
        x0, y0 = 0, int(h * 0.55)
        x1, y1 = int(w * 0.45), h
        crop = np.array(image.crop((x0, y0, x1, y1)))
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([18, 255, 255]))
        return int(np.sum(mask > 0))
    except Exception:
        return -1  # unknown (cv2 missing etc.) -> don't veto

def measure_health_bar(image):
    """Pixel-measure Hades' health bar fill (authoritative, no ML).

    Calibrated against the game's CLIENT area captured via PrintWindow at
    1768x992, validated against live HP readings (69/75 -> boundary x~318,
    45/75 -> boundary x~229): bar track runs x~49 to x~342; a separate
    always-red HUD element sits near x~480-520 and must be ignored.
    Coordinates scale with client height.
    Returns fill 0-100 or None if no bar is found (menus etc.).
    """
    try:
        import cv2
        import numpy as np

        w, h = image.size
        s = h / 992.0  # calibration reference height
        ry0, ry1 = int(950 * s), int(982 * s)
        rx1 = min(int(560 * s), w)
        crop = np.array(image.crop((0, ry0, rx1, ry1)))
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([20, 255, 255]))
        band_h = mask.shape[0]
        col_counts = mask.sum(axis=0) // 255
        active = np.where(col_counts >= max(3, int(band_h * 0.35)))[0]
        if not len(active):
            return None
        # Walk right from the left edge while gaps stay small (<25px),
        # so detached decorations (end caps, other elements) are excluded
        left = int(active[0])
        right = left
        for i in range(1, len(active)):
            if active[i] - active[i - 1] <= int(25 * s):
                right = int(active[i])
            else:
                break
        track = (342 - 49) * s
        fill = (right - 49 * s) / max(1.0, track)
        return {
            'fill_pct': max(0, min(100, round(fill * 100))),
            'bar_left': left,
            'bar_right': right,
            'bar_thickness': band_h,
        }
    except Exception:
        return None

_px_history = []

def smoothed_health_pct(image):
    """Median of recent pixel measurements - kills flash-animation transients."""
    r = measure_health_bar(image)
    if r is None:
        return None
    _px_history.append(r['fill_pct'])
    del _px_history[:-3]
    vals = sorted(_px_history)
    return vals[len(vals) // 2]

def _enrich_with_vl(state):
    """Merge cached VL scene analysis into fresh pixel heuristics.

    Freshness policy: pixel measurements are recomputed on every capture and
    win immediately; VL output (background-refreshed every ~2.5 s) only fills
    gaps while young enough (VL_MAX_AGE_S), so decisions never key off stale
    perception.
    """
    try:
        vl_dir = Path(__file__).resolve().parent.parent / 'qwen-vl-integration'
        if str(vl_dir) not in sys.path:
            sys.path.insert(0, str(vl_dir))
        from vl_client import get_vl_state

        vl = get_vl_state()
        state['vl'] = vl
        analysis = state.setdefault('analysis', {})
        age = vl.get('age')
        vl_ok = (bool(vl.get('available'))
                 and isinstance(age, (int, float)) and age <= VL_MAX_AGE_S)
        analysis['vl_usable'] = vl_ok

        # Enemies: instant positive from fresh pixel blobs; otherwise trust
        # VL (better recall) only when fresh.
        if analysis.get('has_enemies'):
            analysis['enemies_present'] = True
            analysis['enemies_source'] = 'pixels'
        elif vl_ok and vl.get('enemies_visible') is not None:
            analysis['enemies_present'] = bool(vl.get('enemies_visible'))
            analysis['enemies_source'] = 'vl'
        else:
            analysis['enemies_present'] = bool(analysis.get('has_enemies'))
            analysis['enemies_source'] = 'pixels'

        hp = vl.get('health_pct')
        if vl_ok and hp is not None and vl.get('health_bar_visible'):
            pixels = _health_bar_in_hud(_analyzer.screen_capture)
            analysis['hud_health_pixels'] = pixels
            if pixels == -1 or pixels > 300:
                analysis['health_pct_vl'] = hp
            else:
                analysis['health_pct_vl'] = None
                analysis['vl_health_vetoed'] = True
        else:
            analysis['health_pct_vl'] = None
        # Authoritative pixel measurement overrides any VL estimate
        px_hp = smoothed_health_pct(_analyzer.screen_capture)
        if px_hp is not None:
            analysis['health_pct_px'] = px_hp
        # Death screens persist until dismissed, so staleness is tolerable here
        if vl.get('scene') == 'death':
            analysis['player_dead'] = True
        return state
    except Exception as e:
        state['vl'] = {'available': False, 'raw': f'vl integration error: {e}'}
        logger.debug("VL enrichment failed: %s", e)
        return state

def capture_and_analyze():
    """Capture screen and analyze state"""
    analyzer = _analyzer
    analyzer.capture_screen()
    return _enrich_with_vl(analyzer.describe_state())

def get_state_description():
    """Get current state description"""
    analyzer = _analyzer
    return analyzer.describe_state()

def get_analyzer():
    """Get analyzer instance"""
    return _analyzer
