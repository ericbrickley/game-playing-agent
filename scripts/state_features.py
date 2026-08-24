"""Canonical state representation shared by perception, learning, and storage.

StateFeatures is the single contract between the perception stack and the
learner: raw capture dicts (from capture_and_analyze) go in one side via
FeatureExtractor, discretizable feature objects come out the other.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger('agent.state_features')

SCENES = {'combat', 'menu', 'dialogue', 'death', 'victory', 'unknown'}
THREAT_DIRS = {'left', 'right', 'up', 'down', 'none', 'unknown'}


@dataclass
class StateFeatures:
    """Learnable summary of one game step.

    health_pct prefers the authoritative pixel measurement and falls back to
    the VL estimate; health_delta is the change vs the previous observation
    and is the primary learning signal (damage taken / healing gained).
    """
    scene: str = 'unknown'
    enemies_present: Optional[bool] = None
    enemies_source: str = 'none'
    threat_dir: str = 'unknown'
    health_pct: Optional[int] = None
    health_source: Optional[str] = None      # 'px' | 'vl' | None
    health_delta: Optional[float] = None     # vs previous observation
    player_dead: bool = False
    prompt_visible: Optional[bool] = None    # pending Phase 2 detection
    cooldown_ready: dict = field(default_factory=dict)  # pending Phase 4
    vl_next_action: Optional[str] = None     # VL's suggested advancing key

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def read_feature(state, name, default=None):
    """Read a named feature from StateFeatures, a feature dict, or a legacy
    raw capture dict ({'analysis': {...}}). Lets old callers migrate gradually."""
    if isinstance(state, dict):
        if name in state:
            return state[name]
        return state.get('analysis', {}).get(name, default)
    return getattr(state, name, default)


def health_bucket(pct):
    """Bucket a health percentage into 4 bands + 'unknown'."""
    if pct is None:
        return 'unknown'
    x = max(0.0, min(100.0, float(pct)))
    if x < 25:
        return 'critical'
    if x < 50:
        return 'low'
    if x < 75:
        return 'mid'
    return 'high'


class FeatureExtractor:
    """Converts a raw capture dict into StateFeatures; tracks health across
    consecutive observations so health_delta reflects damage/heal per step."""

    def __init__(self):
        self._last_health = None

    def reset(self):
        """Call at episode boundaries: deltas don't cross episodes."""
        self._last_health = None

    def extract(self, raw):
        if isinstance(raw, StateFeatures):
            return raw

        raw = raw if isinstance(raw, dict) else {}
        analysis = raw.get('analysis', {}) or {}
        vl = raw.get('vl', {}) or {}

        # Scene: death is authoritative from pixels/VL merge; else VL scene.
        scene = 'death' if analysis.get('player_dead') else vl.get('scene', 'unknown')
        if scene not in SCENES:
            scene = 'unknown'

        # Health: pixel measurement wins; VL estimate is the fallback.
        hp = analysis.get('health_pct_px')
        source = 'px'
        if hp is None:
            hp = analysis.get('health_pct_vl')
            source = 'vl' if hp is not None else None

        delta = None
        if hp is not None and self._last_health is not None:
            delta = hp - self._last_health
        if hp is not None:
            self._last_health = hp

        enemies = analysis.get('enemies_present')
        threat = vl.get('threat_dir', 'unknown')
        if threat not in THREAT_DIRS:
            threat = 'unknown'

        next_action = vl.get('next_action')
        if not isinstance(next_action, str):
            next_action = None

        return StateFeatures(
            scene=scene,
            enemies_present=bool(enemies) if enemies is not None else None,
            enemies_source=analysis.get('enemies_source', 'none'),
            threat_dir=threat,
            health_pct=int(hp) if hp is not None else None,
            health_source=source,
            health_delta=delta,
            player_dead=bool(analysis.get('player_dead')),
            prompt_visible=analysis.get('prompt_visible'),
            vl_next_action=next_action,
        )


# Module-level singleton
_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def extract_features(raw):
    return get_extractor().extract(raw)


# ---------------------------------------------------------------------------
# Scripted (non-learned) policy for non-gameplay scenes.
#
# Design decision: tabular Q-learning handles combat states only. Menus,
# dialogues and death/victory screens are driven by the VL model's
# next_action suggestion (which the perception layer already produces), so
# navigation never has to be *learned* and Q-table entries stay combat-only.
# ---------------------------------------------------------------------------
SCRIPTED_SCENES = {'menu', 'dialogue'}


def scripted_action(features, fallback='SPACE'):
    """Return the key that advances menus/dialogues/death screens, else None.

    Uses the VL suggestion when available; falls back to E in dialogues and
    SPACE elsewhere (dismiss/advance). Returns None for gameplay scenes so
    callers fall through to the learned policy.
    """
    scene = read_feature(features, 'scene')
    dead = bool(read_feature(features, 'player_dead'))
    if scene not in SCRIPTED_SCENES and not dead:
        return None

    suggestion = read_feature(features, 'vl_next_action')
    if suggestion and suggestion != 'none':
        return suggestion
    if scene == 'dialogue':
        return 'E'
    return fallback
