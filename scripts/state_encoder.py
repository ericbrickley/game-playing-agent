"""Discretizes StateFeatures into bounded tabular state keys for Q-learning.

Key space: scene (6) x threat_dir (6) x health bucket (5) x enemies
(yes/no/unknown = 3) -> at most 540 reachable states, sized for tabular
Q-learning. Accepts StateFeatures objects or plain feature dicts.
"""

import logging
from datetime import datetime

try:
    from state_features import health_bucket, read_feature
except ImportError:  # imported as part of the 'scripts' package
    from .state_features import health_bucket, read_feature

logger = logging.getLogger('agent.state_encoder')

ENCODER_VERSION = 2
HISTORY_LIMIT = 100


def format_state_key(scene, threat_dir, bucket, enemies_present):
    enemy_flag = '?' if enemies_present is None else str(bool(enemies_present))
    return f"{scene}|{threat_dir}|{bucket}|{enemy_flag}"


class StateEncoder:

    def __init__(self):
        self.state_history = []

    @staticmethod
    def _fields(features):
        return (
            read_feature(features, 'scene', 'unknown'),
            read_feature(features, 'threat_dir', 'unknown'),
            health_bucket(read_feature(features, 'health_pct')),
            read_feature(features, 'enemies_present', None),
        )

    def discretize(self, features):
        """Map a feature state onto its discrete tabular key."""
        return format_state_key(*self._fields(features))

    def encode(self, features):
        """Discretize and record in history; returns the history entry."""
        entry = {
            'key': self.discretize(features),
            'scene': read_feature(features, 'scene', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'encoder_version': ENCODER_VERSION,
        }
        self.state_history.append(entry)
        del self.state_history[:-HISTORY_LIMIT]
        return entry

    def get_history(self, limit=10):
        return self.state_history[-limit:]

    def clear_history(self):
        self.state_history = []


# Global encoder instance
_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = StateEncoder()
    return _encoder


def encode_state(features):
    """Encode a feature state (discretize + history)."""
    return get_encoder().encode(features)


def get_state_key(features):
    """Get the discrete state key for a feature state."""
    return get_encoder().discretize(features)
