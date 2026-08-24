"""
Game Playing Agent for Qwen 3.5
================================

This package enables Qwen 3.5 to play video games (starting with Hades)
with reinforcement learning capabilities.

Components:
- visual_state.py: Screen capture and visual analysis
- action_space.py: Game action definitions
- experience_replay.py: Experience replay buffer
- rl_agent.py: Reinforcement learning agent
- game_runner.py: Game launcher and state management
- state_encoder.py: State representation encoding
"""

from .game_runner import GameRunner, get_game_runner
from .action_space import ActionSpace, get_action_space
from .experience_replay import (
    ExperienceReplayBuffer, get_buffer, add_experience,
    sample_batch, get_buffer_stats
)
from .visual_state import VisualStateAnalyzer, capture_and_analyze
from .rl_agent import (
    SimpleRLAgent, get_agent, select_action, update_agent,
    get_agent_stats, save_agent, load_agent
)
from .state_encoder import StateEncoder, encode_state, get_state_key
from .state_features import (
    StateFeatures, FeatureExtractor, get_extractor, read_feature
)

__all__ = [
    'GameRunner',
    'get_game_runner',
    'ActionSpace',
    'get_action_space',
    'ExperienceReplayBuffer',
    'get_buffer',
    'add_experience',
    'sample_batch',
    'get_buffer_stats',
    'VisualStateAnalyzer',
    'capture_and_analyze',
    'SimpleRLAgent',
    'get_agent',
    'select_action',
    'update_agent',
    'get_agent_stats',
    'save_agent',
    'load_agent',
    'StateEncoder',
    'encode_state',
    'get_state_key',
    'StateFeatures',
    'FeatureExtractor',
    'get_extractor',
    'read_feature'
]
