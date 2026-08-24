import json
import random

import pytest

from experience_replay import ExperienceReplayBuffer
from rl_agent import SimpleRLAgent
from state_encoder import get_state_key
from state_features import StateFeatures

# Mirrors the 'actions' section of config/game_config.json
FULL_ACTIONS = {
    'W': {'key': 'w', 'type': 'movement'},
    'A': {'key': 'a', 'type': 'movement'},
    'S': {'key': 's', 'type': 'movement'},
    'D': {'key': 'd', 'type': 'movement'},
    'SPACE': {'key': 'space', 'type': 'attack'},
    'SHIFT': {'key': 'shift', 'type': 'dash'},
    'ALT': {'key': 'g', 'type': 'skill'},
    'TAB': {'key': 'r', 'type': 'skill'},
    'Q': {'key': 'q', 'type': 'skill'},
    'E': {'key': 'e', 'type': 'item'},
    'F': {'key': 'f', 'type': 'item'},
}

# Deterministic hyperparameters for math assertions
CONFIG = {
    'actions': FULL_ACTIONS,
    'learning': {
        'enabled': True,
        'learning_rate': 0.5,
        'discount_factor': 0.9,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.5,
        'epsilon_min': 0.01,
    },
}


@pytest.fixture
def agent():
    return SimpleRLAgent(CONFIG)


def combat_features():
    return {'scene': 'combat', 'threat_dir': 'left', 'health_pct': 60,
            'enemies_present': True}


def test_regression_full_action_set_includes_interact(agent):
    # Regression for the main.py wiring bug: the agent must be built from the
    # project config, which exposes E (interact). Without E it can never open
    # doors and traversal is impossible.
    assert set(agent.actions) == set(FULL_ACTIONS)


def test_default_action_set_is_documented_subset():
    agent = SimpleRLAgent(None)
    assert set(agent.actions) == {'W', 'A', 'S', 'D', 'SPACE', 'SHIFT'}


def test_select_action_returns_known_action(agent):
    random.seed(0)
    for _ in range(50):
        assert agent.select_action({'analysis': {}}) in agent.actions


def test_q_update_td_math(agent):
    s = combat_features()
    agent.update(s, 'SPACE', 1.0, s, False)
    # target = 1.0 + gamma * max(Q[s']) where Q[s'] was empty -> 1.0
    assert agent.policy[get_state_key(s)]['SPACE'] == pytest.approx(0.5)

    agent.update(s, 'SPACE', 1.0, s, False)
    # target = 1.0 + 0.9 * 0.5 = 1.45; q = 0.5 + 0.5*(1.45-0.5) = 0.975
    assert agent.policy[get_state_key(s)]['SPACE'] == pytest.approx(0.975)


def test_q_update_done_zeroes_bootstrap(agent):
    s = combat_features()
    agent.update(s, 'SHIFT', -2.0, s, True)
    assert agent.policy[get_state_key(s)]['SHIFT'] == pytest.approx(-1.0)


def test_epsilon_decays_per_completed_episode(agent):
    s = combat_features()
    start = agent._get_epsilon()
    assert start == pytest.approx(1.0)
    for _ in range(3):
        agent.update(s, 'W', 0.0, s, True)
    assert agent._get_epsilon() == pytest.approx(1.0 * 0.5 ** 3)


def test_eval_mode_forces_greedy(agent):
    s = combat_features()
    key = get_state_key(s)
    agent.policy[key] = {'SHIFT': 2.0, 'W': 0.1, 'A': 0.0}
    agent.eval_mode = True
    for _ in range(30):
        assert agent.select_action(s) == 'SHIFT'


def test_scripted_menu_beats_q_table(agent):
    agent.eval_mode = True
    assert agent.select_action(StateFeatures(scene='menu')) == 'SPACE'
    assert agent.select_action(StateFeatures(scene='dialogue',
                                             vl_next_action='E')) == 'E'


def test_random_baseline_stays_in_action_set(agent):
    agent.random_baseline = True
    random.seed(0)
    for _ in range(20):
        assert agent.select_action(combat_features()) in agent.actions


def test_learning_disabled_skips_update():
    cfg = json.loads(json.dumps(CONFIG))
    cfg['learning']['enabled'] = False
    agent = SimpleRLAgent(cfg)
    agent.update({}, 'W', 1.0, {}, False)
    assert agent.policy == {}
    assert agent.buffer is None


def test_save_load_roundtrip(agent, tmp_path):
    path = tmp_path / 'agent.json'
    s = combat_features()
    agent.update(s, 'SPACE', 1.0, s, False)
    agent.episode_rewards = [3.5]
    agent.total_episodes = 2
    agent.save(path)

    restored = SimpleRLAgent(CONFIG)
    assert restored.load(path) is True
    assert restored.policy == agent.policy
    assert restored.total_episodes == 2


def test_load_legacy_format_ignored_gracefully(agent, tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'policy': {'W': 0.5}, 'stats': {
        'total_episodes': 4, 'wins': 1}}))
    assert agent.load(path) is True
    assert agent.policy == {}
    assert agent.total_episodes == 4


def test_load_missing_file_returns_false(tmp_path):
    agent = SimpleRLAgent(CONFIG)
    assert agent.load(tmp_path / 'nope.json') is False


def test_reset_restores_fresh_state(agent):
    s = combat_features()
    agent.update(s, 'W', 1.0, s, True)
    agent.reset()
    assert agent.policy == {}
    assert agent.total_episodes == 0
    assert agent._get_epsilon() == pytest.approx(1.0)


def test_buffer_receives_feature_states(agent):
    s = combat_features()
    ns = dict(s, health_pct=55)
    agent.update(s, 'W', -1.0, ns, False)
    exp = agent.buffer.buffer[-1]
    assert exp['state']['health_pct'] == 60
    assert exp['next_state']['health_pct'] == 55
