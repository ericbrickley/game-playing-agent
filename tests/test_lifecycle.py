import pytest

import main
from rl_agent import SimpleRLAgent
from state_features import FeatureExtractor, scripted_action

CONFIG = {
    'actions': {'W': {'key': 'w', 'type': 'movement'},
                'E': {'key': 'e', 'type': 'item'},
                'SPACE': {'key': 'space', 'type': 'attack'}},
    'learning': {'enabled': False},
}

COMBAT_RAW = {'analysis': {'enemies_present': True},
              'vl': {'scene': 'combat', 'next_action': 'none'}}

DEAD_RAW = {'analysis': {'player_dead': True, 'enemies_present': False},
            'vl': {'scene': 'death', 'next_action': 'none'}}


class FakeRunner:
    def __init__(self):
        self.keys = []

    def focus_window(self):
        return True

    def execute_action(self, key):
        self.keys.append(key)
        return {'action': key, 'status': 'success'}


# ---------------------------------------------------------------------------
# scripted_action (traversal / menu policy)
# ---------------------------------------------------------------------------
def test_scripted_dialogue_uses_vl_suggestion():
    f = FeatureExtractor().extract({'vl': {'scene': 'dialogue',
                                           'next_action': 'E'}})
    assert scripted_action(f) == 'E'


def test_scripted_menu_falls_back_to_space():
    f = FeatureExtractor().extract({'vl': {'scene': 'menu'}})
    assert scripted_action(f) == 'SPACE'


def test_scripted_death_screen_advances():
    f = FeatureExtractor().extract({'analysis': {'player_dead': True},
                                    'vl': {'scene': 'death'}})
    assert scripted_action(f) == 'SPACE'


def test_no_script_for_gameplay():
    f = FeatureExtractor().extract(COMBAT_RAW)
    assert scripted_action(f) is None


# ---------------------------------------------------------------------------
# ensure_gameplay: drives menus until combat/unknown reached
# ---------------------------------------------------------------------------
def test_ensure_gameplay_drives_menus(monkeypatch):
    seq = [
        {'analysis': {}, 'vl': {'scene': 'menu', 'next_action': 'none'}},
        {'analysis': {}, 'vl': {'scene': 'dialogue', 'next_action': 'E'}},
        COMBAT_RAW,
    ]
    holder = {'i': 0}

    def fake_capture():
        i = holder['i']
        holder['i'] = min(i + 1, len(seq) - 1)
        return seq[i]

    monkeypatch.setattr(main, 'capture_and_analyze', fake_capture)
    runner = FakeRunner()
    ok = main.ensure_gameplay(runner, FeatureExtractor(), 0.0, max_steps=10)
    assert ok is True
    # menu -> fallback SPACE, dialogue -> E suggested by VL
    assert 'SPACE' in runner.keys and 'E' in runner.keys


def test_ensure_gameplay_gives_up_after_max_steps(monkeypatch):
    monkeypatch.setattr(main, 'capture_and_analyze',
                        lambda: {'analysis': {},
                                 'vl': {'scene': 'menu',
                                        'next_action': 'none'}})
    runner = FakeRunner()
    ok = main.ensure_gameplay(runner, FeatureExtractor(), 0.0, max_steps=3)
    assert ok is False
    assert len(runner.keys) == 3


# ---------------------------------------------------------------------------
# run_episode smoke test (full plumbing, no game)
# ---------------------------------------------------------------------------
def test_run_episode_timeout_metrics(monkeypatch):
    monkeypatch.setattr(main, 'capture_and_analyze', lambda: dict(COMBAT_RAW))
    agent = SimpleRLAgent(CONFIG)
    # Seed the Q-table so greedy selection deterministically attacks;
    # otherwise empty states fall back to uniform random exploration.
    agent.eval_mode = True
    from state_encoder import get_state_key
    key = get_state_key(main.get_extractor().extract(dict(COMBAT_RAW)))
    agent.policy[key] = {'W': 0.0, 'E': 0.0, 'SPACE': 1.0}

    config = {
        'agent': {'max_actions_per_episode': 3, 'action_delay_ms': 0},
        'rewards': {'neutral': 0.0, 'time_penalty': 0.0,
                    'health_delta_scale': 0.1, 'attack_hit': 0.5},
        'actions': CONFIG['actions'],
    }
    metrics = main.run_episode(1, FakeRunner(), agent, config, 0.0)
    assert metrics['outcome'] == 'timeout'
    assert metrics['actions'] == 3
    assert metrics['reward'] == pytest.approx(1.5)  # attack shaping x3
    assert metrics['damage_taken'] == 0.0


def test_run_episode_ends_on_death(monkeypatch):
    # Sequence: gameplay consumed by ensure_gameplay, one real combat step,
    # then the next observation shows the death screen -> done=True flows
    # through agent.update and the episode is counted.
    seq = [COMBAT_RAW, COMBAT_RAW, DEAD_RAW]
    holder = {'i': 0}

    def fake_capture():
        i = holder['i']
        holder['i'] = min(i + 1, len(seq) - 1)
        return seq[i]

    monkeypatch.setattr(main, 'capture_and_analyze', fake_capture)
    learning_cfg = json_deep_copy(CONFIG)
    learning_cfg['learning']['enabled'] = True
    learning_cfg['learning']['learning_rate'] = 0.5
    agent = SimpleRLAgent(learning_cfg)
    config = {
        'agent': {'max_actions_per_episode': 10, 'action_delay_ms': 0},
        'rewards': {'neutral': 0.0, 'time_penalty': 0.0,
                    'health_delta_scale': 0.1, 'attack_hit': 0.5,
                    'combat_start': 0.0, 'death': -5.0, 'victory': 10.0},
        'actions': CONFIG['actions'],
    }
    metrics = main.run_episode(1, FakeRunner(), agent, config, 0.0)
    assert metrics['outcome'] == 'dead'
    assert agent.total_episodes == 1


def json_deep_copy(obj):
    import json
    return json.loads(json.dumps(obj))
