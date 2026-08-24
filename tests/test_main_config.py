import pytest

import main
from main import (
    ConfigError,
    check_episode_end,
    effective_rewards,
    get_reward,
    get_terminal_reward,
    get_transition_reward,
    load_config,
    validate_config,
)


@pytest.fixture
def base_config(tmp_path):
    exe = tmp_path / 'game.exe'
    exe.write_bytes(b'')
    return {
        'game': {'title': 'Hades', 'executable': str(exe)},
        'agent': {'max_actions_per_episode': 10, 'action_delay_ms': 100},
        'rewards': {
            'neutral': 0.0,
            'attack_hit': 0.5,
            'health_delta_scale': 0.1,
            'time_penalty': 0.0,
            'combat_start': 1.0,
            'death': -5.0,
            'victory': 10.0,
        },
        'actions': {
            'W': {'key': 'w', 'type': 'movement'},
            'SPACE': {'key': 'space', 'type': 'attack'},
        },
    }


def test_validate_ok(base_config):
    validate_config(base_config)


def test_missing_game_section():
    with pytest.raises(ConfigError):
        validate_config({})


def test_missing_executable_key(base_config):
    del base_config['game']['executable']
    with pytest.raises(ConfigError):
        validate_config(base_config)


def test_nonexistent_executable(base_config):
    base_config['game']['executable'] = r'C:\definitely\not\here.exe'
    with pytest.raises(ConfigError):
        validate_config(base_config)


def test_missing_rewards_section(base_config):
    del base_config['rewards']
    with pytest.raises(ConfigError):
        validate_config(base_config)


def test_bad_action_delay(base_config):
    base_config['agent']['action_delay_ms'] = 0
    with pytest.raises(ConfigError):
        validate_config(base_config)


def test_empty_actions_section(base_config):
    base_config['actions'] = {}
    with pytest.raises(ConfigError):
        validate_config(base_config)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------
def test_attack_shaping_keys_off_action_type(base_config):
    combat = {'enemies_present': True}
    empty = {'enemies_present': False}
    assert get_reward(base_config, combat, 'SPACE') == 0.5
    assert get_reward(base_config, empty, 'SPACE') == 0.0
    assert get_reward(base_config, combat, 'W') == 0.0


def test_health_delta_dominates(base_config):
    assert get_reward(base_config, {'health_delta': -10}, 'W') == -1.0
    assert get_reward(base_config, {'health_delta': 20}, 'W') == 2.0


def test_delta_combines_with_attack_shaping(base_config):
    r = get_reward(base_config, {'health_delta': -10,
                                 'enemies_present': True}, 'SPACE')
    assert r == pytest.approx(-1.0 + 0.5)


def test_time_penalty_default_when_unconfigured():
    cfg = {'rewards': {'neutral': 0.0},
           'actions': {'W': {'key': 'w', 'type': 'movement'}}}
    assert get_reward(cfg, {}, 'W') == -0.01


def test_unknown_action_is_neutral_plus_penalty(base_config):
    assert get_reward(base_config, {'analysis': {}}, 'NOT_A_KEY') == 0.0


def test_transition_reward_combat_entry(base_config):
    dialogue = {'scene': 'dialogue'}
    combat = {'scene': 'combat'}
    menu = {'scene': 'menu'}
    assert get_transition_reward(base_config, dialogue, combat) == 1.0
    assert get_transition_reward(base_config, menu, combat) == 1.0
    assert get_transition_reward(base_config, None, combat) == 0.0
    assert get_transition_reward(base_config, combat, combat) == 0.0
    # leaving combat is not rewarded
    assert get_transition_reward(base_config, combat, menu) == 0.0


def test_terminal_rewards(base_config):
    assert get_terminal_reward(base_config, {'player_dead': True}) == -5.0
    assert get_terminal_reward(base_config, {'scene': 'victory'}) == 10.0
    assert get_terminal_reward(base_config, {'scene': 'combat'}) == 0.0


def test_get_reward_accepts_legacy_raw_form(base_config):
    legacy = {'analysis': {'enemies_present': True}}
    assert get_reward(base_config, legacy, 'SPACE') == 0.5


# ---------------------------------------------------------------------------
# Episode end
# ---------------------------------------------------------------------------
def test_check_episode_end_death_and_victory():
    assert check_episode_end({'player_dead': True}) is True
    assert check_episode_end({'scene': 'victory'}) is True
    assert check_episode_end({'scene': 'combat'}) is False
    assert check_episode_end({}) is False
    assert check_episode_end({'analysis': {'player_dead': True}}) is True


# ---------------------------------------------------------------------------
# Shaping annealing
# ---------------------------------------------------------------------------
def test_effective_rewards_anneals_shaping_terms(base_config):
    base_config['learning'] = {'shaping_anneal_episodes': 10,
                               'shaping_floor': 0.5}
    r0 = effective_rewards(base_config, 0)['rewards']
    r5 = effective_rewards(base_config, 5)['rewards']
    r99 = effective_rewards(base_config, 99)['rewards']
    assert r0['attack_hit'] == 0.5 and r0['combat_start'] == 1.0
    assert r5['attack_hit'] == pytest.approx(0.25)
    assert r5['combat_start'] == pytest.approx(0.5)
    assert r99['attack_hit'] == pytest.approx(0.25)  # floored
    # Non-shaping terms untouched
    assert r5['death'] == -5.0 and r5['victory'] == 10.0


def test_effective_rewards_identity_when_disabled(base_config):
    out = effective_rewards(base_config, 500)
    assert out is base_config  # no annealing keys -> same config object


def test_effective_rewards_does_not_mutate_input(base_config):
    base_config['learning'] = {'shaping_anneal_episodes': 5}
    original = base_config['rewards']['attack_hit']
    effective_rewards(base_config, 5)
    assert base_config['rewards']['attack_hit'] == original


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def test_load_config_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'CONFIG_FILE', tmp_path / 'missing.json')
    default = tmp_path / 'default.json'
    default.write_text('{"game": {"title": "X"}}')
    monkeypatch.setattr(main, 'DEFAULT_CONFIG_FILE', default)
    assert load_config() == {'game': {'title': 'X'}}


def test_load_config_no_files_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'CONFIG_FILE', tmp_path / 'a.json')
    monkeypatch.setattr(main, 'DEFAULT_CONFIG_FILE', tmp_path / 'b.json')
    assert load_config() == {}
