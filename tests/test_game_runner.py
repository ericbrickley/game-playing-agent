import sys

import pytest

from action_space import get_action_space
from game_runner import GameRunner


class FakePyautogui:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(('press', key))

    def keyDown(self, key):
        self.events.append(('down', key))

    def keyUp(self, key):
        self.events.append(('up', key))


@pytest.fixture
def fake_pg(monkeypatch):
    fake = FakePyautogui()
    monkeypatch.setitem(sys.modules, 'pyautogui', fake)
    return fake


@pytest.fixture
def runner():
    return GameRunner(r'C:\game\hades.exe', action_space=get_action_space(),
                      move_hold_ms=50)


def test_discrete_action_presses(runner, fake_pg):
    result = runner.execute_action('SPACE')
    assert result['status'] == 'success'
    assert ('press', 'space') in fake_pg.events


def test_movement_uses_hold_not_tap(runner, fake_pg):
    result = runner.execute_action('W')
    assert result['status'] == 'success'
    downs = [e for e in fake_pg.events if e[0] == 'down']
    ups = [e for e in fake_pg.events if e[0] == 'up']
    assert ('down', 'w') in downs
    assert ('up', 'w') in ups


def test_cooldown_blocks_immediate_repeat(runner, fake_pg):
    first = runner.execute_action('SHIFT')
    second = runner.execute_action('SHIFT')
    assert first['status'] == 'success'
    assert second['status'] == 'cooldown'
    presses = [e for e in fake_pg.events if e[0] == 'press']
    assert len(presses) == 1


def test_remapped_skill_keys_avoid_os_traps(runner, fake_pg):
    runner.execute_action('ALT')
    runner.execute_action('TAB')
    keys = {k for _, k in fake_pg.events}
    assert 'g' in keys and 'r' in keys
    assert 'alt' not in keys and 'tab' not in keys


def test_unknown_action_falls_back_to_lowercase(runner, fake_pg):
    result = runner.execute_action('ESC')
    assert result['status'] == 'success'
    assert ('press', 'esc') in fake_pg.events


def test_focus_window_missing_is_safe(runner):
    # Must return a bool whether or not pywin32/the window exist.
    assert isinstance(runner.focus_window(title='No Such Window'), bool)


def test_pyautogui_missing_reports_error(runner, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyautogui', None)
    result = runner.execute_action('W')
    assert result['status'] == 'error'
    assert 'pyautogui' in result['error']
