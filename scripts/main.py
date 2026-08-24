"""Main entry point: launch the game and run the agent (demo / train / test)."""

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from pathlib import Path

from game_runner import get_game_runner
from rl_agent import get_agent
from state_encoder import get_encoder
from state_features import get_extractor, read_feature, scripted_action
from visual_state import capture_and_analyze
from status_overlay import start_overlay, update_overlay_state, stop_overlay

logger = logging.getLogger('agent.main')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / 'config' / 'game_config.json'
DEFAULT_CONFIG_FILE = PROJECT_ROOT / 'config' / 'default_config.json'

# Global pause control
_pause_lock = threading.Lock()
_is_paused = False
_pause_event = threading.Event()
_emergency_stop_flag = False


def toggle_pause():
    """Toggle pause state (called when F12 is pressed)."""
    global _is_paused
    with _pause_lock:
        _is_paused = not _is_paused
        if _is_paused:
            _pause_event.set()
            logger.info("Agent PAUSED - press F12 to resume")
            update_overlay_state(paused=True)
        else:
            _pause_event.clear()
            logger.info("Agent RESUMED")
            update_overlay_state(paused=False)


def check_emergency_stop():
    """Check if emergency stop was triggered."""
    return _emergency_stop_flag


def listen_for_pause_key(key='f12'):
    """Listen for F12 keypress to toggle pause."""
    try:
        import keyboard
        keyboard.add_hotkey(key, toggle_pause)
        logger.info("Pause hotkey registered: %s", key)
        keyboard.wait()  # Block until interrupted
    except ImportError:
        logger.warning("keyboard module not available; pause via Ctrl+C only")
    except Exception as e:
        logger.warning("Failed to register pause hotkey: %s", e)


class ConfigError(Exception):
    """Raised when the loaded configuration is invalid."""


def load_config():
    """Load game_config.json, falling back to default_config.json."""
    for candidate in (CONFIG_FILE, DEFAULT_CONFIG_FILE):
        if candidate.exists():
            with open(candidate, 'r') as f:
                return json.load(f)
    return {}


def validate_config(config):
    """Raise ConfigError if the configuration is unusable."""
    game = config.get('game')
    if not isinstance(game, dict):
        raise ConfigError("missing 'game' section")
    exe = game.get('executable')
    if not exe:
        raise ConfigError("missing 'game.executable'")
    if not Path(exe).exists():
        raise ConfigError(f"game executable not found: {exe}")

    agent_cfg = config.get('agent')
    if not isinstance(agent_cfg, dict):
        raise ConfigError("missing 'agent' section")
    if not agent_cfg.get('max_actions_per_episode'):
        raise ConfigError("missing 'agent.max_actions_per_episode'")
    delay = agent_cfg.get('action_delay_ms')
    if not isinstance(delay, (int, float)) or delay <= 0:
        raise ConfigError("'agent.action_delay_ms' must be a positive number")

    if not isinstance(config.get('rewards'), dict):
        raise ConfigError("missing 'rewards' section")

    actions = config.get('actions')
    if not isinstance(actions, dict) or not actions:
        raise ConfigError("missing or empty 'actions' section")


def effective_rewards(config, episodes_completed):
    """Config copy with shaping weights (attack_hit, combat_start) annealed
    by training progress. Shaping exists to bootstrap early learning; fading
    it lets the true objective (survival deltas) dominate later. Non-shaping
    weights (health/time/death/victory) are never touched."""
    learn = config.get('learning', {})
    span = int(learn.get('shaping_anneal_episodes', 0))
    rewards = config.get('rewards', {})
    if span <= 0:
        return config
    floor = float(learn.get('shaping_floor', 0.5))
    frac = max(floor, 1.0 - episodes_completed / span)
    out = dict(config)
    out['rewards'] = dict(rewards)
    for key in ('attack_hit', 'combat_start'):
        if key in out['rewards']:
            out['rewards'][key] = round(out['rewards'][key] * frac, 4)
    return out


def get_reward(config, features, action):
    """Per-step reward. Dominant term is the health change caused by the
    action (damage taken / healing gained); attack shaping and a small time
    penalty keep the agent engaged and moving."""
    r_cfg = config.get('rewards', {})
    total = r_cfg.get('time_penalty', -0.01)

    delta = read_feature(features, 'health_delta')
    if delta is not None:
        total += r_cfg.get('health_delta_scale', 0.1) * float(delta)
    else:
        # Fallback: penalize being in combat with unknown health (risk signal)
        scene = read_feature(features, 'scene')
        if scene == 'combat' and read_feature(features, 'health_pct') is None:
            total += r_cfg.get('unknown_health_penalty', -0.05)

    action_type = config.get('actions', {}).get(action, {}).get('type', '')
    if action_type == 'attack' and bool(read_feature(features, 'enemies_present')):
        total += r_cfg.get('attack_hit', 0.2)

    return total


def get_transition_reward(config, prev_features, features):
    """One-shot reward for scene transitions (progression events)."""
    if prev_features is None:
        return 0.0
    prev_scene = read_feature(prev_features, 'scene')
    scene = read_feature(features, 'scene')
    if not prev_scene or not scene or prev_scene == scene:
        return 0.0
    if scene == 'combat':
        # Entering combat from dialogue/menu/unknown = progression into a fight
        return config.get('rewards', {}).get('combat_start', 1.0)
    return 0.0


def get_terminal_reward(config, features):
    """Reward applied once at the transition into a terminal state."""
    r_cfg = config.get('rewards', {})
    total = 0.0
    if bool(read_feature(features, 'player_dead')):
        total += r_cfg.get('death', -5.0)
    if read_feature(features, 'scene') == 'victory':
        total += r_cfg.get('victory', 10.0)
    return total


def check_episode_end(state):
    """Episode ends on death OR victory screens."""
    return bool(read_feature(state, 'player_dead', False)) or \
        read_feature(state, 'scene') == 'victory'


def ensure_gameplay(game_runner, extractor, action_delay, max_steps=60):
    """Drive menus/dialogues/death/victory screens until gameplay is reached.

    Uses the VL model's suggested advancing key via scripted_action(). This
    replaces the old fake ESC+TAB reset: it is how runs actually get started
    after a death screen and how the agent leaves the House of Hades.
    Returns True once scene is 'combat' or 'unknown'.
    """
    for _ in range(max_steps):
        features = extractor.extract(capture_and_analyze())
        if features.scene in ('combat', 'unknown') and not features.player_dead:
            return True
        key = scripted_action(features, fallback='SPACE')
        if key:
            game_runner.execute_action(key)
        time.sleep(max(action_delay, 0.15))
    logger.warning("Failed to reach gameplay after %d scripted steps", max_steps)
    return False


def run_episode(episode, game_runner, agent, config, action_delay):
    """Run one episode; returns a metrics dict.

    Pipeline note: each screen capture is used twice — as next_state of the
    current step and as the state of the following step — so learning costs
    ~1 capture/step instead of 2. The action delay is paced against elapsed
    step time rather than blind-slept.
    """
    max_actions = config.get('agent', {}).get('max_actions_per_episode', 500)
    startup_settle = config.get('agent', {}).get('startup_settle_ms', 60000) / 1000.0
    extractor = get_extractor()
    encoder = get_encoder()
    extractor.reset()
    eff_config = effective_rewards(config, agent.total_episodes)
    game_runner.focus_window()

    logger.info("Episode %d start", episode)
    # Let the game settle after launch/menu before acting
    logger.info("Waiting %.0f seconds for game to fully load...", startup_settle)
    
    # Countdown during startup with pause support
    for i in range(int(startup_settle), 0, -1):
        if _emergency_stop_flag:
            logger.info("Episode %d aborted: emergency stop during startup", episode)
            return {'outcome': 'aborted', 'reward': 0.0, 'actions': 0, 'damage_taken': 0.0}
        if _is_paused:
            logger.info("Startup paused...")
            _pause_event.wait()  # Block until resumed
            if _emergency_stop_flag:
                return {'outcome': 'aborted', 'reward': 0.0, 'actions': 0, 'damage_taken': 0.0}
        time.sleep(1)
        logger.debug("Startup: %d seconds remaining", i)
    
    if not ensure_gameplay(game_runner, extractor, action_delay):
        logger.info("Episode %d aborted: could not reach gameplay", episode)
        return {'outcome': 'no_gameplay', 'reward': 0.0, 'actions': 0,
                'damage_taken': 0.0}

    features = extractor.extract(capture_and_analyze())
    if check_episode_end(features):
        # Death/victory surfaced while still driving menus
        return {'outcome': 'victory' if read_feature(features, 'scene') == 'victory'
                else 'dead', 'reward': 0.0, 'actions': 0, 'damage_taken': 0.0}

    episode_reward = 0.0
    actions_taken = 0
    damage_taken = 0.0
    prev_features = None
    outcome = 'timeout'

    while actions_taken < max_actions:
        # Pause support: block while paused, abort on emergency stop.
        # Without this the F12 hotkey only froze the startup countdown,
        # not actual gameplay.
        if _emergency_stop_flag:
            outcome = 'aborted'
            break
        if _is_paused:
            logger.info("Episode %d paused - press %s to resume",
                        episode, 'F12')
            _pause_event.wait()
            if _emergency_stop_flag:
                outcome = 'aborted'
                break
            logger.info("Episode %d resumed", episode)

        action = agent.select_action(features)
        step_started = time.monotonic()
        game_runner.execute_action(action)

        reward = get_reward(eff_config, features, action)
        reward += get_transition_reward(eff_config, prev_features, features)
        prev_features = features

        next_features = extractor.extract(capture_and_analyze())
        time.sleep(max(0.0, action_delay - (time.monotonic() - step_started)))

        if agent.learning_enabled:
            done = check_episode_end(next_features)
            if done:
                reward += get_terminal_reward(eff_config, next_features)
                outcome = 'victory' if read_feature(next_features, 'scene') == 'victory' \
                    else 'dead'
            agent.update(features, action, reward, next_features, done=done)

        delta = read_feature(features, 'health_delta')
        if delta is not None and delta < 0:
            damage_taken -= delta

        episode_reward += reward
        actions_taken += 1
        logger.debug("step %d: state=%s action=%s reward=%.2f",
                     actions_taken, encoder.discretize(features), action, reward)

        features = next_features

        if check_episode_end(features):
            outcome = 'victory' if read_feature(features, 'scene') == 'victory' \
                else 'dead'
            break

    logger.info("Episode %d end: outcome=%s reward=%.2f actions=%d damage=%.0f",
                episode, outcome, episode_reward, actions_taken, damage_taken)
    return {'outcome': outcome, 'reward': round(episode_reward, 2),
            'actions': actions_taken, 'damage_taken': round(damage_taken, 1)}


def main():
    global _emergency_stop_flag
    
    parser = argparse.ArgumentParser(description="Game playing agent for Hades")
    parser.add_argument('--mode', choices=['demo', 'train', 'test', 'eval'],
                        default='demo')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--config', type=Path, default=CONFIG_FILE)
    parser.add_argument('--baseline', choices=['learned', 'random'],
                        default='learned',
                        help="Eval policy baseline (default: learned Q-table)")
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--pause-key', type=str, default='f12',
                        help="Key to toggle pause (default: f12)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    )

    config = load_config()
    if args.config != CONFIG_FILE and args.config.exists():
        with open(args.config, 'r') as f:
            config = json.load(f)

    try:
        validate_config(config)
    except ConfigError as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    # test/eval modes load a saved Q-table before starting
    agent = get_agent(config)
    if args.mode in ('test', 'eval'):
        if agent.load():
            logger.info("Loaded saved agent (%d known states)", len(agent.policy))
        else:
            logger.warning("No saved agent found; starting from scratch")
        agent.learning_enabled = False
        agent.eval_mode = True
    elif args.mode == 'demo':
        agent.learning_enabled = False
    if args.baseline == 'random':
        agent.random_baseline = True

    game_exe = config['game']['executable']
    game_runner = get_game_runner(game_exe)
    action_delay = config['agent']['action_delay_ms'] / 1000

    # Check if overlay is enabled
    show_overlay = config.get('agent', {}).get('show_status_overlay', False)
    
    title = config.get('game', {}).get('title', 'Hades')
    logger.info("Game Playing Agent for %s", title)
    logger.info("=" * 50)
    logger.info("Mode: %s | Baseline: %s", args.mode, args.baseline)
    logger.info("Episodes: %d", args.episodes)
    logger.info("Learning enabled: %s", agent.learning_enabled)
    logger.info("Pause key: %s (press to toggle pause/resume)", args.pause_key)
    logger.info("Press Ctrl+C for emergency stop")
    logger.info("Status overlay: %s", "enabled" if show_overlay else "disabled")
    logger.info("=" * 50)

    # Start status overlay if enabled
    if show_overlay:
        start_overlay()
        logger.info("Status overlay started - shows RUNNING/PAUSED state")

    if not game_runner.launch():
        logger.error("Failed to launch game!")
        if show_overlay:
            stop_overlay()
        return

    # Start pause listener in background thread
    pause_thread = threading.Thread(target=listen_for_pause_key, 
                                     args=(args.pause_key,), daemon=True)
    pause_thread.start()

    results = []
    try:
        for episode in range(1, args.episodes + 1):
            results.append(run_episode(episode, game_runner, agent, config,
                                       action_delay))

        log_summary(args.mode, args.episodes, agent, results)

        if agent.learning_enabled:
            agent.save()

    except KeyboardInterrupt:
        logger.info("Interrupted by user - triggering emergency stop")
        _emergency_stop_flag = True
        game_runner.emergency_stop()
        if show_overlay:
            update_overlay_state(paused=False)
            stop_overlay()
        if agent.learning_enabled:
            agent.save()
    finally:
        game_runner.shutdown()
        if show_overlay:
            stop_overlay()


def log_summary(mode, episodes, agent, results):
    """Aggregate per-episode metrics into a closing summary."""
    outcomes = [r['outcome'] for r in results]
    victories = outcomes.count('victory')
    deaths = outcomes.count('dead')
    timeouts = outcomes.count('timeout')
    aborted = outcomes.count('no_gameplay')
    actions = sorted(r['actions'] for r in results if r['outcome'] != 'no_gameplay')
    total_reward = sum(r['reward'] for r in results)
    total_damage = sum(r['damage_taken'] for r in results)

    logger.info("=" * 50)
    logger.info("Summary (%s, %d episodes)", mode, episodes)
    logger.info("Victories: %d | Deaths: %d | Timeouts: %d | Aborted: %d",
                victories, deaths, timeouts, aborted)
    if actions:
        logger.info("Median actions/episode: %d", statistics.median(actions))
    logger.info("Total reward: %.2f | Total damage taken: %.0f",
                total_reward, total_damage)
    logger.info("Agent stats: %s", agent.get_stats())
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
