# Usage Guide

## Overview

The agent plays Hades through a perceive→decide→act loop with tabular
Q-learning over a bounded discrete state space. Menus and dialogues are
scripted from VL suggestions; combat is learned.

## Components

| Module | Role |
|---|---|
| `scripts/main.py` | CLI, episode loop, reward model, summaries |
| `scripts/state_features.py` | canonical `StateFeatures`, extractor, scripted menu policy |
| `scripts/state_encoder.py` | discretization to ≤540 tabular state keys |
| `scripts/rl_agent.py` | Q-learning (`Q[state][action]`), ε-greedy, save/load |
| `scripts/experience_replay.py` | ring buffer → `experiments/experiences.jsonl` |
| `scripts/action_space.py` | logical actions → keys/types/cooldowns |
| `scripts/game_runner.py` | launch, focus window, key holds + cooldowns |
| `scripts/visual_state.py` | capture + CV + freshness-aware VL merge |
| `qwen-vl-integration/` | PrintWindow capture; Qwen3-VL client |

## Command Line

```powershell
python scripts/main.py --mode demo   --episodes 5              # act, no learning
python scripts/main.py --mode train  --episodes 100            # learn + save
python scripts/main.py --mode test                             # greedy from saved Q-table
python scripts/main.py --mode eval   --episodes 20             # eval metrics summary
python scripts/main.py --mode eval   --baseline random         # random baseline
python scripts/main.py --config path/to/config.json           # alternate config
python scripts/main.py --mode train --verbose                  # per-step state keys
```

Modes:
- **demo** — perception + acting only (fresh agent).
- **train** — learning enabled; Q-table saved on exit or Ctrl+C.
- **test** — loads `config/agent.json`, acts greedily.
- **eval** — like test plus aggregate summary: victories/deaths/timeouts,
  median actions per episode, total reward and damage taken.

Baselines (`--baseline`): `learned` (default) or `random` — uniform-random
over the action set after the scripted menu policy. Compare eval runs of both
against your trained agent.

## Episode Lifecycle

1. `ensure_gameplay()` presses VL-suggested keys until scene ∈ {combat,
   unknown} — this clears death screens, advances dialogs, and exits menus.
2. The loop runs up to `agent.max_actions_per_episode` steps.
3. Episode ends when the death screen or victory screen is detected
   (`player_dead` or `scene == "victory"`), or on action cap (timeout).

## Reward System

Per step (see README table): health-delta dominant term, attack shaping,
time penalty. One-shot events: entering combat (+1), death (−5), victory (+10).
Tune scales in `config/game_config.json → rewards`.

## Configuration Highlights

```jsonc
{
  "agent": {
    "max_actions_per_episode": 500,
    "action_delay_ms": 100,       // pause between actions
    "move_hold_ms": 120           // WASD key-hold duration
  },
  "learning": {
    "learning_rate": 0.2,         // tabular-friendly alpha
    "discount_factor": 0.99,
    "epsilon_decay": 0.995        // per completed episode
  },
  "actions": {
    "ALT": {"key": "g"},          // remapped off Alt/Tab (OS focus traps)
    "TAB": {"key": "r"}
  }
}
```

## Monitoring & Data

- **Logs** — INFO shows episode boundaries/outcomes; DEBUG (`--verbose`)
  adds each step's state key/action/reward.
- **`experiments/experiences.jsonl`** — every transition with full feature
  dicts (`state`, `next_state`), reward, done flag, episode label.
- **`config/agent.json`** — `{q_table, epsilon, stats, timestamp}`.

## Tuning Tips

1. Train in batches of 50–100 episodes; ε halves roughly every 139 episodes
   at decay 0.995 — lower it for faster exploration collapse.
2. If the agent never attacks, raise `attack_hit`; if it face-tanks damage,
   raise `health_delta_scale`.
3. If movement looks weak, raise `move_hold_ms`; if inputs feel spammy,
   raise `action_delay_ms`.
4. Use `--mode eval --baseline random` as the sanity floor; the learned
   policy must clearly beat it on deaths avoided and damage taken.
5. Shaping (`attack_hit`, `combat_start`) anneals to `shaping_floor` over
   `shaping_anneal_episodes`; set the span to 0 to disable annealing.
6. `replay_batches` controls post-episode off-policy sweeps (0 disables);
   more batches propagate death/victory values backwards faster at slight
   CPU cost.

### Why not threads?

The loop blocks on game time and perception, not on learning (a tabular TD
update is microseconds). Each capture is reused as both next_state of step
*t* and state of step *t+1*, so learning costs ~1 capture per step; delays
are paced against elapsed step time. A producer/consumer architecture would
add GIL-bound threading hazards for near-zero wall-clock gain — the VL
refresher already covers the genuinely async part of perception.

## Python API

```python
import sys; sys.path.insert(0, 'scripts')

from game_runner import get_game_runner
from rl_agent import get_agent
from state_features import get_extractor
from visual_state import capture_and_analyze

runner = get_game_runner(r"D:\Games\Hades\x64\Hades.exe")
agent = get_agent({'learning': {'enabled': True}})
extractor = get_extractor()

runner.launch()
features = extractor.extract(capture_and_analyze())
action = agent.select_action(features)
runner.execute_action(action)
next_features = extractor.extract(capture_and_analyze())
agent.update(features, action, -0.01, next_features, done=False)
runner.shutdown()
agent.save()
```

## Known Limits

- Chamber-to-chamber traversal beyond menu/dialogue advancement is heuristic
  (random exploration in unknown scenes) — see plan.md Phase 2 notes.
- Victory/death detection depends on VL scene classification; without the
  VL server, episodes end mainly by timeout or pixel death detection gaps.
- Live-game behavior must be validated on the machine that owns the Steam/
  game install; the test suite covers logic only.
