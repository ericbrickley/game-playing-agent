# Game Playing Agent for Hades

An autonomous bot that plays Hades on Windows: it captures the game window,
understands the scene through pixel analysis plus a local Qwen3-VL model,
and learns combat behavior with tabular Q-learning.

## Architecture

```
game-playing-agent/
├── scripts/
│   ├── main.py              # CLI, episode loop, reward model, summaries
│   ├── state_features.py    # canonical StateFeatures + extractor + scripted menu policy
│   ├── state_encoder.py     # discretization -> bounded tabular state keys (≤540)
│   ├── rl_agent.py          # tabular Q-learning agent
│   ├── experience_replay.py # ring buffer persisting feature dicts to JSONL
│   ├── action_space.py      # logical actions -> keys, types, cooldowns
│   ├── game_runner.py       # launch, window focus, key holds/cooldowns
│   └── visual_state.py      # capture + pixel CV + VL merge
├── qwen-vl-integration/
│   ├── game_capture.py      # PrintWindow client-area capture
│   └── vl_client.py         # Qwen3-VL via llama.cpp (cached, background)
├── config/                  # game_config.json, agent.json (saved Q-table)
├── tests/                   # 75 pytest cases, pure logic (no game needed)
└── plan.md                  # roadmap and progress log
```

## Quick Start

```powershell
pip install -r config/requirements.txt

# verify everything without the game
python -m pytest tests -q

# watch it act (no learning)
python scripts/main.py --mode demo --episodes 1 --verbose

# train (saves config/agent.json on exit / Ctrl+C)
python scripts/main.py --mode train --episodes 100

# evaluate a trained agent
python scripts/main.py --mode eval --episodes 20

# uniform-random baseline for comparison
python scripts/main.py --mode eval --baseline random --episodes 20
```

Optional but recommended: run a llama.cpp `llama-server` with Qwen3-VL on
`127.0.0.1:8080` (`VL_SERVER` env var to override). Without it the bot still
works on pixel-only perception but cannot classify scenes or advance dialogs.

## How It Works

Each step:

1. **Perceive** — capture the client area (PrintWindow), measure the health
   bar from pixels (calibrated), detect enemy blobs by color; a background
   thread asks Qwen3-VL for `{scene, health, threat_dir, enemies,
   next_action}` every ~2.5 s. Fresh pixels always win; VL output fills gaps
   only while young (<6 s).
2. **Extract** — reduce to `StateFeatures` (scene, enemies, threat, health %,
   health delta vs last step).
3. **Decide** — scene-conditioned policy:
   - menus/dialogues/death screens → scripted key from the VL suggestion
     (this is how runs start after death and how the House of Hades is left);
   - gameplay states → ε-greedy over `Q[state_key][action]`.
4. **Act** — movement keys are held (~120 ms) not tapped; per-action cooldowns
   are enforced; Alt/Tab are remapped to g/r to avoid OS focus traps.
5. **Learn** — TD update `Q[s,a] += α(r + γ·maxQ[s'] − Q[s,a])`, α=0.2,
   γ=0.99, ε decays per completed episode.

## Reward Model

| Term | Config key | Default | Meaning |
|---|---|---|---|
| Health change | `health_delta_scale` × Δhp | 0.1 | dominant signal |
| Attack while enemies | `attack_hit` | 0.2 | small shaping |
| Entering combat | `combat_start` | 1.0 | progression event |
| Death screen | `death` | −5.0 | terminal penalty |
| Victory screen | `victory` | 10.0 | terminal bonus |
| Every step | `time_penalty` | −0.01 | discourages dithering |

State space is bounded: `scene × threat_dir × health_band × enemies`
(≤540 states) — deliberately sized so a table beats a neural net.

## Monitoring

- Console logs: per-episode outcome/reward/actions/damage + closing summary.
- `experiments/experiences.jsonl`: one feature-dict transition per line.
- `config/agent.json`: saved Q-table, epsilon, stats.
- `--verbose` prints the discrete state key of every step.

## Status & Roadmap

See `plan.md`. Completed: foundations, canonical features, real Q-learning,
input quality, eval harness, docs. Remaining known limits are logged there —
notably that live-game validation (actual Hades sessions) must be done on the
target machine, and chamber-to-chamber navigation beyond menus is heuristic.
