# Installation Guide

## Prerequisites

1. **Python 3.9+**
2. **Windows**: input automation and window capture are Windows-specific
3. **Hades** installed locally (path configured in `config/game_config.json`)
4. Optional: **llama.cpp llama-server** running Qwen3-VL with an mmproj on
   port 8080 for full scene understanding (env var `VL_SERVER` overrides)

## Installation Steps

```powershell
cd C:\Users\ericb\game-playing-agent

# 1. Install Python dependencies
pip install -r config/requirements.txt

# 2. Point the config at your game executable
notepad config\game_config.json   # set "game.executable"

# 3. Run the test suite (no game needed)
python -m pytest tests -q

# 4. Smoke-test the CLI
python scripts/main.py --help

# 5. Dry-run against the real game (watch, no learning)
python scripts/main.py --mode demo --episodes 1 --verbose
```

Startup validates the config and fails loudly if the executable path is wrong
or required sections are missing — there is no silent fallback anymore.

## Command Line Reference

```
python scripts/main.py [--mode MODE] [--episodes N] [--config PATH]
                       [--baseline {learned,random}] [--verbose]

--mode demo    act without learning or loading a saved agent
--mode train   learn (epsilon-greedy) and save the Q-table on exit
--mode test    load config/agent.json and act greedily
--mode eval    load and evaluate greedily with aggregate metrics
--verbose      DEBUG logging incl. per-step state keys
```

## Troubleshooting

### Config error at startup
Read the message: it names the missing section/key or the nonexistent
executable path. Fix `config/game_config.json` accordingly.

### Actions not registering in-game
- The runner focuses the Hades window before each episode; make sure no
  other window steals focus mid-run.
- Increase `agent.action_delay_ms`, or `agent.move_hold_ms` if movement
  feels too weak/strong.
- Skill keys were remapped away from Alt/Tab (OS focus traps): logical
  ALT sends `g`, logical TAB sends `r` — match your in-game bindings.

### VL server unavailable
The bot logs a warning and continues on pixel-only features (health bar,
enemy blobs). Scene detection, dialogue advancement and threat direction
require the VL server; start llama-server and check `VL_SERVER`.

### Poor learning
- Ensure you train long enough for ε to decay (default half-life ~139 episodes).
- Check `experiments/experiences.jsonl` for sane rewards (health deltas present).
- Tune `learning.learning_rate` (0.1–0.5 typical for tabular) and reward scales.

## License

Educational tool. Use responsibly and respect the game's terms of service.
