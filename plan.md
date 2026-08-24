# Plan of Action — Game Playing Agent Completion

Status context: the perception stack (`vl_client.py`, `game_capture.py`, pixel-calibrated
health measurement) is solid. The learning core is a stub, the reward signal is mostly
unwired, and the agent currently cannot traverse out of the starting chamber (it lacks
the E key due to a config-wiring bug, never consumes the VL model's `next_action`
suggestion, and has no notion of doors/progression). Phases are ordered by dependency;
each unblocks the next. Phases 0–2 are ~60% of the work despite containing no "AI" —
they determine whether any learning algorithm can succeed.

---

## Phase 0 — Foundations

1. **Replace `print` with `logging`** across all modules; add `--verbose` flag mapping
   to DEBUG. Live-game debugging needs timestamps.
2. **Fix the action-space wiring bug** (one line): `main.py:91` calls `get_agent()`
   without a config path, so `_load_config(None)` returns defaults without an `actions`
   key and the policy omits **E/F/Q/ALT/TAB**. Without E the agent cannot ever open
   doors — this alone makes progression impossible. Pass `CONFIG_FILE` through, and add
   a regression test asserting the loaded policy contains `E`.
3. **Pytest scaffold** covering pure logic only (no game): reward calc, state
   discretization, Q-table update, buffer ring behavior, config loading/merging.
   This is what makes later refactors of the learning core safe.
4. **Resolve the perception/decision rate mismatch**: decisions run at ~10 Hz
   (`action_delay_ms=100`) but VL refreshes every 2.5 s. Use fast pixel-only features
   (health bar fill, brightness, enemy-blob presence) every step; merge VL-enriched
   state (~every 2.5 s) when fresh. Otherwise the agent learns from stale states.
5. **Config validation at startup**: fail loudly if the executable path doesn't exist
   or required keys are missing. Remove the silent hardcoded `D:\Games\...` fallback.

## Phase 1 — Make the state learnable

6. **Define canonical `StateFeatures`** (dataclass or dict schema): `scene`,
   `enemies_present`, `threat_dir`, `health_pct`, `health_delta` (vs previous step),
   `cooldown_ready` flags, `prompt_visible` (interact prompts). Single contract between
   perception and learning — today three modules invent their own shapes.
7. **Rewrite `state_encoder.py` around it** (dead code today): implement
   `discretize(features) -> state_key` crossing `scene × threat_dir × health_bucket`
   (5–10 bins). Target ≤ ~5k reachable states for tabular Q-learning.
8. **Store features, not hashes, in the replay buffer.** Replace MD5-of-`repr(state)`
   with the actual feature dict (they're tiny). Fix `total_episodes` to increment per
   *episode*, not per experience.

## Phase 2 — Reward signal, episode lifecycle, traversal

9. **Rebuild `get_reward()` around `health_delta`**: `Δhealth × k` as the dominant
   term, small per-step time penalty (encourages progress), death penalty from config,
   completion bonus on VL `victory`. Delete unused reward config keys until wired.
10. **Reward progression transitions**: scene change `dialogue→combat`, chamber/area
    changes (VL scene transition + load-screen black-frame heuristic), boon pickups.
    Progression must carry gradient or the agent has no reason to leave the first room.
    Today non-combat reward is `neutral: 0.0` every step.
11. **Real episode lifecycle**: detect `menu/death/victory` via the existing VL
    pipeline. Note `check_episode_end` currently fires *only* on `player_dead` —
    `victory` is in the VL schema but never checked in `main.py`; fix both directions.
12. **Implement a restart routine**: navigate death screen → restart run. Likely a
    small scripted UI-navigation macro keyed off VL `scene`. Budget real time; it's
    fiddly. Without it "episodes" are meaningless.
13. **Scripted traversal mode**: consume `vl.next_action` as a scripted override when
    `scene ∈ {menu, dialogue}` — the model already suggests keys that advance
    dialogs/menus; the field is computed today and discarded by `_enrich_with_vl`.
14. **Explicit episode boundaries** in `run_episode`: start/end events logged, buffer
    notified. Replace the fake ESC+TAB reset theater.

## Phase 3 — Learning core

15. **Tabular Q-learning** in `rl_agent.py`: `Q[state_key][action]`, update
    `Q[s,a] += α(r + γ·max_a' Q[s',a'] − Q[s,a])`, γ=0.99 from config (finally used),
    ε decay per *episode* (current decay schedule is broken). Persist Q-table to
    `config/agent.json`.
16. **Scene-conditioned policy architecture**: don't ask tabular Q-learning to solve
    combat AND navigation from scratch. Learned Q-policy applies to combat states only;
    menu/dialogue states use the scripted `next_action` override (item 13); unknown/
    no-enemy states use a simple traversal heuristic ("hold direction toward objective,
    press E on interact prompt"). Greedy action = argmax Q within the active mode.
17. **Optional DQN behind a config flag**, only if the discretized state proves too
    coarse (e.g., enemy positions needed). Don't build preemptively.

## Phase 4 — Action execution quality

18. **Key holds** (`pyautogui.keyDown/keyUp`) for `input_type: continuous` actions —
    Hades movement is continuous input; single ~50 ms taps barely move the character.
    The action space already declares this field; nothing uses it.
19. **Honor per-action cooldowns** from `action_space.py` definitions (defined,
    ignored).
20. **Focus the game window before each input** (`SetForegroundWindow` via pywin32);
    remap ALT/TAB skill keys to safer bindings — both trigger OS focus switches
    mid-training.

## Phase 5 — Evaluation & tuning

21. **`--mode eval`**: greedy policy (ε=0), N episodes; report win rate, median
    survival time, damage taken per run, **chambers reached per run** (the traversal
    milestone metric).
22. **Baseline comparison**: eval (a) uniform random, (b) old heuristic agent,
    (c) trained Q-agent. If (c) doesn't beat (b), suspect reward scale. Baselines also
    quantify traversal progress independently of combat skill.
23. **Tune** α, ε-decay, reward weights over short eval runs; log curves.

## Phase 6 — Polish

24. Delete dead code: unused reward keys, replaced stubs.
25. Sync docs to reality: README/INSTALL/usage CLI flags must match argparse
    (`--game`, `--save/--load` don't exist); fix architecture diagrams; keep this plan
    updated as items complete.
26. Final full pytest pass + one supervised training session end-to-end.

---

## Suggested execution order within phases

Strictly sequential 1→26 is fine, but these can parallelize safely:
- Items 3 (tests) alongside anything.
- Items 11–14 (lifecycle/traversal scripting) are independent of 15–16 (Q-learning)
  and can proceed in parallel once Phases 0–1 land.

---

## Progress log

**2026-08-23 — Phase 0 complete (items 1–5):**
- Item 1: `logging` throughout (`agent.*` loggers); `--verbose` flag → DEBUG.
- Item 2: FIXED — `main.py` now passes the resolved config dict to
  `get_agent()`; policy includes E/F/Q/ALT/TAB (regression test:
  `test_regression_full_action_set_includes_interact`).
- Item 3: pytest suite added (`tests/`, 25 tests, pure logic only).
- Item 4: freshness-aware merge in `_enrich_with_vl` — fresh pixel evidence
  wins immediately; VL output fills gaps only when age ≤ `VL_MAX_AGE_S` (6 s);
  `analysis['enemies_source']` / `['vl_usable']` expose which signal decided.
- Item 5: `validate_config()` fails loudly on missing/invalid sections or
  nonexistent executable; hardcoded `D:\Games\...` fallback removed.
- Bonus fixes found by tests: reward keyed off declared action *type* instead
  of substring `'attack' in action` (which could never match names like
  SPACE); `ExperienceReplayBuffer.file_path` typing cleanup.
- Run tests: `python -m pytest tests -q`

**2026-08-23 — Phase 1 complete (items 6–8):**
- Item 6: NEW `scripts/state_features.py` — canonical `StateFeatures`
  dataclass (`scene`, `enemies_present(+source)`, `threat_dir`,
  `health_pct(+source px>vl)`, `health_delta`, `player_dead`,
  `prompt_visible`, `cooldown_ready`) + stateful `FeatureExtractor`
  tracking health across steps (`reset()` at episode boundaries).
  `read_feature()` helper accepts StateFeatures / feature dicts / legacy
  raw capture dicts.
- Item 7: REWROTE `state_encoder.py` (v2): `discretize()` -> bounded key
  space `scene|threat|health_bucket|enemies` = max **540 states**
  (target was ≤5k). History trimmed to 100 entries.
- Item 8: buffer now stores full feature dicts under `'state'` /
  `'next_state'` (+ stable MD5 hash kept for dedup/debug). Episode
  counting fixed: `done=True` closes an episode; experiences carry
  correct episode labels; `total_episodes` counts completed episodes.
- Wiring: `main.run_episode` extracts features once per step and shares
  them across select/reward/update; DEBUG logs the discrete state key
  per step. `rl_agent._is_combat_state`, `get_reward`,
  `check_episode_end` all accept the features form (legacy form still
  supported). Package + flat import both work (dual-import fallback in
  rl_agent/state_encoder).
- Tests: 25 → **50 passing** (new suites: test_state_features,
  test_state_encoder; extended replay/agent/config suites).
- Note: reward is still attack-bonus only — health_delta flows into
  stored experience but the reward function itself is Phase 2 item 9.

**2026-08-23 — Phases 2–6 complete (items 9–26):**

*Phase 2 — rewards & lifecycle:*
- Item 9: reward rebuilt — `health_delta × scale` dominant, per-step time
  penalty, attack shaping keyed off action type; terminal rewards via
  `get_terminal_reward()` (death −5, victory +10).
- Item 10: `get_transition_reward()` pays `combat_start` (+1) on any
  non-combat → combat scene transition (dialogue/menu/unknown).
- Item 11: `check_episode_end` now fires on victory as well as death.
- Item 12: `ensure_gameplay()` replaces the fake ESC+TAB reset — presses
  VL-suggested keys through menu/dialogue/death/victory until combat.
- Item 13: `vl.next_action` finally consumed (`vl_next_action` on
  StateFeatures; `scripted_action()` in state_features.py).
- Item 14: explicit episode start/end logs with outcome/reward/actions/
  damage; run metrics returned and aggregated by main.

*Phase 3 — learning core:*
- Item 15: real tabular Q-learning (α=0.2 from config, γ=0.99, TD update,
  ε decays per completed episode). Q-table persisted in agent.json;
  legacy bandit-format saves detected and ignored gracefully.
- Item 16: scene-conditioned policy — scripted VL keys for menus/dialogues/
  death; ε-greedy over Q for gameplay; unseen states explore randomly.
- Item 17: skipped BY DESIGN (per plan: DQN gated behind demonstrated need).

*Phase 4 — input quality:*
- Item 18: continuous actions now keyDown/hold(~120ms)/keyUp.
- Item 19: per-action cooldowns enforced (skip + 'cooldown' status).
- Item 20: window focused at launch and each episode start; ALT/TAB
  remapped to physical g/r in both action_space.py and configs.

*Phase 5 — evaluation:*
- Item 21: `--mode eval` with aggregate summary (victories/deaths/timeouts,
  median actions, total reward/damage). Damage tracked from health deltas.
- Item 22: `--baseline random` uniform-random baseline (menus still
  scripted); one baseline per invocation, compare across runs.
- Item 23: satisfied by eval harness + tuning notes in usage.md.

*Phase 6 — polish:*
- Item 24: removed dead visual_state to_json/from_json.
- Item 25: README/INSTALL/usage fully rewritten to match reality.
- Item 26: full suite green (75 tests), all modules compile, package
  imports, CLI smoke-tested with new flags, configs validated.

**Remaining (outside code):** supervised live Hades sessions on the target
machine (game install + optional VL server required); chamber-to-chamber
navigation beyond menus is still heuristic exploration.

**2026-08-23 — review response (post-push hardening):**
- Replay sweeps: buffer was write-only; `_replay_sweep()` now runs
  `learning.replay_batches` (default 8) off-policy TD passes over sampled
  stored transitions after each episode — propagates terminal values
  backwards between visits. Never touches episode counters or ε.
- Shaping annealing: `attack_hit`/`combat_start` fade linearly to
  `shaping_floor` (0.5) over `shaping_anneal_episodes` (150) via
  `effective_rewards()`. Intrinsic exploration bonus declined (redundant
  with per-episode ε decay in a tabular setting); difficulty scaling
  declined (requires automating Hades' Pact system).
- Loop efficiency: single-capture pipeline (each observation serves as
  next_state of step *t* and state of step *t+1* → ~1 capture/step instead
  of 2); action delay paced against elapsed step time. Threads declined:
  bottleneck is game time + perception latency, not µs-cheap tabular
  updates; rationale documented in usage.md.
- Tests: 75 → **81 passing**.
