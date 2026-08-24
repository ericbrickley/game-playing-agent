import json
import logging
import random
from datetime import datetime
from pathlib import Path

try:
    from state_features import read_feature, scripted_action
except ImportError:  # imported as part of the 'scripts' package
    from .state_features import read_feature, scripted_action

try:
    from state_encoder import get_state_key
except ImportError:
    from .state_encoder import get_state_key

logger = logging.getLogger('agent.rl')


class SimpleRLAgent:
    """Tabular Q-learning agent over discretized StateFeatures.

    Policy architecture (scene-conditioned):
      - menu/dialogue/death scenes -> scripted key from the VL suggestion
        (see state_features.scripted_action); never learned.
      - gameplay states -> epsilon-greedy over Q[state_key][action].
    Unseen states fall back to uniform random exploration.

    `self.policy` holds the Q-table: {state_key: {action: q_value}}.
    """

    def __init__(self, config_source=None):
        self.config = self._load_config(config_source)
        learning = self.config.get('learning', {})
        self.learning_enabled = learning.get('enabled', True)
        self.alpha = float(learning.get('learning_rate', 0.2))
        self.gamma = float(learning.get('discount_factor', 0.99))
        self.epsilon_start = float(learning.get('epsilon_start', 1.0))
        self.epsilon_decay = float(learning.get('epsilon_decay', 0.995))
        self.epsilon_min = float(learning.get('epsilon_min', 0.01))
        self._epsilon = self.epsilon_start

        actions = self.config.get('actions', {})
        self.actions = list(actions) if actions else \
            ['W', 'A', 'S', 'D', 'SPACE', 'SHIFT']

        self.policy = {}          # Q-table
        self.buffer = None        # lazy ExperienceReplayBuffer
        self.episode_rewards = []
        self._current_reward = 0.0
        self.win_count = 0
        self.total_episodes = 0
        self.eval_mode = False         # force greedy (no exploration)
        self.random_baseline = False   # uniform-random baseline mode

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self, source):
        """Load configuration from a dict or a JSON file path."""
        default_config = {
            'learning': {
                'enabled': True,
                'buffer_size': 10000,
                'batch_size': 32,
                'learning_rate': 0.2,
                'discount_factor': 0.99,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'epsilon_start': 1.0
            }
        }

        user_config = None
        if isinstance(source, dict):
            user_config = dict(source)
        elif source and Path(source).exists():
            try:
                with open(source, 'r') as f:
                    user_config = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not read config %s: %s", source, e)

        if not isinstance(user_config, dict):
            return default_config

        merged = dict(default_config)
        merged.update(user_config)
        return merged

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state):
        """Pick an action for the given feature state."""
        scripted = scripted_action(state)
        if scripted is not None:
            return scripted if scripted in self.actions else \
                random.choice(self.actions)

        if self.random_baseline:
            return random.choice(self.actions)

        if not self.eval_mode and random.random() < self._get_epsilon():
            return random.choice(self.actions)

        q_row = self.policy.get(get_state_key(state))
        if q_row:
            best = max(q_row.values())
            return random.choice([a for a, v in q_row.items() if v == best])
        return random.choice(self.actions)

    def _get_epsilon(self):
        """Current exploration rate; decays once per completed episode."""
        return max(self.epsilon_min, self._epsilon)

    def _decay_epsilon(self):
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Learning (tabular Q-learning)
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One TD step: Q[s,a] += alpha * (r + gamma * max Q[s'] - Q[s,a])."""
        if not self.learning_enabled:
            return

        if self.buffer is None:
            self.buffer = self._init_buffer()
        self.buffer.add(state, action, reward, next_state, done)

        s_key = get_state_key(state)
        q_row = self.policy.setdefault(s_key, {})
        old = q_row.get(action, 0.0)

        if done:
            target = reward
        else:
            next_row = self.policy.get(get_state_key(next_state), {})
            target = reward + self.gamma * max(next_row.values()) if next_row \
                else reward
        q_row[action] = old + self.alpha * (target - old)

        self._current_reward += reward
        if done:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
            self.total_episodes += 1
            if self.episode_rewards[-1] > 0:
                self.win_count += 1
            self._decay_epsilon()
            self._replay_sweep()

    def _replay_sweep(self):
        """Off-policy replay: re-apply TD updates over sampled stored
        transitions. Online tabular Q-learning uses each visit once; sweeps
        propagate terminal/backward value between visits. Deliberately never
        touches episode counters or epsilon."""
        batches = int(self.config.get('learning', {}).get('replay_batches', 8))
        if batches <= 0 or self.buffer is None or not self.buffer.buffer:
            return
        for _ in range(batches):
            for exp in self.buffer.sample_batch(32):
                s_row = self.policy.setdefault(
                    get_state_key(exp['state']), {})
                old = s_row.get(exp['action'], 0.0)
                target = exp['reward']
                if not exp.get('done'):
                    n_row = self.policy.get(
                        get_state_key(exp['next_state']), {})
                    if n_row:
                        target += self.gamma * max(n_row.values())
                s_row[exp['action']] = old + self.alpha * (target - old)

    def _init_buffer(self):
        from experience_replay import ExperienceReplayBuffer
        capacity = self.config.get('learning', {}).get('buffer_size', 10000)
        return ExperienceReplayBuffer(capacity)

    # ------------------------------------------------------------------
    # Stats / persistence
    # ------------------------------------------------------------------
    def get_stats(self):
        return {
            'total_episodes': self.total_episodes,
            'wins': self.win_count,
            'win_rate': self.win_count / self.total_episodes if self.total_episodes > 0 else 0,
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0,
            'known_states': len(self.policy),
            'actions': len(self.actions),
            'learning_enabled': self.learning_enabled,
            'epsilon': round(self._get_epsilon(), 4)
        }

    def save(self, filepath=None):
        if filepath is None:
            filepath = Path(__file__).resolve().parent.parent / 'config' / 'agent.json'
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'q_table': self.policy,
            'epsilon': self._epsilon,
            'episode_rewards': self.episode_rewards,
            'stats': self.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Saved Q-table (%d states) to %s", len(self.policy), filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = Path(__file__).resolve().parent.parent / 'config' / 'agent.json'
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("No saved agent at %s", filepath)
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load agent: %s", e)
            return False

        table = data.get('q_table')
        if isinstance(table, dict) and all(
                isinstance(v, dict) for v in table.values()):
            self.policy = table
        elif isinstance(data.get('policy'), dict):
            # Legacy bandit-era save format; value tables are incompatible.
            logger.warning("Ignoring legacy policy format in %s", filepath)
            self.policy = {}

        stats = data.get('stats', {})
        self.total_episodes = int(stats.get('total_episodes', 0))
        self.win_count = int(stats.get('wins', 0))
        self.episode_rewards = data.get('episode_rewards', [])
        eps = data.get('epsilon')
        self._epsilon = float(eps) if isinstance(eps, (int, float)) \
            else self.epsilon_start
        return True

    def reset(self):
        self.policy = {}
        self.episode_rewards = []
        self._current_reward = 0.0
        self.win_count = 0
        self.total_episodes = 0
        self._epsilon = self.epsilon_start


# Global agent instance
_agent = None


def get_agent(config_source=None):
    """Get or create agent instance.

    config_source may be a path to a JSON file or an already-loaded config
    dict; passing the dict keeps the agent's action set in sync with whatever
    config main.py resolved (including --config overrides).
    """
    global _agent
    if _agent is None:
        _agent = SimpleRLAgent(config_source)
    return _agent


def select_action(state):
    agent = get_agent()
    return agent.select_action(state)


def update_agent(state, action, reward, next_state, done):
    agent = get_agent()
    return agent.update(state, action, reward, next_state, done)


def get_agent_stats():
    agent = get_agent()
    return agent.get_stats()


def save_agent():
    agent = get_agent()
    agent.save()


def load_agent():
    agent = get_agent()
    return agent.load()
