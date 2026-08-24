import hashlib
import json
import logging
import random
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('agent.experience_replay')

class ExperienceReplayBuffer:
    """Experience Replay Buffer for RL training"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.counter = 0
        self.total_episodes = 0
        self._current_episode = 1
        self.file_path = self._save_file_path()

    def _save_file_path(self) -> Path:
        """Derive the experiences file path from the project root"""
        file_path = (
            Path(__file__).resolve().parent.parent
            / 'experiments' / 'experiences.jsonl'
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    @staticmethod
    def _state_key(state):
        """Stable cross-process key for a state (hash() is salted per run)"""
        if hasattr(state, 'to_dict'):
            state = state.to_dict()
        if isinstance(state, dict):
            payload = json.dumps(state, sort_keys=True, default=str)
        else:
            payload = repr(state)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    @staticmethod
    def _as_dict(state):
        """Storage payload: StateFeatures -> dict; dicts pass through."""
        if hasattr(state, 'to_dict'):
            return state.to_dict()
        if isinstance(state, dict):
            return state
        return {'repr': str(state)[:200]}

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer.

        States are StateFeatures objects or feature dicts (NOT raw captures,
        which contain non-serializable image data).
        """
        self.counter += 1

        experience = {
            'state_hash': self._state_key(state),
            'state': self._as_dict(state),
            'action': action,
            'reward': reward,
            'next_state_hash': self._state_key(next_state),
            'next_state': self._as_dict(next_state),
            'done': done,
            'timestamp': datetime.now().isoformat(),
            'episode': self._current_episode,
            'counter': self.counter
        }

        # Store in memory
        self.buffer.append(experience)

        # Maintain capacity
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

        # Save to file
        self._save_experience(experience)

        # Episode bookkeeping: done marks a completed episode
        if done:
            self.total_episodes += 1
            self._current_episode = self.total_episodes + 1

        return experience
        
    def _save_experience(self, experience):
        """Save experience to file"""
        try:
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(experience) + '\n')
        except Exception as e:
            logger.warning("Failed to save experience: %s", e)
            
    def sample_batch(self, batch_size=32):
        """Sample a batch for training"""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]
        
    def get_stats(self):
        """Get buffer statistics"""
        if not self.buffer:
            return {
                'empty': True,
                'size': 0,
                'capacity': self.capacity,
                'total_episodes': self.total_episodes,
                'avg_reward': 0,
                'min_reward': float('-inf'),
                'max_reward': float('inf')
            }
            
        rewards = [exp['reward'] for exp in self.buffer]
        episodes = set(exp['episode'] for exp in self.buffer)
        actions = set(exp['action'] for exp in self.buffer)
        
        return {
            'empty': False,
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_episodes': self.total_episodes,
            'unique_episodes': len(episodes),
            'avg_reward': sum(rewards) / len(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'actions_seen': list(actions)
        }
        
    def clear(self):
        """Clear buffer"""
        self.buffer = []
        self.counter = 0
        
    def get_recent_rewards(self, n=10):
        """Get last n rewards"""
        return [exp['reward'] for exp in list(self.buffer)[-n:]]
        
    def save_to_file(self, filepath=None):
        """Save all experiences to file"""
        if filepath is None:
            filepath = self.file_path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for exp in self.buffer:
                f.write(json.dumps(exp) + '\n')
                
    def load_from_file(self, filepath=None):
        """Load experiences from file"""
        if filepath is None:
            filepath = self.file_path
            
        if not Path(filepath).exists():
            return
            
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        exp = json.loads(line)
                        self.buffer.append(exp)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line: %.50s...", line)
        except Exception as e:
            logger.error("Failed to load experiences: %s", e)
            
    def to_dict(self):
        """Convert buffer to dictionary"""
        return {
            'buffer': self.buffer,
            'counter': self.counter,
            'total_episodes': self.total_episodes,
            'current_episode': self._current_episode
        }

    def from_dict(self, data):
        """Load from dictionary"""
        self.buffer = data.get('buffer', [])
        self.counter = data.get('counter', 0)
        self.total_episodes = data.get('total_episodes', 0)
        self._current_episode = data.get('current_episode',
                                         self.total_episodes + 1)

# Global buffer instance
_buffer = None

def get_buffer(capacity=10000):
    """Get or create buffer instance (capacity only applies on first creation)"""
    global _buffer
    
    if _buffer is None:
        _buffer = ExperienceReplayBuffer(capacity)
        
    return _buffer

def add_experience(state, action, reward, next_state, done):
    """Add experience to buffer"""
    buffer = get_buffer()
    return buffer.add(state, action, reward, next_state, done)

def sample_batch(batch_size=32):
    """Sample batch from buffer"""
    buffer = get_buffer()
    return buffer.sample_batch(batch_size)

def get_buffer_stats():
    """Get buffer statistics"""
    buffer = get_buffer()
    return buffer.get_stats()

def save_experiences():
    """Save experiences to file"""
    buffer = get_buffer()
    buffer.save_to_file()

def load_experiences():
    """Load experiences from file"""
    buffer = get_buffer()
    buffer.load_from_file()

def reset_buffer():
    """Reset buffer"""
    global _buffer
    if _buffer:
        _buffer.clear()
    return get_buffer()
