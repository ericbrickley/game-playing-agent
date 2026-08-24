import json

import pytest

from experience_replay import ExperienceReplayBuffer


@pytest.fixture
def buf(tmp_path):
    b = ExperienceReplayBuffer(capacity=3)
    b.file_path = tmp_path / 'experiences.jsonl'
    return b


def test_ring_capacity_drops_oldest(buf):
    for i in range(5):
        buf.add({'n': i}, 'W', 0.1, {'n': i}, False)
    assert len(buf.buffer) == 3
    expected = [buf._state_key({'n': i}) for i in (2, 3, 4)]
    assert [e['state_hash'] for e in buf.buffer] == expected


def test_state_key_stable_across_instances():
    # hash() is salted per process; the MD5 key must be stable so saved
    # experiences stay meaningful across runs.
    a = ExperienceReplayBuffer()._state_key({'a': 1, 'b': [2, 3]})
    b = ExperienceReplayBuffer()._state_key({'a': 1, 'b': [2, 3]})
    assert a == b


def test_appends_jsonl_per_experience(buf):
    buf.add({'x': 1}, 'D', 0.5, {'x': 2}, False)
    buf.add({'x': 3}, 'A', -0.1, {'x': 4}, True)
    lines = buf.file_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec['action'] == 'D'
    assert rec['reward'] == 0.5
    assert rec['done'] is False


def test_sample_batch_sizes(tmp_path):
    b = ExperienceReplayBuffer(capacity=20)
    b.file_path = tmp_path / 'experiences.jsonl'
    for i in range(10):
        b.add({'n': i}, 'W', 0.0, {'n': i}, False)
    assert len(b.sample_batch(4)) == 4
    assert len({id(e) for e in b.sample_batch(4)}) == 4  # distinct samples
    assert b.sample_batch(32) == b.buffer  # request larger than buffer


def test_stats(buf):
    buf.add({}, 'W', 1.0, {}, False)
    buf.add({}, 'D', -1.0, {}, False)
    s = buf.get_stats()
    assert s['size'] == 2
    assert s['avg_reward'] == 0.0
    assert s['min_reward'] == -1.0
    assert s['max_reward'] == 1.0
    assert set(s['actions_seen']) == {'W', 'D'}


def test_clear_resets_counters(buf):
    buf.add({}, 'W', 1.0, {}, False)
    buf.clear()
    assert buf.buffer == []
    assert buf.counter == 0
    assert buf._current_episode == 1


def test_stores_feature_dicts_not_hashes_only(buf):
    from state_features import StateFeatures
    s = StateFeatures(scene='combat', health_pct=70)
    ns = StateFeatures(scene='combat', health_pct=65)
    buf.add(s, 'SPACE', 0.5, ns, False)
    exp = buf.buffer[0]
    assert exp['state']['scene'] == 'combat'
    assert exp['next_state']['health_pct'] == 65
    assert exp['state_hash']  # still kept for dedup/debug


def test_episode_numbering_counts_done_boundaries(buf):
    buf.add({}, 'W', 0, {}, False)   # episode 1
    buf.add({}, 'W', 0, {}, True)    # episode 1 ends
    buf.add({}, 'W', 0, {}, False)   # episode 2
    assert [e['episode'] for e in buf.buffer] == [1, 1, 2]
    assert buf.total_episodes == 1
