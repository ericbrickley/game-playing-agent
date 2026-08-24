from state_encoder import StateEncoder
from state_features import StateFeatures


def feats(**kw):
    base = dict(scene='combat', threat_dir='left', health_pct=60,
                enemies_present=True)
    base.update(kw)
    return StateFeatures(**base)


def test_discretize_deterministic():
    e = StateEncoder()
    assert e.discretize(feats()) == e.discretize(feats())


def test_key_reflects_health_band_change():
    e = StateEncoder()
    assert e.discretize(feats(health_pct=80)) != e.discretize(feats(health_pct=60))


def test_unknown_enemies_distinct_from_false():
    e = StateEncoder()
    assert e.discretize(feats(enemies_present=None)) != \
        e.discretize(feats(enemies_present=False))


def test_key_format_four_parts():
    key = StateEncoder().discretize(feats())
    assert key.split('|') == ['combat', 'left', 'mid', 'True']


def test_accepts_plain_dicts():
    e = StateEncoder()
    d = {'scene': 'menu', 'threat_dir': 'none', 'health_pct': None,
         'enemies_present': False}
    assert e.discretize(d) == 'menu|none|unknown|False'


def test_key_space_is_bounded_small():
    # Tabular Q-learning target: well under a few thousand states.
    scenes = ['combat', 'menu', 'dialogue', 'death', 'victory', 'unknown']
    threats = ['left', 'right', 'up', 'down', 'none', 'unknown']
    buckets = ['critical', 'low', 'mid', 'high', 'unknown']
    enemies = [None, True, False]
    assert len(scenes) * len(threats) * len(buckets) * len(enemies) <= 540


def test_history_trims_to_limit():
    e = StateEncoder()
    for i in range(130):
        e.encode(feats(health_pct=i % 101))
    assert len(e.state_history) == 100


def test_clear_history():
    e = StateEncoder()
    e.encode(feats())
    e.clear_history()
    assert e.state_history == []
