from state_features import FeatureExtractor, StateFeatures, health_bucket


def raw(hp_px=None, hp_vl=None, **kw):
    analysis = {}
    if hp_px is not None:
        analysis['health_pct_px'] = hp_px
    if hp_vl is not None:
        analysis['health_pct_vl'] = hp_vl
    analysis.update(kw)
    return {'analysis': analysis, 'vl': {}}


def test_health_delta_tracks_steps():
    ex = FeatureExtractor()
    f1 = ex.extract(raw(hp_px=100))
    assert f1.health_delta is None
    f2 = ex.extract(raw(hp_px=80))
    assert f2.health_delta == -20
    assert f2.health_pct == 80


def test_reset_clears_delta_tracking():
    ex = FeatureExtractor()
    ex.extract(raw(hp_px=100))
    ex.reset()
    f = ex.extract(raw(hp_px=80))
    assert f.health_delta is None


def test_pixel_health_preferred_over_vl():
    f = FeatureExtractor().extract(raw(hp_px=55, hp_vl=90))
    assert f.health_pct == 55
    assert f.health_source == 'px'


def test_vl_health_fallback():
    f = FeatureExtractor().extract(raw(hp_vl=90))
    assert f.health_pct == 90
    assert f.health_source == 'vl'


def test_no_health_visible():
    f = FeatureExtractor().extract(raw())
    assert f.health_pct is None
    assert f.health_source is None
    assert f.health_delta is None


def test_death_scene_overrides_vl():
    r = raw()
    r['vl'] = {'scene': 'menu'}
    r['analysis']['player_dead'] = True
    f = FeatureExtractor().extract(r)
    assert f.scene == 'death'
    assert f.player_dead is True


def test_invalid_threat_dir_becomes_unknown():
    r = raw()
    r['vl'] = {'threat_dir': 'diagonal'}
    assert FeatureExtractor().extract(r).threat_dir == 'unknown'


def test_enemies_passthrough():
    f = FeatureExtractor().extract(raw(enemies_present=True,
                                       enemies_source='pixels'))
    assert f.enemies_present is True
    assert f.enemies_source == 'pixels'


def test_extractor_is_idempotent_for_features():
    f = StateFeatures(scene='combat')
    assert FeatureExtractor().extract(f) is f


def test_health_buckets():
    assert health_bucket(None) == 'unknown'
    assert health_bucket(0) == 'critical'
    assert health_bucket(24.9) == 'critical'
    assert health_bucket(25) == 'low'
    assert health_bucket(50) == 'mid'
    assert health_bucket(74.9) == 'mid'
    assert health_bucket(75) == 'high'
    assert health_bucket(100) == 'high'
    assert health_bucket(-5) == 'critical'
    assert health_bucket(150) == 'high'


def test_to_dict_roundtrip_ignores_unknown_keys():
    d = StateFeatures(scene='combat').to_dict()
    d['bogus'] = 1
    f = StateFeatures.from_dict(d)
    assert f.scene == 'combat'
    assert 'bogus' not in f.to_dict()
