"""Checks of paired_stats.py. Run with `python test_paired_stats.py` or pytest."""
import numpy as np

import paired_stats as ps


def brute_auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return ((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n))


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(50, 2000))
        s = rng.integers(0, 20, n).astype(float)
        y = (rng.random(n) < .3).astype(int)
        assert abs(ps.auc(s, y) - brute_auc(s, y)) < 1e-6


def test_identical_models_give_zero_difference():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < .2).astype(int)
    s = [rng.normal(size=1000) + y for _ in range(3)]
    res = ps.compare(s, s, y, ps.boot_weights(1000, rng, B=200), rng)
    assert res["d"] == 0 and res["lo"] == 0 and res["hi"] == 0 and res["p"] == 1.0


def test_better_model_is_detected():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < .2).astype(int)
    good = [y * 1.0 + rng.normal(size=3000) for _ in range(5)]
    bad = [y * 0.3 + rng.normal(size=3000) for _ in range(5)]
    res = ps.compare(good, bad, y, ps.boot_weights(3000, rng, B=500), rng)
    assert res["lo"] > 0 and res["lo_s"] > 0 and res["pos"] == 5 and res["p"] < .01


def test_group_bootstrap_resamples_whole_units():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(100), 5)
    w = ps.boot_weights(500, rng, B=50, groups=g)
    assert all((w[:, g == u].std(axis=1) == 0).all() for u in range(100))
    assert (w.sum(axis=1) == 500).all()


def test_benjamini_hochberg():
    q = ps.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
