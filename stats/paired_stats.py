"""Paired, seed-aligned comparison of two scoring models on one test set: the statistics of the paper.

A candidate c and a reference r are each run with the same k seeds (same label subsample, split and initialization
seed) and score the same test rows. For the pair we report:

  d          mean over seeds of AUC_c(s) - AUC_r(s)
  lo, hi     95% percentile interval from a paired bootstrap of the test rows: B resamples, the same for both models
             and every seed; in each resample the per-seed differences are averaged (D*_b)
  p          two-sided bootstrap p-value, min(1, 2 min[(1 + #{D*_b <= 0}) / (B + 1), (1 + #{D*_b >= 0}) / (B + 1)])
  lo_s, hi_s the same interval with the seeds also resampled with replacement inside each test resample, which adds
             the variation among the k training runs (coarse with few seeds)
  lo_t, hi_t t interval over the k per-seed differences (training variation only, test set held fixed)
  pos        number of seeds in which the candidate has the higher AUC

lo/hi and p hold the trained models fixed: they measure the uncertainty from the finite test set, conditional on the
k runs. Rows must be independent units (in the paper each test client appears once); when a unit has several rows,
pass `groups` so that whole units are resampled. Rank ensembles (mean of ranks) and Benjamini-Hochberg q-values over
a family of comparisons are included.

Library use:
    W = boot_weights(len(y), np.random.default_rng(0))
    res = compare([cand_seed0, cand_seed1, ...], [ref_seed0, ref_seed1, ...], y, W, np.random.default_rng(1))
Command line:
    python paired_stats.py --labels y.npy --cand c_s0.npy c_s1.npy ... --ref r_s0.npy r_s1.npy ... [--boot 2000]
Needs numpy and pandas.
"""
import argparse
import json

import numpy as np
import pandas as pd

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def boot_weights(n, rng, B=2000, groups=None):
    """Multiplicity of each row in each of B resamples (B x n). With `groups`, whole groups are resampled."""
    if groups is None:
        w = np.zeros((B, n), dtype=np.float32)
        for b in range(B):
            np.add.at(w[b], rng.integers(0, n, n), 1)
        return w
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    w = np.zeros((B, n), dtype=np.float32)
    for b in range(B):
        cnt = np.zeros(G, dtype=np.float32)
        np.add.at(cnt, rng.integers(0, G, G), 1)
        w[b] = cnt[inv]
    return w


def auc_boot(s, y, w):
    """AUC under each row of resampling weights w (B x n): Mann-Whitney with ties counting one half, one sort.
    Sums are accumulated in float32, as in the paper; the result agrees with the exact AUC to about 1e-7."""
    order = np.argsort(s, kind="stable")
    ss, yy = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    wp = np.add.reduceat(w[:, order] * yy, starts, axis=1)
    wn = np.add.reduceat(w[:, order] * (1 - yy), starts, axis=1)
    below = np.cumsum(wn, axis=1) - wn
    num = (wp * (below + 0.5 * wn)).sum(1)
    return num / (wp.sum(1) * wn.sum(1))


def auc(s, y):
    return float(auc_boot(np.asarray(s, np.float64), np.asarray(y, np.float32), np.ones((1, len(s)), np.float32))[0])


def rank_average(*scores):
    """Rank ensemble: mean of the ranks of several score vectors, scaled to (0, 1]. No weights, not a probability."""
    return sum(pd.Series(s).rank().values for s in scores) / (len(scores) * len(scores[0]))


def compare(cand, ref, y, W, rng):
    """cand, ref: lists of score vectors aligned by seed; y: 0/1 labels; W: boot_weights(...); rng: seed resampling."""
    assert len(cand) == len(ref) >= 1
    y = np.asarray(y, np.float32)
    k, B = len(cand), W.shape[0]
    d_seed = np.array([auc(c, y) - auc(r, y) for c, r in zip(cand, ref)])
    per_seed = np.array([auc_boot(np.asarray(c, np.float64), y, W) - auc_boot(np.asarray(r, np.float64), y, W)
                         for c, r in zip(cand, ref)])                       # k x B
    boot = per_seed.mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min(((boot <= 0).sum() + 1) / (B + 1), ((boot >= 0).sum() + 1) / (B + 1)))
    pick = rng.integers(0, k, (B, k))
    hier = per_seed[pick, np.arange(B)[:, None]].mean(axis=1)
    lo_s, hi_s = np.percentile(hier, [2.5, 97.5])
    sd = d_seed.std(ddof=1) if k > 1 else 0.0
    half = T975[k - 1] * sd / np.sqrt(k) if 1 < k <= 10 else float("nan")
    return dict(k=k, d=float(d_seed.mean()), lo=float(lo), hi=float(hi), p=float(p), lo_s=float(lo_s), hi_s=float(hi_s),
                lo_t=float(d_seed.mean() - half), hi_t=float(d_seed.mean() + half), pos=int((d_seed > 0).sum()),
                d_sd=float(sd))


def benjamini_hochberg(p):
    """q-values of Benjamini-Hochberg for a family of p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, help=".npy with the 0/1 test labels")
    ap.add_argument("--cand", nargs="+", required=True, help=".npy score files of the candidate, one per seed")
    ap.add_argument("--ref", nargs="+", required=True, help=".npy score files of the reference, same seed order")
    ap.add_argument("--groups", help=".npy with a unit id per test row, to resample units instead of rows")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    y = np.load(a.labels)
    W = boot_weights(len(y), np.random.default_rng(a.seed), a.boot, np.load(a.groups) if a.groups else None)
    res = compare([np.load(f) for f in a.cand], [np.load(f) for f in a.ref], y, W, np.random.default_rng(a.seed + 1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
