"""Reproduce the paper's per-cell AUCs and paired comparisons for a public dataset from the released predictions.

The release assets `predictions_sint.npz` and `predictions_mbd.npz` hold, for the synthetic and MBD-mini datasets:
  label|<target>                       0/1 test labels, in test-row order
  as_of                                reference (as-of) date of each test row
  dataset, boot_seed                   dataset name and the seed of its test resamples in the paper
  <method>|<scenario>|<target>|<n>|<seed>   test scores of one run (n = number of training labels)

Rank-ensemble controls (ens_*) are not stored: they are rank averages of two stored methods, built here as in the
paper. For each cell (scenario, target, n) the script reports every method's mean AUC over seeds and every
comparison of the paper (see `refs_for`) with the quantities of `paired_stats.compare`. Benjamini-Hochberg q-values
are not reproduced, because the paper's family also contains the proprietary datasets.

    python stats/reproduce.py predictions_sint.npz --out sint.json [--check expected_sint.json]
"""
import argparse
import json
import zlib

import numpy as np

from paired_stats import auc, boot_weights, compare, rank_average

VIRTUAL = {"ens_rand": ("plat_agg+wtb_trans", "jf_rand"), "ens_tab": ("plat_agg", "plat_agg+wtb_trans"),
           "ens_agg_jf": ("plat_agg", "jf_gpt"), "ens_gru": ("plat_agg+wtb_trans", "gru")}
FIELDS = ("d", "lo", "hi", "p", "lo_s", "hi_s", "lo_t", "hi_t", "pos")


def refs_for(cand):
    """References of each comparison in the paper."""
    r = []
    if cand != "plat_agg":
        r.append("plat_agg")
    if cand in ("wtb_ens", "wtb_full", "wtb_rand", "wtb_gpt", "jf_gpt_trans", "plat_agg+wtb_trans+wtb_enc",
                "wtb_ens@field", "jf_gpt_trans@field"):
        r.append("plat_agg+wtb_trans")
    if cand == "jf_gpt":
        r.append("jf_rand")
    if cand == "wtb_full":
        r += ["wtb_rand", "wtb_plat"]
    if cand == "wtb_ens":
        r += ["jf_gpt", "ens_rand", "ens_tab", "ens_agg_jf"]
    if cand in ("ens_rand", "ens_gru"):
        r.append("plat_agg+wtb_trans")
    if cand.endswith("@field"):
        r.append(cand[:-len("@field")])
    if cand == "jf_gpt":
        r.append("plat_agg+lit_gpt")
    return r


def reproduce(path):
    z = np.load(path)
    ds, boot_seed = str(z["dataset"]), int(z["boot_seed"])
    labels = {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("label|")}
    n_test = len(next(iter(labels.values())))
    cells = {}
    for k in z.files:
        if k.count("|") == 4:
            exp, scen, tgt, n, seed = k.split("|")
            cells.setdefault((scen, tgt, int(n)), {}).setdefault(exp, {})[int(seed)] = k
    W = boot_weights(n_test, np.random.default_rng(boot_seed))
    out = {"n_test": n_test, "pos_test": {t: int(y.sum()) for t, y in labels.items()}, "cells": [], "pairs": []}
    for (scen, tgt, n), by_exp in sorted(cells.items()):
        for v, (a, b) in VIRTUAL.items():
            if a in by_exp and b in by_exp:
                by_exp[v] = {s: (by_exp[a][s], by_exp[b][s]) for s in set(by_exp[a]) & set(by_exp[b])}
        y = labels[tgt].astype(np.float32)

        def score(exp, seed):
            k = by_exp[exp][seed]
            if isinstance(k, tuple):
                return rank_average(z[k[0]].astype(np.float64), z[k[1]].astype(np.float64))
            return z[k].astype(np.float64)

        for exp, seeds in sorted(by_exp.items()):
            a = np.array([auc(score(exp, s), y) for s in sorted(seeds)])
            out["cells"].append(dict(scen=scen, tgt=tgt, n=n, exp=exp, k=len(a), auc=float(a.mean()),
                                     sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, seeds=sorted(seeds)))
        for cand in by_exp:
            for ref in refs_for(cand):
                if ref not in by_exp:
                    continue
                common = sorted(set(by_exp[cand]) & set(by_exp[ref]))
                if not common:
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{ds}|{scen}|{tgt}|{n}|{cand}|{ref}".encode()))
                res = compare([score(cand, s) for s in common], [score(ref, s) for s in common], y, W, rng)
                out["pairs"].append(dict(scen=scen, tgt=tgt, n=n, cand=cand, ref=ref, **res))
        print(f"{ds} {scen} {tgt} n={n}: {len(by_exp)} methods", flush=True)
    return out


def check(out, expected):
    """Largest absolute difference from the paper's values, over cell AUCs and comparison statistics."""
    exp_cells = {(c["scen"], c["tgt"], c["n"], c["exp"]): c for c in expected["cells"]}
    exp_pairs = {(p["scen"], p["tgt"], p["n"], p["cand"], p["ref"]): p for p in expected["pairs"]}
    worst, missing = 0.0, 0
    for c in out["cells"]:
        e = exp_cells.get((c["scen"], c["tgt"], c["n"], c["exp"]))
        if e is None:
            missing += 1
            continue
        worst = max(worst, abs(c["auc"] - e["auc"]), abs(c["sd"] - e["sd"]))
    for p in out["pairs"]:
        e = exp_pairs.get((p["scen"], p["tgt"], p["n"], p["cand"], p["ref"]))
        if e is None:
            missing += 1
            continue
        worst = max([worst] + [abs(p[f] - e[f]) for f in FIELDS])
    n_exp = len(exp_cells) + len(exp_pairs)
    n_out = len(out["cells"]) + len(out["pairs"])
    return worst, missing, n_out, n_exp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", help="predictions_<dataset>.npz from the release")
    ap.add_argument("--out", help="write the cells and comparisons to this JSON file")
    ap.add_argument("--check", help="expected_<dataset>.json from the release: report the largest difference")
    a = ap.parse_args()
    out = reproduce(a.predictions)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print(f"{len(out['cells'])} cells, {len(out['pairs'])} comparisons")
    if a.check:
        with open(a.check) as f:
            worst, missing, n_out, n_exp = check(out, json.load(f))
        print(f"largest absolute difference from the paper: {worst:.2e} ({n_out} reproduced, {n_exp} expected, "
              f"{missing} not found)")


if __name__ == "__main__":
    main()
