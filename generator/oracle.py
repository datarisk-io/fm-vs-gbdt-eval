"""Latent-information oracle of the synthetic dataset, computed from the generator without fitting a model.

Each target is Bernoulli(sigmoid(deterministic part + Gaussian noise + b)), so the probability of the label given the
latent variables is monotone in the deterministic part, and the test AUC of that part is the best AUC achievable by a
model that knows the latent variables. Some of them (the credit risk z, and the credit limit inside utilization) are
in the register rather than in the transactions, so the best AUC achievable from transactions alone is lower. For the
sequence trigger the pattern can be counted exactly from the transactions: the AUC of the pattern count (capped at 4)
alone is a reference for what the transactions reveal.

Usage: python oracle.py <output_dir of gen.py> [oracle.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def auc(s, y):
    r = pd.Series(s).rank().values
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


src = Path(sys.argv[1])
fr = pd.read_parquet(src / "frame.parquet")
lat = pd.read_parquet(src / "latentes.parquet").set_index("client_id").loc[fr.client_id].reset_index()
te = (fr.pool == "rotulado").values & (fr.split == "test").values
m = lat.motif_cnt.values
det = {"y_default": .85 * lat.z + 1.1 * lat.stress_flag + .35 * (m > 0) + .8 * lat.util,
       "y_seq": .9 * np.minimum(m, 4) + .25 * lat.z,
       "y_churn": 4.0 * lat.dis * np.clip(lat.dis_onset / 100, .3, 1)}
out = {"latent": {t: auc(np.asarray(s)[te], fr[t].values[te]) for t, s in det.items()},
       "seq_pattern_only": auc(np.minimum(m, 4)[te], fr.y_seq.values[te])}
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
