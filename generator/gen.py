"""Synthetic credit-card data: transactions, a customer register and monthly statements.

Generator of the synthetic dataset in "How Much Do Foundation Models Add to Gradient Boosting on Financial Event
Sequences?". Three targets carry different kinds of signal, so that no method is favoured by construction:
  y_default : default; driven by a latent credit risk (visible in the register and statements), recent stress
              (dynamics), utilization (an aggregate) and a little of the sequence pattern.
  y_seq     : driven by an ORDER pattern, "bet -> cash withdrawal within 2 h -> bet". Decoy clients have the same
              bets and withdrawals far apart in time, so window counts alone cannot separate them.
  y_churn   : gradual decline of activity in the last months (temporal dynamics).

Strictly point in time: only transactions BEFORE each client's reference date exist; the targets are drawn from
latent variables. The output is deterministic given the seeds below. Column names and category levels are kept
as in the paper's run (Portuguese); README.md has the data dictionary.

Usage: python gen.py [output_dir]   (default: ./data next to this file)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

N_LAB, N_UNL = 24_000, 24_000
N = N_LAB + N_UNL
SNAPS = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
OOT_START = pd.Timestamp("2025-10-01")

MCC = np.array(["supermercado", "restaurante", "combustivel", "farmacia", "ecommerce", "vestuario",
                "eletronicos", "viagem", "servicos", "apostas", "saque", "assinaturas"])
I = {m: i for i, m in enumerate(MCC)}
BASE_SHARE = np.array([.24, .14, .10, .07, .13, .07, .04, .03, .08, .02, .02, .06])
AMT_MED = np.array([120, 60, 180, 70, 150, 180, 900, 800, 150, 80, 400, 40], float)
AMT_SIG = np.array([.7, .6, .4, .6, .9, .7, .8, .9, .8, 1.0, .6, .4])
P_INST = np.array([.03, .01, .01, .02, .30, .35, .70, .50, .10, .0, .0, .0])
P_ONLINE = np.array([.05, .08, .02, .05, 1.0, .25, .45, .70, .40, .95, .0, 1.0])
P_INTL = np.array([.002, .003, .0, .001, .08, .02, .03, .30, .01, .05, .01, .05])
INST_CHOICES = np.array([2, 3, 4, 5, 6, 8, 10, 12])
DAY = 86400.0

sig = lambda x: 1 / (1 + np.exp(-x))


def hours(n, mcc_idx, r):
    """Hour of day: peaks at lunch and in the evening; betting pulls towards late night."""
    h = np.where(r.random(n) < .5, r.normal(12.5, 2.0, n), r.normal(19.5, 2.5, n))
    night = (mcc_idx == I["apostas"]) & (r.random(n) < .55)
    h = np.where(night, r.normal(23.5, 1.5, n), h)
    return np.mod(h, 24)


# ── latent variables per client ─────────────────────────────────────
z = rng.normal(size=N)                       # credit risk
u = rng.normal(size=N)                       # compulsive propensity (independent of z)
income = np.exp(np.log(4500) + .6 * rng.normal(size=N) - .25 * z)
limit = np.round(income * rng.uniform(.8, 2.5, N), -2)
lam = np.exp(np.log(.33) + .5 * rng.normal(size=N) + .15 * np.log(income / 4500))   # transactions per day
gambler = rng.random(N) < sig(-2.5 + .8 * z + .6 * u)
stress = rng.random(N) < sig(-2.0 + 1.1 * z)
stress_onset = rng.uniform(5, 120, N)                     # days before the reference date
dis = rng.random(N) < sig(-2.2 + .3 * rng.normal(size=N))  # churn (gradual disengagement)
dis_onset = rng.uniform(20, 100, N)
chaser = rng.random(N) < sig(-2.3 + 1.5 * u)
n_motif = np.where(chaser, rng.poisson(1 + 3 * sig(u)), 0)
decoy = rng.random(N) < .12                                # same bets and withdrawals, not adjacent
n_decoy = np.where(decoy, rng.poisson(1 + 3 * sig(rng.normal(size=N))), 0)

lab = np.r_[np.ones(N_LAB, bool), np.zeros(N_UNL, bool)]
snap = SNAPS[rng.integers(0, len(SNAPS), N)]
snap = np.where(~lab, SNAPS[rng.integers(0, 9, N)], snap)  # unlabeled pool only in the training period
snap = pd.DatetimeIndex(snap)
snap_s = snap.values.astype("datetime64[s]").astype(np.int64).astype(float)

pref = rng.dirichlet(BASE_SHARE * 40, N)
pref[gambler, I["apostas"]] *= 6
pref /= pref.sum(1, keepdims=True)

cols = {k: [] for k in ["cid", "t", "mcc", "amount", "inst", "online", "intl", "declined"]}
motif_cnt = np.zeros(N)

for c in range(N):
    r = np.random.default_rng(1000 + c)
    S = snap_s[c]
    n = r.poisson(lam[c] * 365)
    ago = np.sort(r.uniform(0, 365, n))[::-1]           # days before the reference date (oldest first)
    if dis[c]:                                          # churn: the share of kept events decays from 1 to .15
        keep = np.ones(n, bool)
        late = ago < dis_onset[c]
        frac = ago[late] / dis_onset[c]
        keep[late] = r.random(late.sum()) < (.15 + .85 * frac)
        ago = ago[keep]; n = len(ago)
    p = np.tile(pref[c], (n, 1))
    in_stress = stress[c] & (ago < stress_onset[c])
    if in_stress.any():
        p[in_stress, I["saque"]] *= 5
        p[in_stress, I["apostas"]] *= 2
        p[in_stress, I["supermercado"]] *= .7
        p[in_stress] /= p[in_stress].sum(1, keepdims=True)
    m = (p.cumsum(1) > r.random((n, 1))).argmax(1)
    t = S - ago * DAY
    extra_t, extra_m = [], []
    for _ in range(n_motif[c]):                         # pattern: bet -> withdrawal (<= 2 h) -> bet, last 150 days
        t0 = S - r.uniform(1, 150) * DAY
        extra_t += [t0, t0 + r.uniform(5, 120) * 60, t0 + r.uniform(125, 240) * 60]
        extra_m += [I["apostas"], I["saque"], I["apostas"]]
    for _ in range(n_decoy[c]):                         # decoy: the same events, days apart
        t0 = S - r.uniform(3, 150) * DAY
        extra_t += [t0, t0 - r.uniform(2, 20) * DAY, t0 - r.uniform(21, 40) * DAY]
        extra_m += [I["apostas"], I["saque"], I["apostas"]]
    if extra_t:
        et = np.array(extra_t); ok = et < S - 60
        t = np.r_[t, et[ok]]; m = np.r_[m, np.array(extra_m)[ok]]
    motif_cnt[c] = n_motif[c]
    n = len(t)
    if n == 0:
        continue
    o = np.argsort(t); t = t[o]; m = m[o]
    st = stress[c] & (t > S - stress_onset[c] * DAY)
    amt = np.exp(np.log(AMT_MED[m] * (income[c] / 4500) ** .4) + AMT_SIG[m] * r.normal(size=n))
    amt = np.where(st & (m == I["supermercado"]), amt * .7, amt)
    p_inst = np.clip(P_INST[m] * np.where(st, 2.0, 1.0), 0, .95)
    inst = np.where(r.random(n) < p_inst, r.choice(INST_CHOICES, n), 1)
    online = (r.random(n) < P_ONLINE[m]).astype(np.int8)
    intl = (r.random(n) < P_INTL[m]).astype(np.int8)
    p_dec = .02 + .03 * sig(z[c]) + np.where(st, .12, 0)
    dec = (r.random(n) < p_dec).astype(np.int8)
    retry = dec.astype(bool) & (r.random(n) < .6)       # retry after a decline: smaller, minutes later
    if retry.any():
        k = retry.sum()
        t = np.r_[t, t[retry] + r.uniform(60, 600, k)]
        m = np.r_[m, m[retry]]; amt = np.r_[amt, amt[retry] * r.uniform(.5, .8, k)]
        inst = np.r_[inst, inst[retry]]; online = np.r_[online, online[retry]]
        intl = np.r_[intl, intl[retry]]; dec = np.r_[dec, np.zeros(k, np.int8)]
        keep = t < S
        t, m, amt, inst, online, intl, dec = (a[keep] for a in (t, m, amt, inst, online, intl, dec))
    cols["cid"].append(np.full(len(t), c, np.int32)); cols["t"].append(t)
    cols["mcc"].append(m.astype(np.int8)); cols["amount"].append(np.round(amt, 2))
    cols["inst"].append(inst.astype(np.int8)); cols["online"].append(online)
    cols["intl"].append(intl); cols["declined"].append(dec)

tx = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
tx = tx.sort_values(["cid", "t"], kind="stable").reset_index(drop=True)
tx["ts"] = pd.to_datetime(tx["t"], unit="s")
hr = hours(len(tx), tx["mcc"].values, rng)
tx["ts"] = tx["ts"].dt.normalize() + pd.to_timedelta(hr * 3600, unit="s")
# chained events (pattern, retry) keep their original time on both sides
orig = pd.to_datetime(tx["t"], unit="s")
_g = tx.groupby("cid")["t"]
chain = (_g.diff().fillna(1e9).values < 3 * 3600) | (_g.diff(-1).abs().fillna(1e9).values < 3 * 3600)
tx.loc[chain, "ts"] = orig[chain]
tx = tx.drop(columns="t")
snap_of = pd.Series(snap, index=np.arange(N))
tx = tx[tx["ts"] < snap_of.loc[tx["cid"]].values].copy()
tx = tx.sort_values(["cid", "ts"], kind="stable").reset_index(drop=True)
tx["mcc_group"] = MCC[tx["mcc"].values]
tx["is_cash_advance"] = (tx["mcc_group"] == "saque").astype(np.int8)
tx["hour"] = tx["ts"].dt.hour + tx["ts"].dt.minute / 60
tx["channel"] = np.where(tx["online"] == 1, "online", np.where(rng.random(len(tx)) < .5, "aproximacao", "chip"))
tx = tx.rename(columns={"cid": "client_id", "inst": "installments", "online": "is_online",
                        "intl": "is_international"})


def calibrate(logit_wo_b, rate=.10):
    lo, hi = -15, 15
    for _ in range(60):
        b = (lo + hi) / 2
        if sig(logit_wo_b + b).mean() > rate: hi = b
        else: lo = b
    return b


tx_recent = tx[tx["ts"] >= snap_of.loc[tx["client_id"]].values - pd.Timedelta(days=30)]
spend30 = tx_recent[tx_recent["declined"] == 0].groupby("client_id")["amount"].sum().reindex(range(N), fill_value=0).values
util = np.minimum(spend30 / limit, 1.5)
stress_flag = (stress & (stress_onset < 90)).astype(float)
e = rng.normal(0, .5, N)
lg_def = .85 * z + 1.1 * stress_flag + .35 * (motif_cnt > 0) + .8 * util + e
lg_seq = .9 * np.minimum(motif_cnt, 4) + .25 * z + rng.normal(0, .4, N)
sev = np.clip(dis_onset / 100, .3, 1)               # earlier onset -> more visible decline
lg_churn = 4.0 * dis * sev + .4 * rng.normal(size=N)
y = {}
for name, lg in [("y_default", lg_def), ("y_seq", lg_seq), ("y_churn", lg_churn)]:
    b = calibrate(lg[lab])
    y[name] = (rng.random(N) < sig(lg + b)).astype(np.int8)

frame = pd.DataFrame({"client_id": np.arange(N), "snapshot_date": snap,
                      "pool": np.where(lab, "rotulado", "nao_rotulado")})
frame["split"] = np.where(~lab, "unl", np.where(frame["snapshot_date"] >= OOT_START, "test", "train"))
for k, v in y.items():
    frame[k] = np.where(lab, v, -1)
lat = pd.DataFrame({"client_id": np.arange(N), "z": z, "u": u, "stress_flag": stress_flag, "util": util,
                    "motif_cnt": motif_cnt, "dis": dis.astype(int), "dis_onset": dis_onset, "decoy": decoy.astype(int), "gambler": gambler.astype(int)})

cad = pd.DataFrame({
    "client_id": np.arange(N),
    "idade": np.clip(rng.normal(40, 12, N) - 2 * z, 18, 80).round(),
    "renda_declarada": np.round(income * np.exp(rng.normal(0, .25, N)), -1),
    "tempo_relacionamento_meses": rng.integers(3, 121, N),
    "limite": limit,
    "regiao": rng.choice(["SE", "S", "NE", "CO", "N"], N, p=[.45, .15, .25, .08, .07]),
    "score_bureau": np.clip(650 - 75 * z + rng.normal(0, 70, N), 300, 1000).round(),
    "n_produtos": rng.poisson(1.5, N),
})

tx["month_end"] = tx["ts"].dt.to_period("M").dt.to_timestamp("M") + pd.Timedelta(days=1)
spend_m = tx[tx["declined"] == 0].groupby(["client_id", "month_end"])["amount"].sum()
rows = []
for c in range(N):
    S = snap[c]
    months = pd.date_range(S - pd.DateOffset(months=12), S - pd.DateOffset(months=1), freq="MS")
    revolv = 0.0
    r = np.random.default_rng(500_000 + c)
    for mth in months:
        due = mth + pd.DateOffset(months=1)               # due date: start of the next month
        if due >= S:
            continue
        bill = float(spend_m.get((c, due), 0.0)) + revolv * 1.12
        post = stress[c] and (S - due).days < stress_onset[c]
        mu = sig(2.0 - .9 * z[c] - 1.5 * post)
        paid_ratio = float(np.clip(r.beta(mu * 6 + .1, (1 - mu) * 6 + .1), 0, 1)) if bill > 0 else 1.0
        late = r.random() < sig(-3.0 + .9 * z[c] + 1.5 * post)
        rows.append((c, due, round(bill, 2), round(paid_ratio, 3), int(paid_ratio < .2),
                     int(r.integers(3, 41)) if late else 0, round(bill * (1 - paid_ratio), 2)))
        revolv = bill * (1 - paid_ratio)
fat = pd.DataFrame(rows, columns=["client_id", "data_vencimento", "valor_fatura", "pct_pago",
                                  "pagou_minimo", "dias_atraso", "saldo_rotativo"])

tx = tx.drop(columns=["month_end", "mcc"])
tx.to_parquet(OUT / "transacoes.parquet", index=False)
cad.to_parquet(OUT / "cadastro.parquet", index=False)
fat.to_parquet(OUT / "faturas.parquet", index=False)
frame.to_parquet(OUT / "frame.parquet", index=False)
lat.to_parquet(OUT / "latentes.parquet", index=False)

L = frame[frame.pool == "rotulado"]
print(f"transactions: {len(tx):,} | clients: {N:,} ({N_LAB:,} labeled) | statements: {len(fat):,}")
print("transactions per client: median %.0f, p90 %.0f" % tuple(np.percentile(tx.groupby('client_id').size(), [50, 90])))
print("split:", L.split.value_counts().to_dict())
for k in y:
    print(k, "rate train %.3f test %.3f" % (L[L.split == 'train'][k].mean(), L[L.split == 'test'][k].mean()))
