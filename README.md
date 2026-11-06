# Synthetic data and paired statistics for "How Much Do Foundation Models Add to Gradient Boosting on Financial Event Sequences?"

This repository holds two parts of the paper's pipeline:

- `generator/`: the generator of the synthetic credit-card dataset and the computation of its latent-information oracle;
- `stats/`: the paired, seed-aligned comparison used for every interval and count in the paper.

The test predictions of every run on the two public datasets are attached to a release, from which `stats/reproduce.py` recomputes the paper's intervals for those datasets. The models, the feature pipeline and the proprietary datasets are not released.

## Synthetic data

```bash
pip install -r requirements.txt
python generator/gen.py data/          # about 30 s and 100 MB; deterministic
python generator/oracle.py data/       # latent-information oracle on the test set
```

The generator writes five Parquet files: 48,000 clients, 24,000 of them labeled, and 6,791,185 transactions. Each client has one reference date, and only transactions before it exist. Labeled clients with reference dates from October 2025 on form the out-of-time test set (6,121 clients). The unlabeled half is meant for self-supervised pretraining. It should print:

```
transactions: 6,791,185 | clients: 48,000 (24,000 labeled) | statements: 528,000
transactions per client: median 125, p90 238
split: {'train': 17879, 'test': 6121}
y_default rate train 0.102 test 0.100
y_seq rate train 0.100 test 0.098
y_churn rate train 0.097 test 0.101
```

The three targets carry different kinds of signal:

- `y_default` depends on a latent credit risk (visible in the register and statements), recent stress and utilization.
- `y_seq` is driven by an order pattern: a bet, then a cash withdrawal within two hours, then another bet. Decoy clients have the same events days apart, so window counts alone cannot separate them.
- `y_churn` follows a gradual decline in activity.

The oracle script should print the following for the test set:

| | AUC |
|---|---|
| default | 0.777 |
| sequence trigger | 0.779 |
| churn | 0.685 |
| sequence trigger, pattern count only | 0.763 |

It is the AUC of the deterministic part of each label's logit. It is the best AUC achievable by a model that knows the latent variables, not by one that sees only transactions (see the script's docstring).

### Data dictionary

Column names and category levels are kept as in the paper's run, in Portuguese, so that the files are identical to the ones we used.

| File | Column | Meaning |
|---|---|---|
| `transacoes.parquet` | `client_id`, `ts` | client, timestamp |
| | `amount`, `installments` | amount, number of installments |
| | `is_online`, `is_international`, `declined`, `is_cash_advance` | flags |
| | `mcc_group` | merchant group: `supermercado` supermarket, `restaurante` restaurant, `combustivel` fuel, `farmacia` pharmacy, `ecommerce`, `vestuario` apparel, `eletronicos` electronics, `viagem` travel, `servicos` services, `apostas` betting, `saque` cash withdrawal, `assinaturas` subscriptions |
| | `channel` | `online`, `aproximacao` contactless, `chip` |
| | `hour` | hour of day |
| `cadastro.parquet` | `idade`, `renda_declarada`, `tempo_relacionamento_meses` | age, declared income, months as a client |
| | `limite`, `score_bureau`, `n_produtos`, `regiao` | credit limit, bureau score, number of products, region |
| `faturas.parquet` | `data_vencimento`, `valor_fatura`, `pct_pago` | statement due date, amount, share paid |
| | `pagou_minimo`, `dias_atraso`, `saldo_rotativo` | paid less than 20%, days late, revolving balance |
| `frame.parquet` | `snapshot_date` | reference date (as-of date) |
| | `pool` | `rotulado` labeled, `nao_rotulado` unlabeled (pretraining only) |
| | `split` | `train`, `test`, `unl` |
| | `y_default`, `y_seq`, `y_churn` | targets (-1 when unlabeled) |
| `latentes.parquet` | `z`, `u`, `stress_flag`, `util`, `motif_cnt`, `dis`, `dis_onset`, `decoy`, `gambler` | latent variables, for the oracle only; never a model input |

The statements and the register enter the paper's *full* scenario only.

## Paired statistics

`stats/paired_stats.py` compares a candidate with a reference model that ran with the same seeds on the same test rows. For each pair it returns the following quantities.

- **Mean difference.** The mean over seeds of the per-seed AUC difference.
- **Test-set interval.** A 95% interval from a paired bootstrap of the test rows. It uses the same resamples for both models and every seed, and averages the per-seed differences within each resample. It holds the trained models fixed, so it measures test-set uncertainty conditional on those runs.
- **p-value.** A two-sided bootstrap p-value, (1 + count)/(B + 1) on each side, doubled.
- **Seed-resampled interval.** The same interval with the seeds also resampled, which adds the variation among the training runs.
- **t interval.** A t interval over the per-seed differences.
- **Seed count.** How many seeds favour the candidate.

The module also has rank ensembles (`rank_average`) and Benjamini–Hochberg q-values (`benjamini_hochberg`). Rows must be independent units. When a unit (a user, a card) has several rows, pass `groups` to resample whole units.

```python
import numpy as np
from paired_stats import boot_weights, compare

W = boot_weights(len(y), np.random.default_rng(0), B=2000)
res = compare([cand_s0, cand_s1, cand_s2], [ref_s0, ref_s1, ref_s2], y, W, np.random.default_rng(1))
```

```bash
python stats/paired_stats.py --labels y.npy --cand c0.npy c1.npy c2.npy --ref r0.npy r1.npy r2.npy
python stats/test_paired_stats.py
```

With the paper's resampling seeds, this code reproduces the paper's intervals exactly.

## Predictions and the paper's intervals

The [v1.0 release](https://github.com/datarisk-io/fm-vs-gbdt-eval/releases/tag/v1.0) holds the test predictions of every run of the paper on the two public datasets, the synthetic dataset and MBD-mini, with their labels:

- `predictions_sint.npz` (51 MB) and `predictions_mbd.npz` (20 MB);
- `expected_sint.json` and `expected_mbd.json`, the paper's per-cell AUCs and paired comparisons for those datasets.
- `comparisons_all.json`, the paper's per-cell AUCs, paired comparisons and q-values for all four datasets, including the proprietary ones (no labels or predictions).

Each `.npz` holds the test labels in row order (`label|<target>`), the as-of date of each row (`as_of`), and one score vector per run under the key `<method>|<scenario>|<target>|<training labels>|<seed>`. Scenarios are `txn` (transactions only) and `full`. The predictions on the proprietary datasets are not released.

```bash
python stats/reproduce.py predictions_sint.npz --out sint.json --check expected_sint.json
```

The script recomputes every cell and every comparison of the paper with the paper's resampling seeds and prints the largest absolute difference from the paper's values. It takes about 35 minutes for the synthetic dataset and 15 for MBD-mini on a laptop. Benjamini–Hochberg q-values are not reproduced from the predictions, because the paper's families of comparisons also contain the proprietary datasets; the next section covers them.

## All comparisons, including the proprietary datasets

`comparisons_all.json`, in the same release, holds the per-cell AUCs and every paired comparison of the paper for all four datasets (`synthetic`, `mbd_mini`, `client_a`, `client_b`), with the fields of `paired_stats.compare` and the paper's q-value `q`. It has no labels or predictions. The Benjamini–Hochberg adjustments of the paper can be recomputed from it:

```python
import json
from paired_stats import benjamini_hochberg

S = json.load(open("comparisons_all.json"))
pairs = [p for d in S.values() for p in d["pairs"]]
# Family 1: every comparison against the baseline (1,005 comparisons), as in the q field.
base = [p for p in pairs if p["ref"] == "plat_agg"]
q = benjamini_hochberg([p["p"] for p in base])
# Family 2: the targeted comparisons of Tables 6 and 7 (transactions only, 273 comparisons).
TARGETED = {("jf_gpt", "jf_rand"), ("jf_gpt", "plat_agg+lit_gpt"), ("wtb_ens", "plat_agg+wtb_trans"),
            ("wtb_ens", "jf_gpt"), ("wtb_full", "plat_agg+wtb_trans"), ("wtb_full", "wtb_rand"),
            ("jf_gpt@field", "jf_gpt"), ("wtb_ens@field", "wtb_ens"), ("ens_tab", "plat_agg"),
            ("ens_rand", "plat_agg+wtb_trans"), ("wtb_ens", "ens_rand"), ("wtb_ens", "ens_agg_jf"),
            ("wtb_ens", "ens_tab")}
fam2 = [p for p in pairs if p["scen"] == "txn" and (p["cand"], p["ref"]) in TARGETED]
q2 = benjamini_hochberg([p["p"] for p in fam2])
```

Method keys and the names used in the paper:

| Key | Paper |
|---|---|
| `plat_agg` | Aggregates + GBDT (Agg), the baseline |
| `plat_agg+plat_tf`, `plat_agg+plat_tf_mcc`, `plat_agg+plat_mlp`, `plat_agg+plat_mlp_masked` | reconstruction autoencoders, frozen, appended to Agg |
| `plat_agg+lit_mlm`, `plat_agg+lit_gpt`, `plat_agg+lit_coles`, `plat_agg+wtb_enc` | MLM, Nep, CoLES and Mlm-Cl encoders, frozen, appended to Agg |
| `gru`, `jf_rand`, `jf_gpt`, `tabpfn_v2` | GRU, JF-rand, JF, TabPFN v2 |
| `exp`, `plat_agg+exp` | expert features alone and with Agg (synthetic only) |
| `plat_agg+wtb_trans`, `plat_agg+wtb_trans+wtb_enc` | Agg + Trans, and with Mlm-Cl |
| `jf_gpt_trans`, `wtb_ens`, `wtb_full`, `wtb_rand`, `wtb_plat`, `wtb_gpt` | JF + Trans, rank ensemble, boosting residual and its variants |
| `<key>@field` | the same method with the field encoder |

The ensemble controls of the paper (`ens_tab`, `ens_agg_jf`, `ens_rand`, `ens_gru`) are rank averages of two stored methods. `reproduce.py` builds them.

## Citation

Relvas, C. E. M. and di Bernardo, G. G. *How Much Do Foundation Models Add to Gradient Boosting on Financial Event Sequences?* arXiv preprint (2026).

## License

Apache License 2.0; see `LICENSE`.
