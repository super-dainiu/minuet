# Minuet

A two-loss decomposition for paired snRNA + snATAC multiome integration.

```python
from minuet import Minuet, MinuetConfig, MinuetLosses
```

## Why two losses

Minuet's design is the result of an ablation that found **contrastive and CMI-BI
target distinct axes of the multiome integration problem**:

|                         | NT-Xent contrastive | CMI-BI |
|-------------------------|---------------------|--------|
| RNA↔ATAC retrieval      | **necessary + sufficient** | irrelevant |
| within-modality scIB    | irrelevant          | **necessary + sufficient** |
| supervised Region F1    | sets a ceiling      | rescues partial structure under compression |

Removing either loss leaves the other's contribution intact:

- `mid_nocontrastive`: Wang scIB Total = **0.590** (vs full mid 0.591) — same
  integration quality, but RNA↔ATAC retrieval collapses to 0.0024 (random).
- `−contrastive` ablation on the no-CMI-BI baseline shows the same pattern:
  retrieval at random floor.

## Label-free false-negative cancellation

A plain cross-modal contrastive treats every other cell in the batch as a
negative, including cells of the same biological state as the anchor. These false
negatives dominate the gradient when cell types are few, and RNA↔ATAC retrieval
degrades on low-diversity cohorts. `MinuetLosses.contrastive(..., fn_sim=τ)` drops,
from each anchor's negatives, the cells that are similar to it in **both**
modalities' shared codes (cosine > τ): likely same-state pairs. It uses only the
embeddings — no cell-type labels or counts — and the absolute-similarity gate
self-adapts: diverse cohorts trip it rarely, low-diversity cohorts often. Default
`fn_sim = 0.6`. This is the same context principle as CMI-BI's conditional donor
term: biology governs both which cells to contrast and which donor signal to
remove.

## Pareto pair (canonical recipes)

Both ship in `configs/`:

| Recipe | η (CMI-BI weight) | class_key | role |
|---|---:|---|---|
| `minuet_mid.yaml`   | 0.15 | `type_updated` (~50 classes)  | integration axis (max scIB Total) |
| `minuet_tuned.yaml` | 0.10 | `type_region`  (264 classes) | biology axis (max kNN classifier F1) |
| `minuet_tuned_d32.yaml` | 0.10 | `type_region` | compact deploy (cell_dim=32, 4× smaller) |

### Numbers (n=3 multi-seed, Wang in-distribution + ROSMAP cross-study)

| Variant | Region F1 | Wang scIB | ROSMAP scIB | Wang MRR | ROSMAP MRR |
|---|---:|---:|---:|---:|---:|
| Minuet-mid (n=3) | 0.714 ± 0.008 | **0.591 ± 0.005** | **0.634 ± 0.002** | 0.0723 ± 0.0030 | 0.0194 ± 0.0005 |
| Minuet-tuned (n=3) | **0.806 ± 0.002** | 0.543 ± 0.006 | 0.605 ± 0.003 | 0.0768 ± 0.0013 | 0.0205 ± 0.0004 |
| Seurat WNN (RPCA+WNN) | 0.308 | 0.576 | — | — | — |
| MultiVI (scVI-tools)  | 0.403 | 0.568 | — | — | — |

Both Minuet variants beat both baselines on Region F1 by **2–2.5×**. Mid wins
canonical scIB Total without post-hoc Harmony.

`minuet_tuned_d32`: Region F1 = 0.770, Wang scIB = 0.552 — 4× smaller embedding
than the d=128 versions; deploy point if storage / throughput matters.

## Layout

- `minuet/`     Python package (model, losses, data, CMI-BI auxiliary)
- `configs/`    canonical recipe yamls
- `scripts/`    train and evaluation entry points
- `tests/`      unit tests
- `docs/`       method notes and migration history

## History

Minuet is the publication-ready successor to the Apollo working package. The
underlying model architecture (factorised modality encoders + PoE shared-token
fusion + Gaussian decoder, 2026-04-27 vintage) is identical; the rename
reflects a sharper architectural identity around the contrastive ↔ CMI-BI
decomposition rather than the original "foundation model" framing. See
`docs/from_apollo.md` for the migration mapping.
