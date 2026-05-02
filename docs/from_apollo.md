# Apollo → Minuet migration

Minuet is a clean publication fork of the Apollo working package. The model
architecture and losses are unchanged — what differs is naming, scope, and
exposed API.

## Renames

| Apollo (working) | Minuet (publication) |
|---|---|
| `software/apollo/` | `software/minuet/` |
| python package `apollo` | python package `minuet` |
| `ApolloV3`, `ApolloV3Config` | `Minuet`, `MinuetConfig` |
| `ApolloV3Losses` | `MinuetLosses` |

## Dropped (no longer shipped)

The publication package keeps only the canonical recipe (formerly Apollo-V3).
The earlier scaffolds are still in `software/apollo/` for ongoing experiments
but are not part of Minuet:

- `apollo.ApolloV1`, `apollo.ApolloLosses` (v1 baseline, no PoE fusion)
- `apollo.ApolloV2`, `apollo.ApolloV2Losses` (intermediate scaffold)
- `apollo.ApolloV4`, `apollo.ApolloV4Config` (Perceiver-style cross-attention iters)

## Kept identical

The internals of the canonical recipe are byte-identical:

- `cmi_bi.py` — Conditional Mutual Information batch-invariance auxiliary network
- `data.py` — paired multiome dataset loader + dataset-aware sampler
- `tracking.py` — wandb integration utilities

## Config naming

| Apollo recipe | Minuet recipe |
|---|---|
| `apollo_fm_0427_cmibi_mid.yaml` | `configs/minuet_mid.yaml` |
| `apollo_fm_0427_cmibi_tuned.yaml` | `configs/minuet_tuned.yaml` |
| `apollo_fm_0427_cmibi_tuned_d32.yaml` | `configs/minuet_tuned_d32.yaml` |

Other Apollo configs (mask sweep, multi-seed, ablations) stay under
`software/apollo/configs/` since they're working artefacts, not part of the
publication.

## Checkpoints

Apollo-trained checkpoints **load into Minuet** via
`load_compatible_state_dict`. The state-dict keys for the v3 architecture
were never renamed — only the wrapping class.
