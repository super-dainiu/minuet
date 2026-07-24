# Minuet

Minuet integrates fully paired single-cell RNA and ATAC measurements. Its public
API follows the standard scvi-tools workflow, without requiring scvi-tools:

```python
import mudata
from minuet import Minuet

mdata = mudata.read_h5mu("paired_multiome.h5mu")

Minuet.setup_mudata(
    mdata,
    modalities={"rna_layer": "rna", "atac_layer": "atac"},
    batch_key="donor",
    donor_key="donor",
    context_key="tissue",
)
model = Minuet(mdata, n_latent=32)
model.train(max_epochs=100, batch_size=128)

mdata.obsm["X_minuet"] = model.get_latent_representation()
rna_denoised = model.get_normalized_expression()
atac_denoised = model.get_normalized_accessibility()
model.save("minuet_model", overwrite=True)
```

Reloading uses the same convention as scvi-tools:

```python
model = Minuet.load("minuet_model", adata=mdata)
```

## Input

MuData is recommended. The RNA and ATAC modalities must have identical
`obs_names`; Minuet currently models fully paired data only.

Concatenated AnnData is also supported when RNA features precede ATAC features:

```python
Minuet.setup_anndata(adata, batch_key="donor")
model = Minuet(adata, n_genes=20_000, n_regions=100_000, n_latent=32)
```

Raw counts should be supplied in `.X` or in the layer passed to
`setup_mudata`/`setup_anndata`. Minuet fits RNA library normalization and ATAC
TF-IDF weights using the training split.

When both `donor_key` and `context_key` are registered, training automatically
uses exact-pair contrastive galleries matched within donor and observed
context. If either is omitted, Minuet falls back to its label-free global
contrastive objective.

`get_normalized_expression()` and `get_normalized_accessibility()` return
non-negative decoder reconstructions on Minuet's log-normalized RNA and
log-TF-IDF ATAC scales. They are API analogues of the MULTIVI methods, not
negative-binomial posterior counts.

## Public API

- `Minuet.setup_mudata(...)`
- `Minuet.setup_anndata(...)`
- `Minuet(...)`
- `model.train(...)`
- `model.get_latent_representation(...)`
- `model.get_normalized_expression(...)`
- `model.get_normalized_accessibility(...)`
- `model.save(...)` and `Minuet.load(...)`
- `model.to_device(...)`

Advanced low-level use remains available as `MinuetModule`, `MinuetConfig`, and
`MinuetLosses`, but those are not needed for a normal analysis.

## Scope

The current release intentionally supports the reliable common path: fully
paired RNA+ATAC integration. It does not claim MultiVI's unpaired-data,
protein-modality, differential-expression, or scArches query-training features.
