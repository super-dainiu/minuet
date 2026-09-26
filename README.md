# Minuet

Minuet learns a representation of paired single-cell RNA and ATAC profiles
from many donors. It is fitted once on reference donors; its two frozen encoders
then embed cells of new donors one cell at a time, with no labels, no parameter
updates, and no statistics of the query set.

## Installation

```bash
pip install git+https://github.com/super-dainiu/minuet.git
```

Minuet requires Python 3.10 or later and PyTorch 2.0 or later. A GPU is used
when available.

## Basic usage

```python
import mudata as md
from minuet import Minuet

mdata = md.read_h5mu("reference.h5mu")
Minuet.setup_mudata(mdata, donor_key="donor", covariate_keys=["disease"])
model = Minuet(mdata)
model.train()
mdata.obsm["X_minuet"] = model.get_latent_representation()
model.save("minuet_model")
```

Embed cells of new donors with the saved model:

```python
query = md.read_h5mu("query.h5mu")
model = Minuet.load("minuet_model", query)
query.obsm["X_minuet"] = model.get_latent_representation()
```

## User guide

### Preliminaries

Minuet takes raw counts of paired cells in a
[MuData](https://mudata.readthedocs.io) object: genes in `mdata["rna"]` and
peaks in `mdata["atac"]`, with the same cells in the same order. Select features
on the reference cells only; the paper uses 2,048 genes and 4,096 peaks.
Fitting needs a donor label for every cell and, optionally, biological
covariates such as disease status or sex. Query cells need the reference
features in the same order and no labels.

### Model

For each assay, a residual MLP encoder returns a shared representation and a
Gaussian private variable. A decoder reconstructs the assay from its private
variable alone, with a negative binomial likelihood for RNA counts and a
Bernoulli likelihood for binarized ATAC, so assay-specific variation has a place
to go. The joint representation of a cell is the average of its RNA and ATAC
representations.

The training objective adds four terms to the reconstruction loss:

| Term | Default weight | Role |
| --- | --- | --- |
| Within-donor contrastive loss | 2.0 | Matches the RNA and ATAC profiles of each cell against other cells of the same donor, after centering each assay within the donor, so a donor cue cannot identify the pair. |
| Shared prediction | 1.5 | One bias-free head, shared by both assays and all donors, predicts a target made of donor-centered low-rank scores of each assay. The common head places every donor in the same coordinates. |
| Donor association | 0.1 | Linear CKA between the joint representation and donor identity within each covariate group. |
| Covariance alignment | 0.35 | Aligns trace-normalized donor covariances within each covariate group. It starts after 80% of the updates, when the learning rate is multiplied by 0.25. |

Covariates define the groups of the two donor penalties: differences between
groups are kept and differences between donors of the same group are reduced.
A biological variable left out of the covariates is treated as a donor effect.
Each batch holds up to two covariate groups, up to four donors per group, and
twelve cells per donor.

### Inference

After fitting, Minuet keeps the two encoders and embeds each cell independently.
With the default `norm="batch"`, the encoders apply BatchNorm statistics of the
reference. When query cells come from a study absent from the reference, fit
the model with `norm="layer"`, which normalizes each cell on its own.

## API

| Method | Description |
| --- | --- |
| `Minuet.setup_mudata(mdata, donor_key, covariate_keys=None, rna_modality="rna", atac_modality="atac", rna_layer=None, atac_layer=None)` | Register paired reference cells. |
| `Minuet(mdata, n_hidden=224, n_latent=64, n_private=16, n_layers=3, n_decoder_hidden=64, dropout_rate=0.2, norm="batch", prediction_rank=32, seed=0)` | Build the model. |
| `Minuet.train(max_updates=16000, lr=3e-4, weight_decay=3e-7, late_lr_factor=0.25, covariance_start=0.8, temperature=0.1, pair_weight=2.0, prediction_weight=1.5, donor_weight=0.1, covariance_weight=0.35, kl_weight=7e-5, groups_per_batch=2, donors_per_group=4, cells_per_donor=12, device=None, progress_bar=True)` | Fit on the reference. |
| `Minuet.get_latent_representation(mdata=None, modality="joint", batch_size=512)` | Embed cells; `modality` is `"joint"`, `"rna"`, or `"atac"`. |
| `Minuet.save(dir_path, overwrite=False)` | Save the model. |
| `Minuet.load(dir_path, mdata)` | Load a saved model and attach `mdata`. |

The defaults are the configuration of the Minuet paper. Every parameter is
documented in its docstring, for example `help(Minuet.train)`.

## License

Apache 2.0.
