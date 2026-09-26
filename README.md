# Minuet

Minuet learns a representation of paired single-cell RNA and ATAC profiles
from many donors. It is fitted once on reference donors; its two frozen encoders
then embed cells of new donors one cell at a time.

## Installation

```bash
pip install git+https://github.com/super-dainiu/minuet.git
```

## Usage

```python
import mudata as md
from minuet import Minuet

# Raw counts of the same cells in mdata["rna"] and mdata["atac"].
mdata = md.read_h5mu("reference.h5mu")

# Differences between covariate groups are kept; donor differences within a group are reduced.
Minuet.setup_mudata(mdata, donor_key="donor", covariate_keys=["disease"])

# Use norm="layer" when query cells come from a study absent from the reference.
model = Minuet(mdata)

# Paper configuration by default; see help(Minuet.train).
model.train()

# Joint representation; modality="rna" or "atac" returns one assay.
mdata.obsm["X_minuet"] = model.get_latent_representation()
model.save("minuet_model")

# Query cells need the reference features and no labels.
query = md.read_h5mu("query.h5mu")
model = Minuet.load("minuet_model", query)
query.obsm["X_minuet"] = model.get_latent_representation()
```

## License

Apache 2.0.
