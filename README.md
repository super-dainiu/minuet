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

mdata = md.read_h5mu("reference.h5mu")
Minuet.setup_mudata(mdata, donor_key="donor", covariate_keys=["disease"])
model = Minuet(mdata)
model.train()
mdata.obsm["X_minuet"] = model.get_latent_representation()
model.save("minuet_model")

query = md.read_h5mu("query.h5mu")
model = Minuet.load("minuet_model", query)
query.obsm["X_minuet"] = model.get_latent_representation()
```

`mdata["rna"]` and `mdata["atac"]` hold raw counts of the same cells in the
same order. Query cells need the reference features and no labels. Differences
between the groups defined by `covariate_keys` are kept; differences between
donors within a group are reduced. For query cells from a study absent from the reference,
use `Minuet(mdata, norm="layer")`. The defaults are the configuration of the
paper; see the docstrings for all parameters.

## License

Apache 2.0.
