# Minuet

Minuet learns a fixed per-cell representation from fully paired single-cell
RNA and ATAC measurements. The fitting objective has three explicit roles:

- exact RNA--ATAC pairs are contrasted within donor;
- a shared head predicts a donor-centered, fitting-only paired target;
- residual donor association is measured within recorded biological contexts.

Donor and context labels supervise fitting only. Query encoding uses the two
measurement matrices and never recomputes cohort statistics.

```python
import mudata
from minuet import Minuet

mdata = mudata.read_h5mu("paired_multiome.h5mu")
Minuet.setup_mudata(
    mdata,
    modalities={"rna": "rna", "atac": "atac"},
    donor_key="donor",
    context_key="cell_type",
)

model = Minuet(mdata, n_latent=64)
model.train(max_epochs=100, batch_size=128)
mdata.obsm["X_minuet"] = model.get_latent_representation()
model.save("minuet_model", overwrite=True)
```

Apply the fitted encoder to paired query cells with the same ordered features:

```python
model = Minuet.load("minuet_model", adata=mdata)
query_embedding = model.get_latent_representation(adata=query_mdata)
```

MuData is recommended. Concatenated AnnData is also supported when RNA features
precede ATAC features; pass `n_genes` to the constructor. Minuet currently
supports fully paired RNA+ATAC data only.
