# Minuet

Minuet learns a fixed per-cell representation from fully paired single-cell
RNA and ATAC counts. Its fitting objective separates four roles:

- modality-private reconstruction;
- exact RNA--ATAC matching within donor;
- prediction of one donor-centered paired target through a shared linear head;
- donor-effect correction within recorded biological contexts, including a
  covariance-agreement term during the final part of fitting.

The production encoder is a three-block residual MLP with width 224 and a
64-dimensional representation. Fitting uses 16,000 updates. Donor and context
labels, private decoders, target bases, and the shared prediction head are
discarded after fitting. A saved Minuet model therefore contains only the two
encoders and accepts paired count matrices at query time.

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

model = Minuet(mdata)
model.train()
mdata.obsm["X_minuet"] = model.get_latent_representation()
model.save("minuet_model", overwrite=True)
```

The frozen encoder maps paired query cells with the same ordered features and
does not use query labels, donor identifiers, cohort statistics, or query-time
optimization:

```python
model = Minuet.load("minuet_model", adata=reference_mdata)
query_embedding = model.get_latent_representation(adata=query_mdata)
```

MuData is recommended. Concatenated AnnData is also supported when RNA
features precede ATAC features; pass `n_genes` to the constructor. Inputs must
be finite non-negative counts and the two modalities must be paired in the same
cell order.
