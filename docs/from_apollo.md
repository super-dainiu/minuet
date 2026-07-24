# Migrating from the internal Apollo/Minuet code

Version 0.2 separates the public analysis API from the research implementation.
Experiment launchers, artifact paths, wandb configuration, and paper-specific
checkpoint factories are intentionally not part of this package.

## Public workflow

Old internal code manually created `MinuetConfig`, datasets, optimizers, and
losses. External analyses should now use:

```python
from minuet import Minuet

Minuet.setup_mudata(mdata, batch_key="donor")
model = Minuet(mdata, n_latent=32)
model.train()
latent = model.get_latent_representation()
```

The low-level network is still available as `MinuetModule` for research code
that needs direct PyTorch access. `MinuetConfig` and `MinuetLosses` remain
available from the top-level package.

## Compatibility boundary

The public model supports fully paired RNA and ATAC observations. Internal
Apollo checkpoints should continue to be evaluated with the corresponding
frozen research code; they are not silently reinterpreted as version 0.2
public-model checkpoints.
