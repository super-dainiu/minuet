# Changelog

## 0.3.0

- Replace the token/attention network with the trial45 three-block residual MLP.
- Freeze the 224-width, 64-dimensional, 16,000-update production configuration.
- Add the late within-context covariance-agreement objective used by trial45.
- Save only the encoder required for frozen-query mapping; fitting labels,
  decoders, target bases, and the training-only prediction head are excluded.
- Make old 0.2 checkpoints explicitly incompatible rather than loading them
  under changed model semantics.

## 0.2.0

- Add a scvi-tools-style `Minuet` analysis API for AnnData and MuData.
- Add built-in training, latent extraction, modality reconstruction, and
  save/load workflows.
- Rename the direct PyTorch network export to `MinuetModule`.
- Add end-to-end API and checkpoint round-trip tests.
- Remove internal artifact configs, wandb helpers, experiment factories, and
  other research-repository code from the public distribution.

## 0.1.0

- Initial low-level research package.
