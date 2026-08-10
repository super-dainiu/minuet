from __future__ import annotations

import tempfile
import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from minuet import Minuet, MinuetConfig, MinuetEncoder, MinuetModule
from minuet.losses import (
    COVARIANCE_AGREEMENT_WEIGHT,
    COVARIANCE_START_FRACTION,
    DONOR_ASSOCIATION_WEIGHT,
    PAIRED_TARGET_WEIGHT,
    PAIR_TEMPERATURE,
    PAIR_WEIGHT,
    within_context_covariance_agreement,
)
from minuet.model import (
    PRODUCTION_HIDDEN_DIM,
    PRODUCTION_LATE_LR_MULTIPLIER,
    PRODUCTION_LEARNING_RATE,
    PRODUCTION_RESIDUAL_BLOCKS,
    PRODUCTION_SHARED_DIM,
    PRODUCTION_UPDATES,
    PRODUCTION_WEIGHT_DECAY,
)


def paired_anndata(
    n_genes: int = 8, n_regions: int = 10
) -> ad.AnnData:
    # Two contexts, each containing two donors with twelve cells.
    donor = np.asarray(["d1"] * 12 + ["d2"] * 12 + ["d1"] * 12 + ["d2"] * 12)
    context = np.asarray(["c1"] * 24 + ["c2"] * 24)
    rng = np.random.default_rng(7)
    rna = rng.poisson(2.0, size=(48, n_genes)).astype(np.float32)
    atac = rng.binomial(1, 0.25, size=(48, n_regions)).astype(np.float32)
    result = ad.AnnData(sp.csr_matrix(np.concatenate([rna, atac], axis=1)))
    result.obs_names = [f"cell-{index}" for index in range(48)]
    result.var_names = [
        *[f"gene-{index}" for index in range(n_genes)],
        *[f"peak-{index}" for index in range(n_regions)],
    ]
    result.obs["donor"] = donor
    result.obs["context"] = context
    return result


class MinuetProductionTest(unittest.TestCase):
    def make_model(self) -> tuple[ad.AnnData, Minuet]:
        adata = paired_anndata()
        Minuet.setup_anndata(adata, donor_key="donor", context_key="context")
        return adata, Minuet(adata, n_genes=8, n_regions=10)

    def test_trial45_configuration_and_capacity_are_frozen(self) -> None:
        config = MinuetConfig(2048, 4096)
        self.assertEqual(config.hidden_dim, PRODUCTION_HIDDEN_DIM)
        self.assertEqual(config.shared_dim, PRODUCTION_SHARED_DIM)
        self.assertEqual(config.residual_blocks, PRODUCTION_RESIDUAL_BLOCKS)
        model = MinuetModule(config)
        self.assertEqual(model.trainable_parameter_count(), 2_127_680)
        self.assertEqual(model.encoder().parameter_count(), 1_724_096)
        self.assertEqual(PRODUCTION_UPDATES, 16_000)
        self.assertAlmostEqual(PRODUCTION_LEARNING_RATE, 0.0003090937377859925)
        self.assertAlmostEqual(PRODUCTION_WEIGHT_DECAY, 3.569446559492511e-7)
        self.assertAlmostEqual(PRODUCTION_LATE_LR_MULTIPLIER, 0.2410370721169952)
        self.assertAlmostEqual(PAIR_WEIGHT, 1.9435173526733809)
        self.assertAlmostEqual(PAIR_TEMPERATURE, 0.08808674812619335)
        self.assertAlmostEqual(PAIRED_TARGET_WEIGHT, 1.5841466269183442)
        self.assertAlmostEqual(DONOR_ASSOCIATION_WEIGHT, 0.11343234492276047)
        self.assertAlmostEqual(COVARIANCE_AGREEMENT_WEIGHT, 0.3423206519502428)
        self.assertAlmostEqual(COVARIANCE_START_FRACTION, 0.789253759592594)

    def test_training_and_three_representation_views(self) -> None:
        adata, model = self.make_model()
        model.train(max_steps=1, accelerator="cpu")
        self.assertIsInstance(model.module_, MinuetEncoder)
        self.assertTrue(model.is_trained)
        for modality in ("joint", "expression", "accessibility"):
            latent = model.get_latent_representation(modality=modality)
            self.assertEqual(latent.shape, (adata.n_obs, 64))
            self.assertTrue(np.isfinite(latent).all())
        self.assertEqual(
            model.get_latent_representation(indices=[0], batch_size=1).shape,
            (1, 64),
        )

    def test_encoder_only_save_load_and_label_free_query(self) -> None:
        adata, model = self.make_model()
        model.train(max_steps=1, accelerator="cpu")
        expected = model.get_latent_representation()
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory, overwrite=True)
            payload = torch.load(
                f"{directory}/model.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["schema"], "minuet-public-frozen-query-v2")
            serialized = repr(payload.keys()) + repr(payload["encoder"].keys())
            for forbidden in ("donor_values", "context_values", "target_head", "decoder"):
                self.assertNotIn(forbidden, serialized)

            # The reference object supplies only counts and feature order at load.
            query = adata.copy()
            del query.obs["donor"]
            del query.obs["context"]
            query.uns.clear()
            loaded = Minuet.load(directory, adata=query, device="cpu")
            actual = loaded.get_latent_representation()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)

    def test_covariance_agreement_is_context_specific(self) -> None:
        generator = torch.Generator().manual_seed(17)
        base = torch.randn(12, 5, generator=generator)
        z = torch.cat((base, 2.0 * base, base, base), dim=0)
        donor = torch.tensor([0] * 12 + [1] * 12 + [0] * 12 + [1] * 12)
        context = torch.tensor([0] * 24 + [1] * 24)
        loss = within_context_covariance_agreement(z, donor, context)
        # Trace normalization removes scalar covariance changes within context 0.
        self.assertLess(float(loss), 1.0e-10)

    def test_query_feature_order_is_enforced(self) -> None:
        adata, model = self.make_model()
        model.train(max_steps=1, accelerator="cpu")
        query = adata[:, list(range(7, -1, -1)) + list(range(8, 18))].copy()
        with self.assertRaisesRegex(ValueError, "feature names and order"):
            model.get_latent_representation(adata=query)


if __name__ == "__main__":
    unittest.main()
