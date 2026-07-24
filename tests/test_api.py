from __future__ import annotations

import tempfile
import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from minuet import Minuet, MinuetModule


def paired_anndata(n_obs: int = 12, n_genes: int = 5, n_regions: int = 7) -> ad.AnnData:
    rng = np.random.default_rng(7)
    rna = rng.poisson(2.0, size=(n_obs, n_genes)).astype(np.float32)
    atac = rng.binomial(1, 0.25, size=(n_obs, n_regions)).astype(np.float32)
    matrix = sp.csr_matrix(np.concatenate([rna, atac], axis=1))
    result = ad.AnnData(matrix)
    result.obs_names = [f"cell-{index}" for index in range(n_obs)]
    result.var_names = [
        *[f"gene-{index}" for index in range(n_genes)],
        *[f"peak-{index}" for index in range(n_regions)],
    ]
    result.obs["donor"] = ["d1"] * (n_obs // 2) + ["d2"] * (n_obs - n_obs // 2)
    result.obs["context"] = ["brain"] * n_obs
    return result


class MinuetApiTest(unittest.TestCase):
    def make_model(self) -> tuple[ad.AnnData, Minuet]:
        adata = paired_anndata()
        Minuet.setup_anndata(adata, batch_key="donor")
        model = Minuet(
            adata,
            n_genes=5,
            n_regions=7,
            n_hidden=8,
            n_latent=4,
            n_layers_encoder=1,
            n_layers_decoder=1,
            group_size=4,
            num_heads=2,
        )
        return adata, model

    def test_scvi_style_training_and_inference(self) -> None:
        adata, model = self.make_model()
        self.assertIsInstance(model.module_, MinuetModule)
        model.train(
            max_epochs=1,
            train_size=0.75,
            batch_size=4,
            accelerator="cpu",
            early_stopping=False,
        )
        self.assertEqual(model.get_latent_representation().shape, (adata.n_obs, 4))
        self.assertEqual(
            model.get_latent_representation(modality="expression").shape,
            (adata.n_obs, 4),
        )
        expression = model.get_normalized_expression(gene_list=["gene-1", "gene-3"])
        accessibility = model.get_normalized_accessibility(
            region_list=["peak-2"], return_numpy=True
        )
        self.assertEqual(expression.shape, (adata.n_obs, 2))
        self.assertEqual(accessibility.shape, (adata.n_obs, 1))
        self.assertTrue(np.isfinite(expression.to_numpy()).all())
        self.assertTrue((accessibility >= 0).all())

    def test_save_load_roundtrip(self) -> None:
        adata, model = self.make_model()
        model.train(
            max_epochs=1,
            train_size=1.0,
            batch_size=4,
            accelerator="cpu",
            early_stopping=False,
        )
        expected = model.get_latent_representation()
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory, overwrite=True)
            loaded = Minuet.load(directory, adata=adata, device="cpu")
            actual = loaded.get_latent_representation()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)
            registry = Minuet.load_registry(directory)
            self.assertEqual(registry["rna_var_names"][0], "gene-0")

    def test_setup_is_required_and_unpaired_mode_is_explicitly_rejected(self) -> None:
        adata = paired_anndata()
        with self.assertRaisesRegex(ValueError, "setup"):
            Minuet(adata, n_genes=5)
        Minuet.setup_anndata(adata)
        with self.assertRaises(NotImplementedError):
            Minuet(adata, n_genes=5, fully_paired=False)

    def test_mudata_setup_matches_multivi_call_pattern(self) -> None:
        try:
            import mudata as mu
        except ImportError:
            self.skipTest("mudata optional dependency is not installed")
        combined = paired_anndata()
        rna = ad.AnnData(combined.X[:, :5].copy())
        atac = ad.AnnData(combined.X[:, 5:].copy())
        rna.obs_names = combined.obs_names.copy()
        atac.obs_names = combined.obs_names.copy()
        rna.var_names = combined.var_names[:5].copy()
        atac.var_names = combined.var_names[5:].copy()
        rna.obs["donor"] = combined.obs["donor"].to_numpy()
        mdata = mu.MuData({"rna": rna, "atac": atac})
        Minuet.setup_mudata(
            mdata,
            modalities={"rna_layer": "rna", "atac_layer": "atac"},
            batch_key="donor",
        )
        model = Minuet(
            mdata,
            n_hidden=8,
            n_latent=4,
            n_layers_encoder=1,
            n_layers_decoder=1,
            group_size=4,
            num_heads=2,
        )
        self.assertEqual(model.n_genes_, 5)
        self.assertEqual(model.n_regions_, 7)

    def test_registered_design_enables_conditional_alignment(self) -> None:
        adata = paired_anndata()
        Minuet.setup_anndata(
            adata,
            batch_key="donor",
            donor_key="donor",
            context_key="context",
        )
        model = Minuet(
            adata,
            n_genes=5,
            n_hidden=8,
            n_latent=4,
            n_layers_encoder=1,
            n_layers_decoder=1,
            group_size=4,
            num_heads=2,
        )
        model.train(
            max_epochs=1,
            train_size=1.0,
            batch_size=12,
            accelerator="cpu",
            early_stopping=False,
        )
        self.assertEqual(model.alignment_mode_, "within_group_raw_exact_pair")
        self.assertTrue(np.isfinite(model.history["train_loss"]).all())


if __name__ == "__main__":
    unittest.main()
