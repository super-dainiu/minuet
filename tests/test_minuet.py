import anndata as ad
import mudata as md
import numpy as np
import pytest

from minuet import Minuet


def make_mdata(n_donors=4, n_groups=2, n_cells=16, n_genes=30, n_peaks=40, seed=0):
    rng = np.random.default_rng(seed)
    n = n_donors * n_groups * n_cells
    names = [f"cell{i}" for i in range(n)]
    rna = ad.AnnData(rng.poisson(2.0, (n, n_genes)).astype(np.float32))
    atac = ad.AnnData(rng.poisson(0.5, (n, n_peaks)).astype(np.float32))
    rna.obs_names = names
    atac.obs_names = names
    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.obs["donor"] = np.repeat([f"d{i}" for i in range(n_donors)], n_groups * n_cells)
    mdata.obs["disease"] = np.tile(np.repeat([f"g{j}" for j in range(n_groups)], n_cells), n_donors)
    return mdata


def trained_model(norm="batch"):
    mdata = make_mdata()
    Minuet.setup_mudata(mdata, donor_key="donor", covariate_keys=["disease"])
    model = Minuet(mdata, n_hidden=32, n_latent=8, n_private=4, n_decoder_hidden=8, norm=norm, prediction_rank=4)
    model.train(max_updates=20, cells_per_donor=8, device="cpu", progress_bar=False)
    return model


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_train_and_embed(norm):
    model = trained_model(norm)
    for modality in ("joint", "rna", "atac"):
        z = model.get_latent_representation(modality=modality)
        assert z.shape == (model.mdata.n_obs, 8)
        assert np.isfinite(z).all()
    assert len(model.history) == 1


def test_query_cells_are_embedded_one_at_a_time():
    model = trained_model()
    full = model.get_latent_representation()
    query = model.get_latent_representation(make_mdata()[5:12].copy(), batch_size=3)
    np.testing.assert_allclose(query, full[5:12], atol=1e-5)


def test_save_and_load(tmp_path):
    model = trained_model()
    model.save(str(tmp_path / "model"))
    with pytest.raises(FileExistsError):
        model.save(str(tmp_path / "model"))
    loaded = Minuet.load(str(tmp_path / "model"), make_mdata())
    np.testing.assert_array_equal(loaded.get_latent_representation(), model.get_latent_representation())


def test_query_features_must_match():
    model = trained_model()
    query = make_mdata(n_genes=29)
    with pytest.raises(ValueError, match="features"):
        model.get_latent_representation(query)
