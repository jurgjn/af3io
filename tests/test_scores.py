import functools, importlib.util, json, runpy, sys
import numpy as np, pytest, af3io, af3io.scores
from testdata import TEST_DATA, fetch_test_data, fetch_test_dir

# Real AlphaFold3 example (AURKA + TPX2, PDB-derived) distributed with the ipSAE reference
# implementation; pinned to a commit (see test_data_registry.txt) so the fixture and its published reference numbers
# (https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt)
# stay in sync: ipSAE=0.448952/0.866498 (A->B/B->A), pDockQ=0.5235, pDockQ2=0.7120/0.6278
@pytest.fixture(scope='session')
def ipsae_example_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('ipsae_example')
    for fname in ['ipsae/ipsae.py', 'ipsae/fold_aurka_0_tpx2_0_full_data_0.json', 'ipsae/fold_aurka_0_tpx2_0_model_0.cif']:
        fetch_test_data(fname, tmpdir)
    return tmpdir

@pytest.fixture(scope='session')
def ipsae_reference(ipsae_example_dir):
    """ Run the published ipsae.py on the example AF3 prediction to get its internal, per-residue
    arrays (chains/pae_matrix/distances/cb_plddt) and its own computed scores. AlphaFold3 tokenises
    modified residues (e.g. the two phosphothreonines here) and ligands atom-by-atom; ipsae.py
    collapses these down to one token per real polymer residue (dropping ligand chains), which
    af3io.scores's chain_pair_reduce does not do on its own -- so its arrays, rather than the raw
    full_data.json tokens, are the correct like-for-like input to compare af3io's scoring functions
    against the published per-chain-pair values.
    """
    json_path = str(ipsae_example_dir / 'fold_aurka_0_tpx2_0_full_data_0.json')
    cif_path = str(ipsae_example_dir / 'fold_aurka_0_tpx2_0_model_0.cif')
    saved_argv = sys.argv
    try:
        sys.argv = ['ipsae.py', json_path, cif_path, '10', '10']
        return runpy.run_path(str(ipsae_example_dir / 'ipsae.py'), run_name='ipsae_reference')
    finally:
        sys.argv = saved_argv

def test_ipsae_literature(ipsae_reference):
    # https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    ipsae_mat = af3io.scores.chain_pair_reduce(functools.partial(af3io.scores.ipsae, pae_cutoff=10), chains, pae_matrix)
    assert ipsae_mat[0, 1] == pytest.approx(0.448952, abs=1e-5)  # A -> B, asym
    assert ipsae_mat[1, 0] == pytest.approx(0.866498, abs=1e-5)  # B -> A, asym
    assert af3io.scores.ptm_symm(ipsae_mat)[0, 1] == pytest.approx(0.866498, abs=1e-3)  # max
    assert af3io.scores.min_symm(ipsae_mat)[0, 1] == pytest.approx(0.448952, abs=1e-3)  # min

def test_interface_scores_synthetic():
    # Two chains (A: 3 tokens, B: 2 tokens); the A/B interface is a confident contact (A3-B1, PAE 3),
    # a confident non-contact (A2-B1, PAE 6) and an unconfident contact (A3-B2, PAE 20)
    chains = np.array(['A', 'A', 'A', 'B', 'B'])
    pae = np.full((5, 5), 30.0)
    pae[:3, :3] = pae[3:, 3:] = 1.0
    pae[2, 3] = pae[3, 2] = 3.0
    pae[1, 3] = pae[3, 1] = 6.0
    pae[2, 4] = pae[4, 2] = 20.0
    contacts = np.zeros((5, 5), dtype=bool)
    contacts[2, 3] = contacts[3, 2] = contacts[2, 4] = contacts[4, 2] = True

    def ab(func, *arrs):
        return af3io.scores.chain_pair_reduce(func, chains, *arrs)[0, 1]

    # LIS transform: 1 - PAE/12 for PAE < 12; cLIS/cLIA only count confident contacts
    assert ab(af3io.scores.lis, pae) == pytest.approx(((1 - 3/12) + (1 - 6/12)) / 2)
    assert ab(af3io.scores.clis, contacts, pae) == pytest.approx(1 - 3/12)
    assert ab(af3io.scores.lia, pae) == 2
    assert ab(af3io.scores.clia, contacts, pae) == 1
    lia_symm = af3io.scores.sum_symm(af3io.scores.chain_pair_reduce(af3io.scores.lia, chains, pae))
    clia_symm = af3io.scores.sum_symm(af3io.scores.chain_pair_reduce(af3io.scores.clia, chains, contacts, pae))
    assert af3io.scores.ilia(lia_symm, clia_symm)[0, 1] == pytest.approx(np.sqrt(4 * 2))

    # As in AFM-LIS, PAE exactly at the cutoff is confident (counted by LIA) but contributes 0 to LIS
    pae_cutoff = pae.copy()
    pae_cutoff[0, 4] = 12.0
    assert ab(af3io.scores.lia, pae_cutoff) == 3
    assert ab(af3io.scores.lis, pae_cutoff) == pytest.approx(((1 - 3/12) + (1 - 6/12) + 0) / 3)

    # Interface size: 2 contacting pairs, involving A3 and B1/B2
    assert ab(af3io.scores.n_contacts, contacts) == 2
    assert ab(af3io.scores.n_interface_residues, contacts) == 3

    # Pair pTM over both chains (d0 from 5 tokens, clipped to 19), max over aligned tokens of the mean TM-transformed PAE
    d0 = 1.24 * (19 - 15) ** (1/3) - 1.8
    tm = 1 / (1 + (pae / d0) ** 2)
    chain_pair_ptm = af3io.scores.chain_pair_ptm_from_pae(chains, pae)
    assert chain_pair_ptm[0, 1] == chain_pair_ptm[1, 0] == pytest.approx(tm.mean(axis=1).max())
    assert chain_pair_ptm[0, 0] == pytest.approx(tm[:3, :3].mean(axis=1).max())
    assert af3io.scores.model_confidence(0.5, chain_pair_ptm)[0, 1] == pytest.approx(0.8 * 0.5 + 0.2 * chain_pair_ptm[0, 1])

    # Symmetrisation
    asym = np.array([[1.0, 0.2], [0.7, 1.0]])
    assert af3io.scores.min_symm(asym)[0, 1] == af3io.scores.min_symm(asym)[1, 0] == 0.2
    assert af3io.scores.ptm_symm(asym)[0, 1] == 0.7

@pytest.fixture(scope='session')
def afm_lis():
    # AFM-LIS reference implementation (lis.py) imported as a module
    spec = importlib.util.spec_from_file_location('afm_lis', TEST_DATA.fetch('afm-lis/lis.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_lis_afm_lis(afm_lis):
    # LIS/cLIS/iLIS/LIA/cLIA/iLIA should match the AFM-LIS reference implementation for every model of a heterodimer
    # prediction (barnase-barstar has PAE values of exactly 12.0, at the confident interface cutoff)
    pred_dir = fetch_test_dir('alphafold3_examples/alphafold3_predictions/barnase_barstar')
    df = af3io.predictions.read_summary_scores(pred_dir)
    assert len(df) == 5
    with af3io.predictions.Predictions(pred_dir) as pred:
        for _, row in df.iterrows():
            with pred.open(row['model_path']) as fh:
                struct_text = fh.read().decode()
            with pred.open(row['confidences_path']) as fh:
                pae = np.array(json.load(fh)['pae'])
            (expected,) = afm_lis.analyze_single_model(struct_text, pae, {}, 'cif', 'af3', row['confidences_path'], None)
            assert (expected['ci'], expected['cj']) == ('A', 'B')
            for col, key in [('lis', 'LIS'), ('clis', 'cLIS'), ('ilis', 'iLIS'), ('lia', 'LIA'), ('clia', 'cLIA'), ('ilia', 'iLIA')]:
                assert row[f'chain_pair_{col}'][0][1] == pytest.approx(expected[key], abs=1e-6), (row['model_path'], col)

def test_pdockq_pdockq2_literature(ipsae_reference):
    # https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    distances, cb_plddt = ipsae_reference['distances'], ipsae_reference['cb_plddt']
    isin_8A = distances <= 8.0
    plddt_row = np.broadcast_to(cb_plddt[:, None], pae_matrix.shape)
    plddt_col = np.broadcast_to(cb_plddt[None, :], pae_matrix.shape)

    pdockq_mat = af3io.scores.chain_pair_reduce(af3io.scores.pdockq, chains, isin_8A, plddt_row, plddt_col)
    assert pdockq_mat[0, 1] == pytest.approx(0.5235, abs=1e-4)  # direction-independent
    assert pdockq_mat[1, 0] == pytest.approx(0.5235, abs=1e-4)

    pdockq2_mat = af3io.scores.chain_pair_reduce(af3io.scores.pdockq2, chains, isin_8A, pae_matrix, plddt_row, plddt_col)
    assert pdockq2_mat[0, 1] == pytest.approx(0.7120, abs=1e-4)  # A -> B, asym
    assert pdockq2_mat[1, 0] == pytest.approx(0.6278, abs=1e-4)  # B -> A, asym
    assert af3io.scores.ptm_symm(pdockq2_mat)[0, 1] == pytest.approx(0.7120, abs=1e-3)  # max
