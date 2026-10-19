import af3io

def test_pipeline_sequence():
    # modified_rna example: residues at modified positions are replaced by the parent nucleotide (OMG=>G, PSU=>U, 5MC=>C)
    rna_mods = [{'modificationType': 'PSU', 'basePosition': 13}, {'modificationType': '5MC', 'basePosition': 18}, {'modificationType': 'OMG', 'basePosition': 4}]
    assert af3io.residue_names.rna_sequence('GGCCCGAUAGCUCAGUCGGUAGAGC', rna_mods) == 'GGCGCGAUAGCUUAGUCCGUAGAGC'
    assert af3io.residue_names.rna_sequence('GGCXT') == 'GGCNN' # unknown RNA residues
    # Protein: phosphothreonine/-tyrosine (TPO/PTR) on T/Y are unchanged, unknown residues (e.g. selenocysteine U) => X
    ptms = [{'ptmType': 'TPO', 'ptmPosition': 2}, {'ptmType': 'PTR', 'ptmPosition': 4}]
    assert af3io.residue_names.protein_sequence('ATGYU', ptms) == 'ATGYX'
    assert af3io.residue_names.protein_sequence('AAAA', ptms) == 'ATAY'
    # DNA is written as-is by the data pipeline
    dna = {'sequence': 'GATTACA', 'modifications': [{'modificationType': '5CM', 'basePosition': 1}]}
    assert af3io.residue_names.pipeline_sequence('dna', dna) == 'GATTACA'
