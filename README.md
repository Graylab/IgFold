# IgFold

Official repository for [IgFold](https://www.biorxiv.org/content/10.1101/2022.04.20.488972): Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.

The code, data, and weights for this work are made available for non-commercial use (including at commercial entities) under the terms of the [JHU Academic Software License Agreement](LICENSE.md). For commercial inquiries, please contact `jruffolo[at]jhu.edu`.

## Install

For easiest use, install IgFold via PyPI:

```bash
$ pip install igfold
```

To access the latest version of the code, clone and install the repository:

```bash
$ git clone git@github.com:Graylab/IgFold.git 
$ pip install IgFold
```

Two refinement methods are supported for IgFold predictions. To follow the manuscript, PyRosetta should be installed following the instructions [here](http://pyrosetta.org/downloads). If PyRosetta is not installed, refinement with OpenMM will be attempted. For this option, OpenMM must be installed and configured before running IgFold as follows:

```bash
$ conda install conda-forge openmm pdbfixer
```

## Usage

_Note_: The first time `IgFoldRunner` is initialized, it will download the pre-trained weights. This may take a few minutes and will require a network connection.

### Antibody structure prediction from sequence

Paired antibody sequences can be provided as a dictionary of sequences, where the keys are chain names and the values are the sequences.

```python
from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=True, # Send predicted structure to AbNum server for Chothia renumbering
)
```

To predict a nanobody structure (or an individual heavy or light chain), simply provide one sequence:

```python
from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS"
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Nanobody sequence
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=True, # Send predicted structure to AbNum server for Chothia renumbering
)
```

To predict a structure without PyRosetta refinement, set `do_refine=False`:

```python
from igfold import IgFoldRunner

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS"
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Nanobody sequence
    do_refine=False, # Refine the antibody structure with PyRosetta
    do_renum=True, # Send predicted structure to AbNum server for Chothia renumbering
)
```

### Predicted RMSD for antibody structures

RMSD estimates are calculated per-residue and recorded in the B-factor column of the output PDB file. These values are also returned from the `fold` method.

```python
from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
out = igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=True, # Send predicted structure to AbNum server for Chothia renumbering
)

out.prmsd # Predicted RMSD for each residue's N, CA, C, CB atoms (dim: 1, L, 4)
```

### Antibody sequence embedding

Representations from IgFold may be useful as features for machine learning models. The `embed` method can be used to surface a variety of antibody representations from the model:

```python
from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}

igfold = IgFoldRunner()
emb = igfold.embed(
    sequences=sequences, # Antibody sequences
)

emb.bert_embs # Embeddings from AntiBERTy final hidden layer (dim: 1, L, 512)
emb.gt_embs # Embeddings after graph transformer layers (dim: 1, L, 64)
emb.strucutre_embs # Embeddings after template incorporation IPA (dim: 1, L, 64)
```

## Bug reports

If you run into any problems while using IgFold, please create a [Github issue](https://github.com/Graylab/IgFold/issues) with a description of the problem and the steps to reproduce it.

## Citing this work

```bibtex
@article{ruffolo2021deciphering,
    title = {Deciphering antibody affinity maturation with language models and weakly supervised learning},
    author = {Ruffolo, Jeffrey A and Gray, Jeffrey J and Sulam, Jeremias},
    journal = {arXiv},
    year= {2021}
}
@article{ruffolo2022fast,
    title = {Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies},
    author = {Ruffolo, Jeffrey A and Chu, Lee-Shin and Mahajan, Sai Pooja and Gray, Jeffrey J},
    journal = {bioRxiv},
    year= {2022}
}
```
