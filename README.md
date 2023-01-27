# IgFold

Official repository for [IgFold](https://www.biorxiv.org/content/10.1101/2022.04.20.488972): Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.

The code and pre-trained models from this work are made available for non-commercial use (including at commercial entities) under the terms of the [JHU Academic Software License Agreement](LICENSE.md). For commercial inquiries, please contact Johns Hopkins Tech Ventures at `dmalon11@jhu.edu`.

<!-- Try antibody structure prediction in [Google Colab](https://colab.research.google.com/github/Graylab/IgFold/blob/main/IgFold.ipynb). -->

## Updates

 - Version 0.2.0
   - Add gradient-based refinement to resolve clashes and improve backbone geometry.

## Install

For easiest use, [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) and install IgFold via PyPI:

```bash
$ pip install igfold
```

To access the latest version of the code, clone and install the repository:

```bash
$ git clone git@github.com:Graylab/IgFold.git 
$ pip install IgFold
```

### Refinement

Two refinement methods are supported for IgFold predictions. To follow the manuscript, PyRosetta should be installed following the instructions [here](http://pyrosetta.org/downloads). If PyRosetta is not installed, refinement with OpenMM will be attempted. For this option, OpenMM must be installed and configured before running IgFold as follows:

```bash
$ conda install -c conda-forge openmm pdbfixer
```

### Renumbering

Antibody renumbering requires installation of AbNumber. To install AbNumber, run the following command:

```bash
$ conda install -c bioconda abnumber
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
    do_renum=True, # Renumber predicted antibody structure (Chothia)
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
    do_renum=True, # Renumber predicted antibody structure (Chothia)
)
```

To predict a structure without refinement, set `do_refine=False`:

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
    do_renum=True, # Renumber predicted antibody structure (Chothia)
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
    do_renum=True, # Renumber predicted antibody structure (Chothia)
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

### Extra options

Refinement with OpenMM can be prioritized over PyRosetta by setting `use_openmm=True`.

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
    use_openmm=True, # Use OpenMM for refinement
    do_renum=True, # Renumber predicted antibody structure (Chothia)
)
```

## Synthetic antibody structures

To demonstrate the capabilities of IgFold for large-scale prediction of antibody structures, we applied the model to two sets of natural paired antibody sequences. 

The first set contains 104,994 non-redundant paired antibody sequences from the Observed Antibody Space database. These predicted structures are made available for use [online](https://data.graylab.jhu.edu/igfold_oas_paired95.tar.gz).

```bash
$ wget https://data.graylab.jhu.edu/igfold_oas_paired95.tar.gz
```

The second set contains 1.3M unique paired antibodies from four human donors, collected by [Jaffe et al.](https://www.nature.com/articles/s41586-022-05371-z). These predicted structures are made available for use [online](https://data.graylab.jhu.edu/Jaffe2022.tar.gz).

```bash
$ wget https://data.graylab.jhu.edu/Jaffe2022.tar.gz
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
