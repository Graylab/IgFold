# IgFold

Official repository for [IgFold](https://www.nature.com/articles/s41467-023-38063-x): Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.

The code and pre-trained models from this work are made available for non-commercial use (including at commercial entities) under the terms of the [JHU Academic Software License Agreement](https://github.com/Graylab/IgFold/blob/main/LICENSE.md). For commercial inquiries, please obtain a license through [Johns Hopkins Technology Ventures](https://jhtv.e-lucid.com/product/antibody-structure-prediction-from-pre-trained-language-model-igfold).

Try antibody structure prediction in [Google Colab](https://colab.research.google.com/github/Graylab/IgFold/blob/main/IgFold.ipynb).

## Updates
```
 - Version 1.0.1
   - Pre-trained weights are stored in the repository and included in the wheel (1.0.0 wheels shipped without them)
 - Version 1.0.0
   - Requires PyTorch >= 2.0 and transformers >= 4.36 (5.x supported); PyTorch-Lightning is no longer a dependency
   - Model weights in safetensors format (`igfold convert-weights` converts .ckpt files); AntiBERTy weights
     are downloaded from the Hugging Face Hub on first use
   - `igfold` command-line interface; batched prediction and pooled sequence embeddings in the Python API
   - Device selection (`--device cuda:1`; "mps" supported); `IGFOLD_WEIGHTS_DIR` for external weights
   - Locked pixi environments; renumbering uses ANARCII (pip-installable, no HMMER)
   - mmCIF output (`.cif` extension); residue numbering restarts for each chain
   - Input validation (chain ids, residue alphabet, sequence length)
   - Templates are aligned to the input sequences, allowing mutations and missing residues
   - OpenMM >= 8 supported for refinement
   - Fixes to the gradient refinement (chain-boundary masks in the violation losses, coordinates now
     correspond to the final refined frames) and to backbone O placement at chain ends
```

## Installation

IgFold is managed with [pixi](https://pixi.sh), which installs Python, PyTorch and the optional
refinement and renumbering dependencies (OpenMM, pdbfixer, ANARCII) into a locked environment inside
the repository. [Install pixi](https://pixi.sh/latest/#installation), then:

```bash
git clone git@github.com:Graylab/IgFold.git
cd IgFold
pixi install -e full
```

Three environments are defined in `pyproject.toml`:

| Environment | Command | Contents |
| --- | --- | --- |
| `default` | `pixi install` | Structure prediction only (PyTorch, AntiBERTy, Biopython) |
| `full` | `pixi install -e full` | Adds OpenMM + pdbfixer refinement, ANARCII renumbering, and plotting |
| `dev` | `pixi install -e dev` | `full` plus pytest, ruff and pre-commit |

Run anything inside an environment with `pixi run -e <env> <command>`, or open a shell with
`pixi shell -e full`. The `-e full` flag is omitted in the examples below when the default
environment is enough.

### Pre-trained weights

The four pre-trained model weights (`igfold/trained_models/IgFold/igfold_*.safetensors`, 25 MB in
total) are stored in this repository and included in the PyPI package. To use weights from another
location, point the `IGFOLD_WEIGHTS_DIR` environment variable (or `--weights-dir`) at a directory
containing them. Legacy `.ckpt` weights from IgFold <= 0.4.0 are converted automatically on first
use (or explicitly with `igfold convert-weights`).

The AntiBERTy language-model weights (about 100 MB) are downloaded from
[huggingface.co/jeffruffolo/AntiBERTy](https://huggingface.co/jeffruffolo/AntiBERTy) on first use
and cached in `~/.cache/huggingface`; set `ANTIBERTY_WEIGHTS_DIR` to use a local copy offline.

### Installing with pip

IgFold (with the pre-trained weights) is also on PyPI, for Python >= 3.11 and PyTorch >= 2.0:

```bash
pip install igfold                       # prediction only
pip install "igfold[refine,renum,viz]"   # with OpenMM refinement, ANARCII renumbering and plotting
```

### Renumbering and truncation

Chothia renumbering (`--renumber`) and trimming of full-length chains to the variable domain
(`--truncate`) use [ANARCII](https://github.com/oxpig/ANARCII), a pip-installable numbering tool
(`pixi install -e full`, or `pip install "igfold[renum]"`). Its weights ship with the package, so no
external binaries or databases are needed.

### PyRosetta refinement

The manuscript refined structures with [PyRosetta](http://pyrosetta.org/downloads), which is
distributed under its own license and is not installed by pixi. If PyRosetta is importable in the
environment, `--refine pyrosetta` uses it; otherwise use `--refine openmm`.

## Quick start

Predict a paired Fv, refine it with OpenMM, and renumber it (Chothia):

```bash
pixi run -e full igfold fold \
  -H EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS \
  -L DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK \
  -o my_antibody.pdb --refine openmm --renumber
```

A nanobody (or a single heavy or light chain) is predicted by giving one sequence:

```bash
pixi run igfold fold \
  -H QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS \
  -o my_nanobody.cif
```

`pixi run fold ...` is a shortcut for `pixi run igfold fold ...`.

## Command-line reference

### `igfold fold`

Predicts one structure from sequences given on the command line or in a FASTA file. Run
`igfold fold --help` for the full option list.

**Input.** Give exactly one of the following. Sequences must be antibody variable domains (Fv,
roughly 110-130 residues per chain) made of the 20 standard amino acids; lowercase and whitespace
are accepted.

| Option | Description |
| --- | --- |
| `-H`, `--heavy SEQ` | Heavy chain (or nanobody) sequence, written as chain `H`. Combine with `-L` for a paired Fv. |
| `-L`, `--light SEQ` | Light chain sequence, written as chain `L`. |
| `--chain ID SEQ` | Any chain id (one character) and sequence; repeat for several chains. |
| `--fasta FILE` | FASTA file with one record per chain. Record ids (or the part after a `:`) become chain ids. |
| `--truncate` | Trim each sequence to its variable domain (ANARCII numbering) before prediction. Use this if you have full-length chains. |

**Output.**

| Option | Description |
| --- | --- |
| `-o`, `--output FILE` | Required. `.pdb` writes PDB; `.cif` or `.mmcif` writes mmCIF. Per-residue predicted RMSD (Å) is stored in the B-factor column. Residue numbering starts at 1 for each chain unless `--renumber` is given. |
| `--renumber` | Renumber residues with the Chothia scheme (ANARCII). |
| `-q`, `--quiet` | Suppress progress messages. |

**Refinement.** IgFold predicts backbone and CB atoms; refinement builds and relaxes the full-atom
structure while restraining the predicted backbone.

| Option | Description |
| --- | --- |
| `--refine none` | Default. Backbone-only output (N, CA, C, O, CB). Takes a few seconds on CPU. |
| `--refine openmm` | Full-atom refinement with OpenMM and pdbfixer (`full` environment). |
| `--refine pyrosetta` | Full-atom refinement with PyRosetta, as in the manuscript (must be installed separately). |

**Templates.** A template structure guides the prediction, for example a parent antibody when
predicting a point mutant. Template chains are matched to the input by chain id (`H`/`L`, or the
legacy `A`/`B`) and then aligned to the sequences, so missing residues and mutations are tolerated.

| Option | Description |
| --- | --- |
| `--template PDB` | Template structure. |
| `--ignore-cdrs [CDR ...]` | Predict these CDRs without the template: any of `h1 h2 h3 l1 l2 l3` (Chothia definitions, requires a Chothia-numbered template with chains `H`/`L`). With no value, all six CDRs are ignored. |
| `--ignore-chain ID` | Predict this whole chain without the template. |

**Model and hardware.**

| Option | Description |
| --- | --- |
| `--num-models N` | Number of ensemble models to run, 1 to 4 (default 4). The prediction with the lowest predicted RMSD is kept. |
| `--device DEV` | `cpu`, `cuda`, `cuda:1`, `mps`, ... Default: the first CUDA device if available, otherwise CPU. |
| `--weights-dir DIR` | Directory containing `igfold_*.safetensors` (default: packaged weights or `IGFOLD_WEIGHTS_DIR`). |

Examples:

```bash
# FASTA input, mmCIF output, single model on a specific GPU
pixi run igfold fold --fasta my_antibody.fasta -o my_antibody.cif --num-models 1 --device cuda:1

# Full-length chains: trim to the Fv, refine and renumber
pixi run -e full igfold fold --fasta full_chains.fasta -o fv.pdb --truncate --refine openmm --renumber

# Predict a mutant using the parent structure as a template, except for CDR H3
pixi run igfold fold -H <mutant heavy> -L <light> --template parent.pdb --ignore-cdrs h3 -o mutant.pdb
```

### `igfold convert-weights`

Converts legacy IgFold <= 0.4.0 `.ckpt` weight files to safetensors, writing a `.safetensors` file
next to each input. Without arguments it converts every `.ckpt` in the weights directory.

```bash
pixi run igfold convert-weights                    # all .ckpt files in the weights directory
pixi run igfold convert-weights path/to/igfold_1.ckpt
```

## Python API

The same functionality is available from Python, including batched prediction of many antibodies,
per-residue and pooled sequence embeddings, and access to the raw coordinates and predicted RMSD:

```python
from igfold import IgFoldRunner

igfold = IgFoldRunner()
out = igfold.fold("my_antibody.pdb", sequences={"H": heavy, "L": light}, do_refine=False, do_renum=True)
out.coords  # (1, L, 5, 3) N, CA, C, CB, O
out.prmsd   # (1, L, 4) predicted RMSD for N, CA, C, CB
```

See [docs/python_api.md](docs/python_api.md) for the full reference.

## Development

```bash
pixi install -e dev
pixi run -e dev test      # pytest (tests needing weights are skipped if none are installed)
pixi run -e dev lint      # pre-commit: ruff lint + format, whitespace and file checks
pixi run -e dev pre-commit install
```

Predictions are checked against reference structures in `tests/data/`; CI runs the test suite on
Linux and macOS.

## Synthetic antibody structures

To demonstrate the capabilities of IgFold for large-scale prediction of antibody structures, we applied the model to two sets of natural paired antibody sequences.

The first set contains 104K non-redundant paired antibody sequences from the Observed Antibody Space database. These predicted structures are made available for use [online](https://data.graylab.jhu.edu/OAS_paired.tar.gz).

```bash
wget https://data.graylab.jhu.edu/OAS_paired.tar.gz
```

The second set contains 1.3M unique paired antibodies from four human donors, collected by [Jaffe et al.](https://www.nature.com/articles/s41586-022-05371-z). These predicted structures are made available for use [online](https://data.graylab.jhu.edu/Jaffe2022.tar.gz).

```bash
wget https://data.graylab.jhu.edu/Jaffe2022.tar.gz
```

## Bug reports

If you run into any problems while using IgFold, please create a [Github issue](https://github.com/Graylab/IgFold/issues) with a description of the problem and the steps to reproduce it.

## Citing this work

```bibtex
@article{ruffolo2023fast,
  title={Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies},
  author={Ruffolo, Jeffrey A and Chu, Lee-Shin and Mahajan, Sai Pooja and Gray, Jeffrey J},
  journal={Nature communications},
  volume={14},
  number={1},
  pages={2389},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
@article{ruffolo2021deciphering,
    title = {Deciphering antibody affinity maturation with language models and weakly supervised learning},
    author = {Ruffolo, Jeffrey A and Gray, Jeffrey J and Sulam, Jeremias},
    journal = {arXiv},
    year= {2021}
}
```
