# IgFold Python API

All examples assume an environment with IgFold installed (see the [README](../README.md)); with
pixi, run scripts as `pixi run -e full python my_script.py`.

```python
from igfold import IgFoldRunner
```

## `IgFoldRunner`

Loads the pre-trained IgFold models and the AntiBERTy language model once; create it at the start
of a session and reuse it for every prediction.

```python
igfold = IgFoldRunner(
    num_models=4,        # ensemble size, 1-4; the prediction with the lowest predicted RMSD is kept
    device=None,         # "cpu", "cuda", "cuda:1", "mps", ...; default: CUDA if available, else CPU
    weights_dir=None,    # directory with igfold_*.safetensors; default: packaged weights or $IGFOLD_WEIGHTS_DIR
    model_ckpts=None,    # explicit list of weight files, overriding num_models/weights_dir
    verbose=True,        # print license notice and progress
)
```

`IgFoldRunner.models` is the list of loaded `igfold.IgFold` modules and `IgFoldRunner.antiberty`
the `antiberty.AntiBERTyRunner`; both live on `IgFoldRunner.device`. Note that on Apple silicon the
CPU is faster than `mps` for a model of this size.

## Predicting a structure: `fold`

```python
heavy = "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"
light = "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"

out = igfold.fold(
    "my_antibody.pdb",                   # output file; .pdb, or .cif/.mmcif for mmCIF
    sequences={"H": heavy, "L": light},  # chain id -> sequence (or fasta_file="...")
    do_refine=True,                      # full-atom refinement (default True)
    use_openmm=True,                     # OpenMM instead of PyRosetta for refinement
    do_renum=True,                       # Chothia renumbering with ANARCII (default True)
    truncate_sequences=False,            # trim inputs to the Fv with ANARCII first
    template_pdb=None,                   # optional template structure, see below
    skip_pdb=False,                      # True: return the prediction without writing a file
)
```

A nanobody or a single chain is predicted by passing one sequence, e.g. `sequences={"H": heavy}`.

### Inputs

- `sequences` maps single-character chain ids to sequences; `H` and `L` are conventional but any
  alphanumeric character works. Sequences are upper-cased and whitespace is removed. Only the 20
  standard amino acids are accepted.
- `fasta_file` may be given instead; record ids (or the text after a `:` in the id) become chain ids.
- IgFold is trained on antibody variable domains (Fv). Chains longer than 150 residues produce a
  warning, and longer than 510 residues an error; use `truncate_sequences=True` to trim full-length
  chains to the Fv (requires ANARCII).
- Invalid input raises `ValueError` before the model runs and no files are written.

### Output files

The format follows the extension: `.pdb` for PDB, `.cif`/`.mmcif` for mmCIF. Files contain N, CA,
C, O and CB atoms (all atoms after refinement), chains in the order of `sequences`, residue numbers
starting at 1 per chain (or Chothia numbers with `do_renum=True`), and the per-residue predicted
RMSD in the B-factor column.

### Return value: `IgFoldOutput`

| Field | Shape | Meaning |
| --- | --- | --- |
| `coords` | `(1, L, 5, 3)` | Backbone coordinates for N, CA, C, CB, O over all chains concatenated |
| `prmsd` | `(1, L, 4)` | Predicted RMSD (Å) of N, CA, C, CB per residue; `prmsd.square().mean(-1).sqrt()` is the per-residue value written to the B-factor column |
| `translations`, `rotations` | `(1, L, 3)`, `(1, L, 3, 3)` | Residue frames |
| `bert_embs` | `(1, L, 512)` | AntiBERTy final-layer embeddings |
| `gt_embs` | `(1, L, 64)` | Node features after the graph transformer |
| `structure_embs` | `(1, L, 64)` | Node features after template incorporation |

Coordinates are those of the unrefined backbone even when refinement was requested; refinement
changes the written file only.

### Renumbering and truncation

`do_renum=True` renumbers the written structure with the Chothia scheme and `truncate_sequences=True`
trims each input to its numbered variable domain. Both use ANARCII (`pip install anarcii`, or the
`renum` extra); the functions are also available directly:

```python
from igfold.utils.numbering import number_sequences, renumber_pdb, truncate_seq

truncate_seq(full_length_heavy)                 # Fv sequence
renumber_pdb("pred.pdb", "pred_chothia.pdb")    # Chothia-numbered copy
numbering, start, end = number_sequences([heavy], scheme="chothia")[0]  # ((number, insertion), residue) pairs
```

### Refinement

- `do_refine=True, use_openmm=False`: PyRosetta minimization and repacking with backbone
  restraints, as in the manuscript. Requires PyRosetta.
- `do_refine=True, use_openmm=True`: pdbfixer builds side chains and hydrogens, then OpenMM
  minimizes with harmonic restraints on the predicted backbone. Requires OpenMM >= 8 and pdbfixer.
- `do_refine=False`: backbone-only output, a few seconds on CPU.

A missing backend raises `igfold.utils.folding.MissingDependencyError` (a subclass of
`ImportError`) naming what to install.

### Templates

```python
out = igfold.fold(
    "mutant.pdb",
    sequences={"H": mutant_heavy, "L": light},
    template_pdb="parent.pdb",
    ignore_cdrs=["h3"],   # list of h1 h2 h3 l1 l2 l3, or True for all six, or None
    ignore_chain=None,    # e.g. "L" to predict that chain without template information
    do_refine=False,
    do_renum=True,
)
```

Template chains are matched to sequence chain ids (identical ids first, then the legacy `A`->`H`,
`B`->`L` mapping, then a lone chain to a lone sequence) and aligned to the sequences with
Biopython's `PairwiseAligner`, so point mutations and missing residues are handled; a warning is
issued below 50% identity. CDR masking uses the Chothia residue numbers of the template, so the
template must be Chothia-numbered with chains `H`/`L` for `ignore_cdrs`.

## Predicting many antibodies: `fold_batch` and `predict`

Antibodies with the same set of chain ids are padded into batches for the network forward pass;
gradient refinement, model selection, and file writing run per antibody, so results match `fold`.
Templates are not supported in batch mode.

```python
sequences = [
    {"H": heavy, "L": light},
    {"H": other_heavy, "L": other_light},
    {"H": nanobody},
]

# predict and write one file per antibody
outputs = igfold.fold_batch(
    ["ab1.pdb", "ab2.cif", "nb.pdb"],
    sequences,
    batch_size=8,
    do_refine=False,
    do_renum=True,
)

# coordinates only, no files
outputs = igfold.predict(sequences, batch_size=8)
```

Both return a list of `IgFoldOutput` in input order. Refinement dominates run time, so batching
mostly helps when `do_refine=False` or for embeddings.

## Embeddings: `embed` and `sequence_embedding`

```python
emb = igfold.embed(sequences={"H": heavy, "L": light}, model_idx=0)
emb.bert_embs       # (1, L, 512) AntiBERTy final hidden layer
emb.gt_embs         # (1, L, 64)  after the graph transformer
emb.structure_embs  # (1, L, 64)  after template incorporation
emb.prmsd           # (1, L, 4)   predicted RMSD from the single forward pass (no gradient refinement)

pooled = igfold.sequence_embedding(sequences={"H": heavy, "L": light}, reduce="mean")  # or "max"
pooled["H"]["gt_embs"]      # (64,)  heavy chain, pooled over residues
pooled["all"]["bert_embs"]  # (512,) both chains together
```

`embed` accepts the same `template_pdb`, `ignore_cdrs` and `ignore_chain` arguments as `fold`.
Because the graph transformer mixes information between chains, the heavy-chain embedding of a
paired Fv differs from the embedding of the heavy chain alone.

## Weights

```python
from igfold.utils.checkpoint import find_weights, load_model, save_model, convert_legacy_checkpoint

find_weights()                      # list of packaged igfold_*.safetensors (honors $IGFOLD_WEIGHTS_DIR)
model = load_model(find_weights()[0], device="cpu")   # an igfold.IgFold nn.Module in eval mode
save_model(model, "my_igfold.safetensors")
convert_legacy_checkpoint("igfold_1.ckpt")           # -> igfold_1.safetensors
```

Weight files are safetensors with the architecture config stored as JSON in the file metadata
(`igfold.utils.checkpoint.MODEL_CONFIG_KEYS`). `find_weights` converts legacy `.ckpt` files
automatically if no safetensors files are present.

## Visualization

With the `viz` extra (`pixi install -e full`, or `pip install py3Dmol matplotlib seaborn`):

```python
from igfold.utils.visualize import show_pdb, plot_prmsd

show_pdb("my_antibody.pdb", num_sequences=2, color="b")           # py3Dmol view colored by pRMSD
plot_prmsd({"H": heavy, "L": light}, out.prmsd.cpu(), "prmsd.png")  # per-residue pRMSD plot
```

## Errors

| Exception | Cause |
| --- | --- |
| `ValueError` | Invalid chain ids, residues or lengths; unsupported output extension; renumbering a sequence that extends beyond the Fv; mismatched template chains |
| `igfold.utils.folding.MissingDependencyError` | PyRosetta, OpenMM/pdbfixer or ANARCII requested but not importable |
| `FileNotFoundError` | No weights found (see `IGFOLD_WEIGHTS_DIR`) |
