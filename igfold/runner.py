import os
import warnings
from time import time
from typing import List, Optional, Union

import torch
from antiberty import AntiBERTyRunner

from igfold.utils.checkpoint import find_weights, load_model
from igfold.utils.embed import embed, pool_embeddings
from igfold.utils.folding import fold, fold_batch, get_sequence_dict, predict_structures, validate_sequences
from igfold.utils.general import exists

MAX_NUM_MODELS = 4


LICENSE_URL = "https://github.com/Graylab/IgFold/blob/main/LICENSE.md"
COMMERCIAL_LICENSE_URL = (
    "https://jhtv.e-lucid.com/product/antibody-structure-prediction-from-pre-trained-language-model-igfold"
)


def display_license():
    print(
        "IgFold code and pre-trained models are available for non-commercial use (including at commercial "
        f"entities) under the JHU Academic Software License Agreement ({LICENSE_URL}). "
        f"For commercial use, obtain a license through Johns Hopkins Technology Ventures ({COMMERCIAL_LICENSE_URL})."
    )


def resolve_device(device=None, try_gpu: bool = True) -> torch.device:
    """
    Pick the device to run on.

    :param device: Explicit device ("cpu", "cuda", "cuda:1", "mps", or a torch.device).
    :param try_gpu: When no device is given, use CUDA if available. (MPS is not selected
        automatically: for a model this small the CPU is faster on Apple silicon.)
    """
    if exists(device):
        return torch.device(device)
    if try_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class IgFoldRunner:
    """
    Wrapper for IgFold model predictions.
    """

    def __init__(
        self,
        num_models: int = MAX_NUM_MODELS,
        model_ckpts: Optional[List[str]] = None,
        try_gpu: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        weights_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize IgFoldRunner.

        :param num_models: Number of pre-trained IgFold models to use for prediction (1-4).
        :param model_ckpts: Explicit list of weight files to use (instead of the pre-trained set).
        :param try_gpu: Use CUDA if available (ignored when `device` is given).
        :param device: Device to run on, e.g. "cpu", "cuda:1" or "mps".
        :param weights_dir: Directory containing pre-trained weights (default: packaged weights,
            or the directory named by the IGFOLD_WEIGHTS_DIR environment variable).
        :param verbose: Print progress messages.
        """
        self.verbose = verbose
        if verbose:
            display_license()

        if exists(model_ckpts):
            if len(model_ckpts) == 0:
                raise ValueError("model_ckpts is empty.")
            for p in model_ckpts:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Model weights not found: {p}")
            model_ckpts = list(model_ckpts)
        else:
            if num_models < 1 or num_models > MAX_NUM_MODELS:
                raise ValueError(f"num_models must be between 1 and {MAX_NUM_MODELS}.")
            model_ckpts = find_weights(weights_dir, num_models=num_models)
            if len(model_ckpts) < num_models:
                warnings.warn(f"Requested {num_models} models but found only {len(model_ckpts)} weight files.")

        self.device = resolve_device(device, try_gpu=try_gpu)

        self.models = [load_model(ckpt_file, device=self.device) for ckpt_file in model_ckpts]
        self._log(f"Loaded {len(self.models)} IgFold models on {self.device}.")

        self.antiberty = AntiBERTyRunner(device=self.device)
        self._log("Loaded AntiBERTy.")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @property
    def num_models(self) -> int:
        return len(self.models)

    def fold(
        self,
        pdb_file,
        fasta_file=None,
        sequences=None,
        template_pdb=None,
        ignore_cdrs=None,
        ignore_chain=None,
        skip_pdb=False,
        do_refine=True,
        use_openmm=False,
        do_renum=True,
        truncate_sequences=False,
    ):
        """
        Predict antibody structure with IgFold.

        :param pdb_file: Output structure file (.pdb, or .cif/.mmcif for mmCIF).
        :param fasta_file: FASTA file containing sequences.
        :param sequences: Dictionary of sequences.
        :param template_pdb: PDB file containing template structure.
        :param ignore_cdrs: List of CDRs to ignore.
        :param ignore_chain: Chain to ignore.
        :param skip_pdb: Skip PDB processing.
        :param do_refine: Perform PyRosetta refinement.
        :param use_openmm: Use OpenMM instead of PyRosetta for refinement.
        :param do_renum: Renumber the output to the Chothia scheme.
        :param truncate_sequences: Truncate sequences to the variable domain before prediction.
        """
        start_time = time()
        model_out = fold(
            self.antiberty,
            self.models,
            pdb_file=pdb_file,
            fasta_file=fasta_file,
            sequences=sequences,
            template_pdb=template_pdb,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
            skip_pdb=skip_pdb,
            do_refine=do_refine,
            use_openmm=use_openmm,
            do_renum=do_renum,
            truncate_sequences=truncate_sequences,
            log=self._log,
        )

        self._log(f"Completed folding in {time() - start_time:.2f} seconds.")

        return model_out

    def fold_batch(
        self,
        out_files: List[str],
        sequences: List[dict],
        batch_size: int = 8,
        do_refine=True,
        use_openmm=False,
        do_renum=True,
        truncate_sequences=False,
    ):
        """
        Predict structures for many antibodies at once, batching the network forward pass
        (refinement and renumbering still run per structure). Templates are not supported here.

        :param out_files: One output structure file per antibody (.pdb, .cif or .mmcif).
        :param sequences: One sequence dict per antibody, e.g. [{"H": ..., "L": ...}, ...].
        :param batch_size: Maximum antibodies per forward pass.
        :return: List of IgFoldOutput, one per antibody.
        """
        start_time = time()
        outputs = fold_batch(
            self.antiberty,
            self.models,
            out_files,
            sequences,
            batch_size=batch_size,
            do_refine=do_refine,
            use_openmm=use_openmm,
            do_renum=do_renum,
            truncate_sequences=truncate_sequences,
            log=self._log,
        )
        self._log(f"Completed folding {len(sequences)} structures in {time() - start_time:.2f} seconds.")

        return outputs

    def predict(self, sequences: List[dict], batch_size: int = 8):
        """
        Predict coordinates for many antibodies without writing files.

        :return: List of IgFoldOutput (``coords`` (1, L, 5, 3) for N, CA, C, CB, O; ``prmsd`` (1, L, 4)).
        """
        return predict_structures(self.antiberty, self.models, sequences, batch_size=batch_size)

    def sequence_embedding(
        self,
        sequences=None,
        fasta_file=None,
        model_idx=0,
        reduce: str = "mean",
    ):
        """
        Fixed-size embeddings per chain (and for all chains together), pooled over residues.

        :return: dict of chain id -> {"bert_embs": (512,), "gt_embs": (64,), "structure_embs": (64,)},
            plus an "all" entry.
        """
        seq_dict = validate_sequences(get_sequence_dict(sequences, fasta_file))
        out = self.embed(model_idx=model_idx, sequences=seq_dict)

        return pool_embeddings(out, seq_dict, reduce=reduce)

    def embed(
        self,
        model_idx=0,
        fasta_file=None,
        sequences=None,
        template_pdb=None,
        ignore_cdrs=None,
        ignore_chain=None,
    ):
        """
        Embed antibody sequences with IgFold.

        :param model_idx: Index of the loaded model to use.
        :param fasta_file: FASTA file containing sequences.
        :param sequences: Dictionary of sequences.
        :param template_pdb: PDB file containing template structure.
        :param ignore_cdrs: List of CDRs to ignore.
        :param ignore_chain: Chain to ignore.
        """

        start_time = time()
        model_out = embed(
            self.antiberty,
            self.models[model_idx],
            fasta_file=fasta_file,
            sequences=sequences,
            template_pdb=template_pdb,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
        )

        self._log(f"Completed embedding in {time() - start_time:.2f} seconds.")

        return model_out
