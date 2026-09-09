"""
Command-line interface: ``igfold fold`` predicts a structure, ``igfold convert-weights``
converts legacy checkpoint files.
"""

import argparse
import sys

from igfold import __version__


def _add_fold_parser(subparsers):
    p = subparsers.add_parser("fold", help="Predict an antibody structure from sequence(s).")
    inp = p.add_argument_group("input (give --heavy/--light, --chain, or --fasta)")
    inp.add_argument("--heavy", "-H", metavar="SEQ", help="Heavy chain (or nanobody) sequence; written as chain H.")
    inp.add_argument("--light", "-L", metavar="SEQ", help="Light chain sequence; written as chain L.")
    inp.add_argument(
        "--chain",
        nargs=2,
        action="append",
        metavar=("ID", "SEQ"),
        default=[],
        help="Arbitrary chain id and sequence (repeatable).",
    )
    inp.add_argument("--fasta", metavar="FILE", help="FASTA file; record ids are used as chain ids.")
    p.add_argument(
        "--output", "-o", required=True, metavar="FILE", help="Output structure file (.pdb, or .cif/.mmcif for mmCIF)."
    )
    p.add_argument(
        "--refine",
        choices=["none", "openmm", "pyrosetta"],
        default="none",
        help="Full-atom refinement backend (default: none).",
    )
    p.add_argument("--renumber", action="store_true", help="Renumber the output residues to the Chothia scheme.")
    p.add_argument("--truncate", action="store_true", help="Trim sequences to the variable domain before prediction.")
    p.add_argument("--template", metavar="PDB", help="Template structure (chains matched by id and alignment).")
    p.add_argument(
        "--ignore-cdrs",
        nargs="*",
        metavar="CDR",
        help="CDRs to mask in the template (h1 h2 h3 l1 l2 l3); no value masks all six.",
    )
    p.add_argument("--ignore-chain", metavar="ID", help="Chain id to mask in the template.")
    p.add_argument(
        "--num-models", type=int, choices=range(1, 5), default=4, metavar="N", help="Ensemble size, 1-4 (default 4)."
    )
    p.add_argument("--device", help='Torch device, e.g. "cpu", "cuda:1", "mps" (default: CUDA if available).')
    p.add_argument("--weights-dir", help="Directory with igfold_*.safetensors (default: packaged weights).")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages.")
    p.set_defaults(func=_run_fold)


def _run_fold(args):
    from igfold import IgFoldRunner

    sequences = {}
    if args.heavy:
        sequences["H"] = args.heavy
    if args.light:
        sequences["L"] = args.light
    for chain_id, seq in args.chain:
        sequences[chain_id] = seq
    if not sequences and not args.fasta:
        raise SystemExit("error: provide sequences with --heavy/--light, --chain, or --fasta")

    ignore_cdrs = None
    if args.ignore_cdrs is not None:
        ignore_cdrs = True if len(args.ignore_cdrs) == 0 else args.ignore_cdrs

    runner = IgFoldRunner(
        num_models=args.num_models,
        device=args.device,
        weights_dir=args.weights_dir,
        verbose=not args.quiet,
    )
    out = runner.fold(
        args.output,
        fasta_file=args.fasta,
        sequences=sequences or None,
        template_pdb=args.template,
        ignore_cdrs=ignore_cdrs,
        ignore_chain=args.ignore_chain,
        do_refine=args.refine != "none",
        use_openmm=args.refine == "openmm",
        do_renum=args.renumber,
        truncate_sequences=args.truncate,
    )
    if not args.quiet:
        res_rmsd = out.prmsd.square().mean(-1).sqrt()
        print(f"Wrote {args.output} (median predicted RMSD {res_rmsd.median().item():.2f} A)")

    return 0


def _add_convert_parser(subparsers):
    p = subparsers.add_parser("convert-weights", help="Convert legacy .ckpt weights to safetensors.")
    p.add_argument("paths", nargs="*", help="Checkpoint files (default: all .ckpt in the weights directory).")
    p.set_defaults(func=_run_convert)


def _run_convert(args):
    import glob
    import os

    from igfold.utils.checkpoint import LEGACY_EXT, convert_legacy_checkpoint, get_default_weights_dir

    paths = args.paths or sorted(glob.glob(os.path.join(get_default_weights_dir(), f"*{LEGACY_EXT}")))
    if len(paths) == 0:
        print(f"No legacy checkpoints found in {get_default_weights_dir()}.")
        return 1
    for p in paths:
        print(f"{p} -> {convert_legacy_checkpoint(p)}")

    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="igfold", description="IgFold: antibody structure prediction.")
    parser.add_argument("--version", action="version", version=f"igfold {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fold_parser(subparsers)
    _add_convert_parser(subparsers)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
