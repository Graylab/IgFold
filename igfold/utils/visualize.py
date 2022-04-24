import math
import torch
import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
import seaborn as sns

from igfold.utils.folding import get_sequence_dict
from igfold.utils.general import exists
from igfold.utils.pdb import get_cdr_range_dict


def show_pdb(
        pdb_filename: str,
        num_sequences,
        bb_sticks=False,
        sc_sticks=False,
        color="b",
        view_size=(500, 500),
):
    return show_pdbs(
        [pdb_filename],
        num_sequences,
        bb_sticks=bb_sticks,
        sc_sticks=sc_sticks,
        color=color,
        view_size=view_size,
    )


###
#   Inspired by ColabFold visualization from https://github.com/sokrypton/ColabFold
###


def show_pdbs(
        pdb_filenames,
        num_sequences,
        bb_sticks=False,
        sc_sticks=False,
        color="b",
        view_size=(800, 800),
):
    grid_width = math.ceil(math.sqrt(len(pdb_filenames)))
    grid_height = math.ceil(len(pdb_filenames) / grid_width)
    view = py3Dmol.view(
        js="https://3dmol.org/build/3Dmol.js",
        viewergrid=(grid_height, grid_width),
        width=view_size[0],
        height=view_size[1],
    )

    for pdb_i, pdb_filename in enumerate(pdb_filenames):
        grid_row, grid_col = pdb_i // grid_width, pdb_i % grid_width
        view.addModel(
            open(pdb_filename, "r").read(),
            "pdb",
            viewer=(grid_row, grid_col),
        )

    if color == "b":
        view.setStyle({
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 1.5,
                    "max": 0.5,
                }
            }
        })
    elif color == "rainbow":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif color == "chain":
        for n, chain, color_ in zip(
                range(num_sequences),
                list("ABCDEFGH"),
            [
                "lime", "cyan", "magenta", "yellow", "salmon", "white", "blue",
                "orange"
            ],
        ):
            view.setStyle({"chain": chain}, {"cartoon": {"color": color_}})
    if sc_sticks:
        BB = ["C", "O", "N"]
        view.addStyle(
            {
                "and": [
                    {
                        "resn": ["GLY", "PRO"],
                        "invert": True
                    },
                    {
                        "atom": BB,
                        "invert": True
                    },
                ]
            },
            {"stick": {
                "colorscheme": f"WhiteCarbon",
                "radius": 0.2
            }},
        )
        view.addStyle(
            {"and": [{
                "resn": "GLY"
            }, {
                "atom": "CA"
            }]},
            {"sphere": {
                "colorscheme": f"WhiteCarbon",
                "radius": 0.3
            }},
        )
        view.addStyle(
            {"and": [{
                "resn": "PRO"
            }, {
                "atom": ["C", "O"],
                "invert": True
            }]},
            {"stick": {
                "colorscheme": f"WhiteCarbon",
                "radius": 0.3
            }},
        )
    if bb_sticks:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB},
            {"stick": {
                "colorscheme": f"WhiteCarbon",
                "radius": 0.3
            }},
        )

    view.zoomTo()
    return view


def plot_prmsd(
    sequences,
    prmsd,
    out_file=None,
    shade_cdr=False,
    pdb_file=None,
):
    seq_dict = get_sequence_dict(sequences, None)
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()

    res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)
    chain_res_rmsd = np.split(res_rmsd, delims)

    if shade_cdr and exists(pdb_file):
        heavy_only = len(sequences) == 1 and "H" in sequences
        light_only = len(sequences) == 1 and "L" in sequences
        cdr_range_dict = get_cdr_range_dict(
            pdb_file,
            heavy_only=heavy_only,
            light_only=light_only,
            offset_heavy=False,
        )
        cdr_ranges = np.split(np.array(list(cdr_range_dict.values())), [3])

    plt.figure(figsize=(8, 4))
    for i, (chain, rmsd) in enumerate(zip(seq_dict.keys(), chain_res_rmsd)):
        plt.subplot(1, len(seq_dict), i + 1)

        res_nums = torch.arange(1, len(rmsd) + 1)
        sns.lineplot(x=res_nums, y=rmsd)

        if shade_cdr and exists(pdb_file):
            for r in cdr_ranges[i]:
                plt.axvspan(r[0], r[1], color="gray", alpha=0.5)

        plt.xlabel("Residue Number")
        plt.ylabel("Predicted RMSD (A)")
        plt.title(f"Chain {chain}")

    plt.tight_layout()

    if exists(out_file):
        plt.savefig(out_file, dpi=400)
