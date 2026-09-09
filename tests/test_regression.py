"""Golden-structure regression: predictions must stay within PDB precision of the stored references."""

import os

import numpy as np
import pytest
from conftest import HEAVY, LIGHT, NANOBODY, requires_weights

pytestmark = requires_weights

DATA = os.path.join(os.path.dirname(__file__), "data")


def _read_atoms(path):
    names, xyz, bfac = [], [], []
    for line in open(path):
        if line.startswith("ATOM"):
            names.append((line[12:16].strip(), line[21], int(line[22:26])))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            bfac.append(float(line[60:66]))
    return names, np.array(xyz), np.array(bfac)


@pytest.fixture(scope="module")
def full_runner():
    from igfold import IgFoldRunner

    return IgFoldRunner(num_models=4, device="cpu", verbose=False)


@pytest.mark.parametrize(
    "golden, sequences",
    [("golden_paired.pdb", {"H": HEAVY, "L": LIGHT}), ("golden_nanobody.pdb", {"H": NANOBODY})],
)
def test_matches_golden(full_runner, tmp_path, golden, sequences):
    out = tmp_path / golden
    full_runner.fold(str(out), sequences=sequences, do_refine=False, do_renum=False)

    ref_names, ref_xyz, ref_b = _read_atoms(os.path.join(DATA, golden))
    names, xyz, bfac = _read_atoms(str(out))
    assert names == ref_names
    # coordinates are printed to 0.001 A; allow a little slack for platform-dependent float math
    assert np.abs(xyz - ref_xyz).max() < 0.05
    assert np.abs(bfac - ref_b).max() < 0.05
