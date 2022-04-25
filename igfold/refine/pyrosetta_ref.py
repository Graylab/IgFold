import pyrosetta

from igfold.utils.general import exists


def init_pyrosetta(init_string=None):
    if not exists(init_string):
        init_string = "-mute all -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string)


def get_min_mover(
        max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom minimization mover
    """

    sf = pyrosetta.create_score_function('ref2015_cst')
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded,
        1,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
        0,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint,
        1,
    )

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap,
        sf,
        'lbfgs_armijo_nonmonotone',
        0.0001,
        True,
    )
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def refine(pdb):
    pose = pyrosetta.pose_from_pdb(pdb)

    cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
    cst_mover.cst_sidechain(False)
    cst_mover.apply(pose)

    min_mover = get_min_mover(50)
    min_mover.apply(pose)
    idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
    idealize_mover.apply(pose)

    pose.dump_pdb(pdb)