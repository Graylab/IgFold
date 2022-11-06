import pyrosetta

from igfold.utils.general import exists


def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)


def get_min_mover(
        max_iter: int = 1000,
        sf_name: str = "ref2015_cst"
) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom minimization mover
    """

    sf = pyrosetta.create_score_function(sf_name)
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
    mmap.set_jump(False)
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


def get_fa_relax_mover(
        max_iter: int = 200) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom relax mover
    """

    sf = pyrosetta.create_score_function('ref2015_cst')

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf)
    relax.max_iter(max_iter)
    relax.set_movemap(mmap)

    return relax


def get_repack_mover():
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(
    )
    packer.task_factory(tf)

    return packer


def refine(pdb, minimization_iter=100, constrain=True, idealize=False):
    pose = pyrosetta.pose_from_pdb(pdb)

    if constrain:
        cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
        cst_mover.cst_sidechain(False)
        cst_mover.apply(pose)
    min_mover = get_min_mover(max_iter=minimization_iter)
    min_mover.apply(pose)
    if idealize:
        idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
        idealize_mover.apply(pose)
    packer = get_repack_mover()
    packer.apply(pose)

    min_mover.apply(pose)

    pose.dump_pdb(pdb)