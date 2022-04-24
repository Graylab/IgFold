import pyrosetta


def get_vh_vl_orientation(pose_1):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)

    #vl_vh_distance, opening_angle, opposite_angle, packing_angle
    results = pyrosetta.protocols.antibody.vl_vh_orientation_coords(
        pose_1,
        pose_i1,
    )

    results_labels = [
        'vl-vh_distance', 'opening_angle', 'opposite_angle', 'packing_angle'
    ]
    results_dict = {}
    for i in range(4):
        results_dict[results_labels[i]] = results[i + 1]
    return results_dict


def get_ab_metrics(
    pose_1,
    pose_2,
):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    results = pyrosetta.rosetta.protocols.antibody.cdr_backbone_rmsds(
        pose_1,
        pose_2,
        pose_i1,
        pose_i2,
    )

    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for i in range(9):
        results_dict[results_labels[i]] = results[i + 1]

    return results_dict


def get_pose_cdr_clusters(pose):
    clus_name = {1: 'h1', 2: 'h2', 3: 'h3', 4: 'l1', 5: 'l2', 6: 'l3'}
    ab_info = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(
        pose, pyrosetta.rosetta.protocols.antibody.CDRDefinitionEnum.North)
    ab_info.setup_CDR_clusters(pose)

    clusters = {}
    for enum in range(1, ab_info.get_total_num_CDRs() + 1):
        result = ab_info.get_CDR_cluster(
            pyrosetta.rosetta.protocols.antibody.CDRNameEnum(enum))
        clus = ab_info.get_cluster_name(result.cluster())
        clusters[clus_name[enum]] = clus

    return clusters