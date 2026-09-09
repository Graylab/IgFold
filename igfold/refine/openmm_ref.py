import openmm
import openmm.app
import openmm.unit
import pdbfixer

ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms
# OpenMM >= 8 expresses the minimizer tolerance as a force (energy / length) rather than an energy.
FORCE = openmm.unit.kilojoules_per_mole / openmm.unit.nanometer


def _tolerance_quantity(tolerance):
    """Convert a kcal/mol tolerance to the unit expected by the installed OpenMM version."""
    if int(openmm.__version__.split(".")[0]) >= 8:
        # 2.39 kcal/mol was the OpenMM 7 default; 10 kJ/mol/nm is the OpenMM 8 default.
        return (tolerance / 2.39) * 10.0 * FORCE
    return tolerance * ENERGY


def refine(pdb_file, stiffness=10.0, tolerance=2.39, use_gpu=False):
    tolerance = _tolerance_quantity(tolerance)
    stiffness = stiffness * ENERGY / (LENGTH**2)

    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)
    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["N", "CA", "C", "CB"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True,
        )
