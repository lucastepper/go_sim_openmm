import argparse
import numpy as np
import openmm
import simtk.openmm.app as app
import mdtraj as md
import tqdm


def get_system(args):
    """Get system the cmd arguments."""
    ### hardcode some stuff here, should probably be a config file, later ###
    mass = 39.9 * openmm.unit.amu
    charge = 0.0 * openmm.unit.elementary_charge
    sigma = 0.6 * openmm.unit.nanometer
    epsilon = 1e-2 * openmm.unit.kilojoule_per_mole
    l_bonds = 0.21 * openmm.unit.nanometer
    k_bonds = 60000 * openmm.unit.kilojoule_per_mole / openmm.unit.nanometers**2
    sigma_nat_contc = 0.1 * openmm.unit.nanometer
    epsilon_nat_contc = 1e-7 * openmm.unit.kilojoule_per_mole

    system = openmm.System()
    for index in range(args.nparticles):
        # Particles are added one at a time
        # Their indices in the System will correspond with their indices
        # in the Force objects we will add later
        system.addParticle(mass)

    # Add harmonic bond forces for a chain polymer
    bond_force = openmm.HarmonicBondForce()
    for i in range(args.nparticles - 1):
        bond_force.addBond(i, i + 1, l_bonds, k_bonds)
    _bond_force_index = system.addForce(bond_force)

    if not args.use_nonbond == "false":
        nonbond_force = openmm.NonbondedForce()
        nonbond_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        # all particles must have parameters assigned for the NonbondedForce
        for index in range(args.nparticles):
            # Particles are assigned properties in the same order as they appear in the System object
            nonbond_force.addParticle(charge, sigma, epsilon)
        # set cutoff (truncation) distance at 3*sigma,
        nonbond_force.setCutoffDistance(3 * sigma)
        # use a smooth switching function to avoid force discontinuities at cutoff
        nonbond_force.setUseSwitchingFunction(True)
        # turn on switch at 2.5*sigma, if switchfunction used
        nonbond_force.setSwitchingDistance(2.5 * sigma)
        # do not use long-range isotropic dispersion correction, which would improve pressure sampling
        nonbond_force.setUseDispersionCorrection(False)
        _nonbond_force_index = system.addForce(nonbond_force)

    if not args.use_custom_nonbond == "false":
        # Add an interaction that represents the native contancts of the protein
        # We cant add a cutoff, so the energy should go to 0 for r -> inf
        # Here, I just use an LJ interaction Engergy!
        native_contacts_force = openmm.CustomBondForce(
            "4 * epsilon_nat_contc * ( (sigma_nat_contc / r)^12 - (sigma_nat_contc / r)^6 )"
        )
        native_contacts_force.addPerBondParameter("epsilon_nat_contc")
        native_contacts_force.addPerBondParameter("sigma_nat_contc")
        # I add contacts between the ends
        native_contacts_force.addBond(0, args.nparticles - 1, [epsilon_nat_contc, sigma_nat_contc])
        native_contacts_force.addBond(1, args.nparticles - 1, [epsilon_nat_contc, sigma_nat_contc])
        native_contacts_force.addBond(0, args.nparticles - 2, [epsilon_nat_contc, sigma_nat_contc])
        native_contacts_force.addBond(1, args.nparticles - 2, [epsilon_nat_contc, sigma_nat_contc])
        _native_contacts_force_index = system.addForce(native_contacts_force)

    # Get the number of particles in the System
    print("\nThe system has %d particles" % system.getNumParticles())
    # Print a few particle masses
    for index in range(min(args.nparticles, 1)):
        print(
            "Particle %5d has mass %12.3f amu"
            % (index, system.getParticleMass(index) / openmm.unit.amu)
        )
    # Print number of constraints
    print("The system has %d constraints" % system.getNumConstraints())
    # Get the number of forces and iterate through them
    print("There are %d forces" % system.getNumForces())
    for (index, force) in enumerate(system.getForces()):
        print("Force %5d : %s" % (index, force.__class__.__name__))
    print()
    return system


def get_simulation(args, system, dt, temperature, friction):
    """ Get the Simulation object, hack together a topology. """
    top = app.Topology()
    chain = top.addChain()
    for i in range(args.nparticles):
        res = top.addResidue(f"RES{i}", chain)
        _atom = top.addAtom(f"A{i}", app.element.sulfur, res)
    # This is BAOAB integrator
    integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    # Create a Context using the multithreaded mixed-precision CPU platform
    platform = openmm.Platform.getPlatformByName(args.platform)
    sim = app.Simulation(top, system, integrator, platform)
    return sim


def set_pos_init(sim, args):
    """ Init all particles on a chain. """
    positions = np.zeros([args.nparticles, 3]) * openmm.unit.nanometer
    positions[:, 0] = np.arange(args.nparticles) * 0.21 * openmm.unit.nanometer
    sim.context.setPositions(positions)
    # get positions
    # sim.context.getState(get_positions=True).getPositions(asNumpy=True)


def add_state_reporter(sim, args):
    """ Save traj in mdf5 format. """
    reporter = md.reporters.HDF5Reporter(args.save_file, args.save_interval)
    sim.reporters.append(reporter)
    return reporter


def main():
    """Get simulation parameters from command line and run simulation"""
    dt = 2 * openmm.unit.femtosecond
    friction = 5.0 / openmm.unit.picosecond
    temperature = 300.0 * openmm.unit.kelvin
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_file", "-f", type=str, default="/tmp/traj.h5")
    parser.add_argument("--platform", "-p", type=str, default="CPU")
    parser.add_argument("--nsteps", "-s", type=int, default=1000)
    parser.add_argument("--save_interval", "-i", type=int, default=1)
    parser.add_argument("--nparticles", "-n", type=int, default=10)
    parser.add_argument("--use_nonbond", type=str, default="false")
    parser.add_argument("--use_custom_nonbond", type=str, default="false")
    args = parser.parse_args()
    print(f"Running simulations with {args=}.")
    system = get_system(args)
    sim = get_simulation(args, system, dt, temperature, friction)
    set_pos_init(sim, args)
    reporter = add_state_reporter(sim, args)
    pbar = tqdm.tqdm(range(args.nsteps), desc="Running simulation")
    for _i in range(args.nsteps // 100):
        sim.step(100)
        pbar.update(100)
    pbar.close()
    reporter.close()


if __name__ == "__main__":
    main()
