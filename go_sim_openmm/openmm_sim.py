from .config import Config
import numpy as np
import openmm
import simtk.openmm.app as app
import mdtraj as md
import tqdm


class Simulation:
    """
    Main Object that handles the simulation. Build the simulation
    object from the config class.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.system = self.get_system()
        if self.config.use_bond_forces:
            self.add_harmonic_bond_force()
        if self.config.use_nonbond_forces:
            self.add_nonbonded_force()
        if self.config.use_native_contc_forces:
            self.add_native_contacts_force()
        self.sim = None
        self.reporter = None

    def get_system(self):
        """Build the system object"""
        system = openmm.System()
        for _ in range(self.config.nparticles):
            system.addParticle(self.config.mass)
        return system

    def add_harmonic_bond_force(self):
        """Add the force representing chain connected via springs."""
        bond_force = openmm.HarmonicBondForce()
        for i in range(self.config.nparticles - 1):
            bond_force.addBond(i, i + 1, self.config.l_bond, self.config.k_bond)
        bond_force_index = self.system.addForce(bond_force)
        return bond_force_index

    def add_nonbonded_force(self):
        """Add the LJ and electrostatic interactions between particles."""
        nonbond_force = openmm.NonbondedForce()
        nonbond_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        # all particles must have parameters assigned for the NonbondedForce
        for index in range(self.config.nparticles):
            # Particles are assigned properties in the same order
            # as they appear in the System object
            nonbond_force.addParticle(self.config.charge, self.config.sigma, self.config.epsilon)
        nonbond_force_index = self.system.addForce(nonbond_force)
        return nonbond_force_index

    def add_native_contacts_force(self):
        """Add an interaction that represents the native contancts of the protein
        We cant add a cutoff, so the energy should go to 0 for r -> inf
        Here, I just use an LJ interaction Engergy.
        """
        native_contacts_force = openmm.CustomBondForce(
            "4 * epsilon_nat_contc * ((sigma_nat_contc / r)^12 - (sigma_nat_contc / r)^6)"
        )
        native_contacts_force.addPerBondParameter("epsilon_nat_contc")
        native_contacts_force.addPerBondParameter("sigma_nat_contc")
        for contact in self.config.native_contc:
            native_contacts_force.addBond(
                contact[0],
                contact[1],
                [self.config.epsilon_nat_contc, self.config.sigma_nat_contc],
            )
        native_contacts_force_index = self.system.addForce(native_contacts_force)
        return native_contacts_force_index

    def print_sim_info(self):
        """Print the system info"""
        print("Running a simulation with the following parameters:\n")
        print(f"{self.config:fancy}\n")
        print(f"Forces in the system:")
        for (index, force) in enumerate(self.system.getForces()):
            print("Force %5d : %s" % (index, force.__class__.__name__))
        print("")

    def get_simulation(self):
        """Get the Simulation object, hack together a topology."""
        top = app.Topology()
        chain = top.addChain()
        for i in range(self.config.nparticles):
            res = top.addResidue(f"RES{i}", chain)
            _atom = top.addAtom(f"A{i}", app.element.sulfur, res)
        # This is BAOAB integrator
        integrator = openmm.LangevinIntegrator(
            self.config.temperature, self.config.friction, self.config.dt
        )
        # Create a Context using the multithreaded mixed-precision CPU platform
        platform = openmm.Platform.getPlatformByName(self.config.platform)
        sim = app.Simulation(top, self.system, integrator, platform)
        return sim

    def set_positions(self, positions=None):
        """Init all particles on a chain, of no other position given."""
        if positions is None:
            positions = np.zeros([self.config.nparticles, 3])
            positions[:, 0] = np.arange(self.config.nparticles) * 0.21
        if self.sim is None:
            self.sim = self.get_simulation()
        self.sim.context.setPositions(positions)

    def get_forces(self):
        """Get the force for the current state."""
        if self.sim is None:
            self.sim = self.get_simulation()
        state = self.sim.context.getState(getForces=True)
        return state.getForces(asNumpy=True)

    def add_state_reporter(self):
        """Save traj in mdf5 format."""
        self.reporter = md.reporters.HDF5Reporter(self.config.save_file, self.config.stride)
        if self.sim is None:
            self.sim = self.get_simulation()
        self.sim.reporters.append(self.reporter)

    def run(self):
        """Run the simulation"""
        pbar = tqdm.tqdm(range(self.config.steps), desc="Running simulation")
        for _i in range(self.config.steps // 100):
            self.sim.step(100)
            pbar.update(100)
        pbar.close()
        self.reporter.close()
