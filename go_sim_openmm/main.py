from .config import Config
from .openmm_sim import Simulation


def run_sim(config: str):
    """Load config and run simulation"""
    config = Config.from_toml(config)
    sim = Simulation(config)
    sim.sim = sim.get_simulation()
    sim.set_positions()
    sim.print_sim_info()
    sim.run()
