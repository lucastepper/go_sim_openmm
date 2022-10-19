import sys
from .config import Config
from .openmm_sim import Simulation


def run_sim():
    """Load config and run simulation"""
    if len(sys.argv) < 2:
        print("Please provide the path to a config file or 'help' as first argument")
        sys.exit(1)
    if sys.argv[1] == "help":
        print(Config.help())
        return
    config = Config.from_toml(sys.argv[1])
    sim = Simulation(config)
    sim.set_positions()
    sim.print_sim_info()
    sim.run()
