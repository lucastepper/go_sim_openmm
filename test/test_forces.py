from pathlib import Path
import pytest
import numpy as np
import go_sim_openmm


def get_test_dir():
    parent = Path(__file__).parent
    if not str(parent).endswith("test"):
        parent = parent / "test"
    return parent


@pytest.mark.parametrize(
    "positions, ref_forces, use_bond_forces, use_nonbond_forces, use_native_contc_forces",
    [
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros((2, 3)), True, False, False),
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros((2, 3)), False, True, False),
    ],
)
def test_forces(
    positions, ref_forces, use_bond_forces, use_nonbond_forces, use_native_contc_forces
):
    """Test that forces are correctly set, by comparing the force
    of go_sim_openmm.Simulation to the force to a predefined reference
    value. Switch the indivudual force contributions on/off.
    Arguments:
        positions (np.NDArray[float]): positions of the system
        ref_force (np.NDArray[float]): reference force to check against
        use_bond_forces (bool): use this use bond_forces
        use_nonbond_forces (bool): use this use nonbond_forces
        use_native_contc_forces (bool): use this use native_contc_forces
    """
    config = go_sim_openmm.Config.from_toml(get_test_dir() / "data/config.toml")
    if positions.shape[1] != 3 or positions.ndim != 2:
        raise ValueError("Positions must be of shape (nparticles, 3)")
    if ref_forces.shape != positions.shape:
        raise ValueError("Ref_forces and positions must be of same shape")
    config.nparticles = positions.shape[0]
    config.use_bond_forces = use_bond_forces
    config.use_nonbond_forces = use_nonbond_forces
    config.use_native_contc_forces = use_native_contc_forces
    sim = go_sim_openmm.Simulation(config)
    sim.sim = sim.get_simulation()
    sim.set_positions(positions=positions)
    forces = sim.get_forces()
    print(type(forces))
    np.testing.assert_allclose(forces, ref_forces)
