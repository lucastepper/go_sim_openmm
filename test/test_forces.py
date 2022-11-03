from pathlib import Path
import pytest
import numpy as np
import go_sim_openmm
from pydantic import BaseModel

def get_test_dir():
    parent = Path(__file__).parent
    if not str(parent).endswith("test"):
        parent = parent / "test"
    return parent

go_model_positions = np.zeros((9,3))
go_model_positions = np.insert(go_model_positions, 0, [1, 0, 0] , axis = 0)
ref_native_contact_force = [[0.00048, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-0.00024, 0, 0], [-0.00024, 0, 0]]
@pytest.mark.parametrize(
    "positions, ref_forces, use_bond_forces, use_nonbond_forces, use_native_contc_forces",
    [
        (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ([[-54000,0,0], [54000,0,0]]), True, False, False),
        (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ([[24e-5,0,0], [-24e-5,0,0]]), False, True, False),
        (go_model_positions, ref_native_contact_force, False, False, True)
    ],
)
def test_forces(positions, ref_forces, use_bond_forces, use_nonbond_forces, use_native_contc_forces
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
    positions = np.array(positions)
    ref_forces = np.array(ref_forces)
    use_bond_forces = np.array(use_bond_forces)
    use_nonbond_forces = np.array(use_nonbond_forces)  
    use_native_contc_forces = np.array(use_native_contc_forces)


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

#test_forces(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), ([[60000,0,0], [60000,0,0]]), True, False, False)