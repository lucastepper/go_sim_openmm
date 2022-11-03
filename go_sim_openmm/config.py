import re
import toml
from pydantic import BaseModel
#from typing import list 

class Config(BaseModel):
    steps: int
    stride: int
    nparticles: int
    dt: float
    temperature: float
    friction: float
    mass: float
    charge: float
    epsilon: float
    sigma: float
    l_bond: float
    k_bond: float
    epsilon_nat_contc: float
    sigma_nat_contc: float
    native_contc: list[list[int]]
    use_bond_forces: bool = True
    use_nonbond_forces: bool = True
    use_native_contc_forces: bool = True
    platform: str = "CPU"
    save_file: str = "/tmp/traj.h5"

    @classmethod
    def help(_cls):
        return """
    Configuration file for the GO-Style simulation using OpenMM. The
    simulation particles are part of a harmonic chain and interact
    via a Lennard-Jones force. Furthermore, one can define an addi-
    tional LJ-interaction for a certain set of particles intended to
    represent the native contacts of a protein. Uses the BAOAB Lange-
    vin integrator.
    ###########
    Fields:
    ###########
        steps (int): Total number of steps to integrate
        stride (int): Interval to save position frames during sim
        nparticles (int): Number of particles to simulate
        dt (float): Time step [ps]
        temperature (float): Temperature for integrator [K]
        friction (float): Friction for integrator [1/ps]
        mass (float): Mass for each particle [u]
        charge (float): Charge of the particles [e]
        epsilon (float): Epsilon for Lennard-Jones interaction [kJ/mol]
        sigma (float): Sigma for Lennard-Jones interaction [nm]
        l_bond (float): Bond length for harmonic chain [nm]
        k_bond (float): Bond stiffnes for harmonic chain [kJ/mol / nm^2]
        epsilon_nat_contc (float): Sigma for Native Contacts interaction [kJ/mol]
        sigma_nat_contc (float): Epsilon for Native Contacts interaction [nm]
        native_contc (list[list[int; 2]]): contanct pairs to connect via the
            native contact interaction.
        use_bond_forces (bool): Add bond_forces to sim; default: True
        use_nonbond_forces (bool): Add nonbond_forces to sim; default: True
        use_native_contc_forces (bool): Add native_contc_forces to sim; default: True
        save_file (str): File to write the traj to; default '/tmp/traj.h5'
        platform (str): OpenMM platform to use; default: CPU
        """

    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)

    def __format__(self, __format_spec: str) -> str:
        """Format the config as a toml file such that atrributes
        are on multiple lines. Activate if __format_spec is 'fancy'
        """
        output = str(self)
        if __format_spec == "'fancy'" or __format_spec == "fancy":
            # find all spaces that are followed by an attribute of
            # the dataclass and replace them with a newline
            output = re.split("( [a-z|_]+=)", output)
            for i, item in enumerate(output):
                if item.startswith(" "):
                    output[i] = f"\n{item[1:]}"
            output = "".join(output)
            output = output.replace("=", " = ")
        return output
