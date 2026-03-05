from functools import lru_cache
from typing import Dict, Iterable, Literal, Set, Tuple, Type, Union

from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.interop.openmm._valence import _is_constrained
from openff.interchange.models import PotentialKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.toolkit import Quantity
from openff.toolkit import unit as off_unit
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import openmm


from smirnoff_plugins.handlers.valence import (
    EspalomaValenceHandler,
    UreyBradleyHandler,
    OpenmmmlValenceHandler,
)


@lru_cache
def _cache_urey_bradley_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    parameter = parameter_handler.parameters[potential_key.id]

    return {
        parameter_name: getattr(parameter, parameter_name)
        for parameter_name in ["k", "length"]
    }


class SMIRNOFFUreyBradleyCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["UreyBradleys"] = "UreyBradleys"

    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (UreyBradleyHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "k", "length"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "k", "length"

    @classmethod
    def valence_terms(cls, topology):
        """Return all angles in this topology."""
        return [(angle[0], angle[2]) for angle in topology.angles]

    def store_potentials(self, parameter_handler: UreyBradleyHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            self.potentials.update(
                {
                    potential_key: Potential(
                        parameters=_cache_urey_bradley_parameter_lookup(
                            potential_key,
                            parameter_handler,
                        ),
                    ),
                },
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        # Mainly taken from
        # https://github.com/openforcefield/openff-interchange/blob/83383b8b3af557c167e4a3003495e0e5ffbeff73/openff/interchange/interop/openmm/_valence.py#L50

        harmonic_bond_force = openmm.HarmonicBondForce()
        harmonic_bond_force.setName("UreyBradleyForce")
        system.addForce(harmonic_bond_force)

        has_constraint_handler = "Constraints" in interchange.collections

        for top_key, pot_key in self.key_map.items():
            openff_indices = top_key.atom_indices
            openmm_indices = tuple(particle_map[index] for index in openff_indices)

            if len(openmm_indices) != 2:
                raise ValueError(
                    f"Expected 2 indices for Urey-Bradley potential, got {len(openmm_indices)}: {openmm_indices}",
                )

            if has_constraint_handler and not add_constrained_forces:
                if _is_constrained(
                    constrained_pairs,
                    (openmm_indices[0], openmm_indices[1]),
                ):
                    # This 1-3 length is constrained, so not add a bond force
                    continue

            params = self.potentials[pot_key].parameters
            k = params["k"].m_as(
                off_unit.kilojoule / off_unit.nanometer**2 / off_unit.mol,
            )
            length = params["length"].m_as(off_unit.nanometer)

            harmonic_bond_force.addBond(
                particle1=openmm_indices[0],
                particle2=openmm_indices[1],
                length=length,
                k=k,
            )


class OpenmmmlValenceCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["OpenmmmlValence"] = "OpenmmmlValence"

    expression: Literal[""] = ""

    model_name: str = "aceff-2.0"
    model_path: Union[str, None] = None

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (OpenmmmlValenceHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return ()

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return ()

    # @classmethod
    # def valence_terms(cls, topology):
    #     """Return all angles in this topology."""
    #     return [(angle[0], angle[2]) for angle in topology.angles]

    def store_potentials(self, parameter_handler: OpenmmmlValenceHandler) -> None:
        """Store the potentials from the parameter handler."""
        self.model_name = parameter_handler.model_name
        self.model_path = parameter_handler.model_path

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        import copy
        from openmmml import MLPotential

        assert len(interchange.topology._molecules) == 1

        mlp = MLPotential(self.model_name, modelPath=self.model_path)

        new_system = mlp.createSystem(
            interchange.topology.to_openmm(),
        )

        while system.getNumForces() > 0:
            system.removeForce(0)

        for force in new_system.getForces():
            system.addForce(copy.deepcopy(force))


class EspalomaValenceCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["EspalomaValence"] = "EspalomaValence"

    expression: Literal[""] = ""

    espaloma_python_path: Union[str, None] = None

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (EspalomaValenceHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return ()

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return ()

    # @classmethod
    # def valence_terms(cls, topology):
    #     """Return all angles in this topology."""
    #     return [(angle[0], angle[2]) for angle in topology.angles]

    def store_potentials(self, parameter_handler: EspalomaValenceHandler) -> None:
        """Store the potentials from the parameter handler."""
        self.espaloma_python_path = parameter_handler.espaloma_python_path

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        # Rip out valence forces and replace with Espaloma.

        # Note that we have to run espaloma in another environment
        # due to requirements conflicts (pydantic...).
        import copy
        import fcntl
        import hashlib
        import os
        import subprocess
        import tempfile

        from openmm import XmlSerializer

        if self.espaloma_python_path is None:
            raise ValueError(
                "espaloma_python_path must be specified for EspalomaValenceCollection/EspalomaValenceHandler"
            )

        smiles = interchange.topology.molecule(0).to_smiles(mapped=True)
        smiles_hash = hashlib.sha256(smiles.encode()).hexdigest()

        cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(cache_base, "smirnoff_plugins", "espaloma")
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, f"{smiles_hash}.xml")
        lock_path = cache_path + ".lock"

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if not os.path.exists(cache_path):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".py"
                    ) as script_file:
                        tmp_xml_path = cache_path + ".tmp"
                        script = f"""
import warnings
warnings.filterwarnings("ignore", message="Recommend creating graphs")
warnings.filterwarnings("ignore", module="dgl")
import espaloma as esp
from openff.toolkit import Molecule
from openmm import XmlSerializer

mol = Molecule.from_mapped_smiles({smiles!r}, allow_undefined_stereo=True)
mol_graph = esp.Graph(mol)
model = esp.get_model("latest")
model(mol_graph.heterograph)
esp_system = esp.graphs.deploy.openmm_system_from_graph(
mol_graph,
forcefield="openff_unconstrained-2.2.1",  # The default
charge_method="nn",
)
with open("{tmp_xml_path}", "w") as f:
    f.write(XmlSerializer.serialize(esp_system))
"""
                        with open(script_file.name, "w") as f:
                            f.write(script)

                        try:
                            subprocess.run(
                                [self.espaloma_python_path, script_file.name],
                                check=True,
                            )
                            # Atomic rename so readers never see a partial file
                            os.rename(tmp_xml_path, cache_path)
                        finally:
                            if os.path.exists(script_file.name):
                                os.remove(script_file.name)
                            if os.path.exists(tmp_xml_path):
                                os.remove(tmp_xml_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        with open(cache_path, "r") as f:
            esp_system = XmlSerializer.deserialize(f.read())

        # Replace all of the forces in the system with those from Espaloma,
        # including non-bonded forces
        while system.getNumForces() > 0:
            system.removeForce(0)

        for force in esp_system.getForces():
            system.addForce(copy.deepcopy(force))
