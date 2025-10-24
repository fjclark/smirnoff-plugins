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
    Egret1ValenceHandler,
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


class Egret1ValenceCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["Egret1Valence"] = "Egret1Valence"

    expression: Literal[""] = ""

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (Egret1ValenceHandler,)

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

    def store_potentials(self, parameter_handler: UreyBradleyHandler) -> None:
        """Store the potentials from the parameter handler."""
        pass

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        import copy

        def get_egret_1() -> "MLPotential":
            """Get the Egret-1 MLPotential from GitHub."""
            from openmmml import MLPotential
            import atexit
            import urllib.request
            import os
            import tempfile

            # Model accessed 24/05/25
            url = "https://github.com/rowansci/egret-public/raw/227d6641e6851eb1037d48712462e4ce61c1518f/compiled_models/EGRET_1.model"
            tmp_file = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
            tmp_file.close()  # Close so urllib can write to it
            urllib.request.urlretrieve(url, filename=tmp_file.name)

            # Register file for deletion at program exit
            atexit.register(
                lambda: (
                    os.remove(tmp_file.name) if os.path.exists(tmp_file.name) else None
                )
            )

            return MLPotential("mace", modelPath=tmp_file.name)

        assert len(interchange.topology._molecules) == 1

        mlp = get_egret_1()
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

    def store_potentials(self, parameter_handler: UreyBradleyHandler) -> None:
        """Store the potentials from the parameter handler."""
        pass

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

        # Rip out valence forces and replace with Espaloma,

        # Note that we have to run espaloma part in another environment
        # due to requirements conflicts (pydantic...)
        import copy
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as script_file:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".xml"
            ) as esp_omm_system_file:
                # with NamedTemporaryFile(
                #     delete=False, suffix=".
                script = f"""
import espaloma as esp
from openff.toolkit import Molecule
from openmm import XmlSerializer

mol = Molecule.from_mapped_smiles({interchange.topology.molecule(0).to_smiles(mapped=True)!r})
mol_graph = esp.Graph(mol)
model = esp.get_model("latest")
model(mol_graph.heterograph)
esp_system = esp.graphs.deploy.openmm_system_from_graph(
mol_graph,
forcefield="openff_unconstrained-2.2.1",  # Could be any OpenFF force field - non-bonded parameters are replaced
charge_method="nn",
)
# Save the OpenMM system to a file
with open("{esp_omm_system_file.name}", "w") as f:
    xml_data = XmlSerializer.serialize(esp_system)
    f.write(xml_data)
"""
                with open(script_file.name, "w") as f:
                    f.write(script)

                ESP_ENV_PYTHON = (
                    "/home/campus.ncl.ac.uk/nfc78/miniforge3/envs/espaloma/bin/python"
                )

                import subprocess

                subprocess.run(
                    [ESP_ENV_PYTHON, script_file.name],
                    check=True,
                )

                # Load in the OpenMM system from the file
                from openmm import XmlSerializer

                with open(esp_omm_system_file.name, "r") as f:
                    esp_system = XmlSerializer.deserialize(f.read())

        # Take the forces from the Espaloma system, remove the equivalent forces from "system",
        # and add the Espaloma forces to the OpenMM system.
        force_types_to_replace = [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
        ]
        for force in system.getForces():
            if force.getName() in force_types_to_replace:
                system.removeForce(force)

        for force in esp_system.getForces():
            if force.getName() in force_types_to_replace:
                system.addForce(copy.deepcopy(force))

    # def modify_openmm_forces(
    #     self,
    #     interchange: Interchange,
    #     system: openmm.System,
    #     add_constrained_forces: bool,
    #     constrained_pairs: Set[Tuple[int, ...]],
    #     particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    # ) -> None:
    #     # Mainly taken from
    #     # https://github.com/openforcefield/openff-interchange/blob/83383b8b3af557c167e4a3003495e0e5ffbeff73/openff/interchange/interop/openmm/_valence.py#L50

    #     # Rip out valence forces and replace with Espaloma,

    #     # Note that we have to run espaloma part in another environment
    #     # due to requirements conflicts (pydantic...)

    #     import copy

    #     import espaloma as esp

    #     assert len(interchange.topology._molecules) == 1
    #     mol_graph = esp.Graph(interchange.topology.molecule(0))
    #     model = esp.get_model("latest")
    #     model(mol_graph.heterograph)
    #     esp_system = esp.graphs.deploy.openmm_system_from_graph(
    #         mol_graph,
    #         forcefield="openff_unconstrained-2.2.1",  # Could be any OpenFF force field - non-bonded parameters are replaced
    #         charge_method="from-molecule",
    #     )

    #     # Take the forces from the Espaloma system, remove the equivalent forces from "system",
    #     # and add the Espaloma forces to the OpenMM system.
    #     force_types_to_replace = [
    #         "HarmonicBondForce",
    #         "HarmonicAngleForce",
    #         "PeriodicTorsionForce",
    #     ]
    #     for force in system.getForces():
    #         if force.getName() in force_types_to_replace:
    #             system.removeForce(force)

    #     for force in esp_system.getForces():
    #         if force.getName() in force_types_to_replace:
    #             system.addForce(copy.deepcopy(force))
