import functools

import numpy
from openff.interchange.components.potentials import Potential
from openff.interchange.models import (
    PotentialKey,
    SingleAtomChargeTopologyKey,
    TopologyKey,
)
from openff.interchange.smirnoff._nonbonded import SMIRNOFFElectrostaticsCollection
from openff.toolkit import Molecule, Quantity
from openff.toolkit.utils.exceptions import MissingPackageError

from smirnoff_plugins.handlers.charges import NAGLMBISChargesHandler


# Only a gas/water pair of models is normally used, so keep the (large) models cache small
@functools.lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Load a pre-trained ``naglmbis`` charge model."""
    from naglmbis.models import load_charge_model

    return load_charge_model(charge_model=model_name)


@functools.lru_cache(maxsize=1024)
def _compute_nagl_mbis_charges(
    mapped_smiles: str,
    gas_model: str,
    water_model: str,
    alpha: float,
) -> numpy.ndarray:
    """Compute partially polarised NAGL-MBIS charges (in units of e), normalised to the formal charge."""
    try:
        from naglmbis.models import ComputePartialPolarised
    except ImportError as error:
        raise MissingPackageError(
            "The force field has a NAGLMBISCharges section, but naglmbis is not installed. "
            "Use the `naglmbis` pixi environment, e.g. `pixi run -e naglmbis ...`.",
        ) from error

    molecule = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)

    polarised_model = ComputePartialPolarised(
        model_gas=_load_model(gas_model),
        model_water=_load_model(water_model),
        alpha=alpha,
    )
    charges = polarised_model.compute_polarised_charges(molecule.to_rdkit()).detach().numpy().reshape(-1)

    # The charges should already sum to the formal charge, but spread any small residual
    # evenly over the atoms, as the OpenFF toolkit does for NAGL charges:
    # https://github.com/openforcefield/openff-toolkit/blob/f4566b8694d7d744a70800d8f165ab05955fc334/openff/toolkit/utils/nagl_wrapper.py#L162
    return charges + (molecule.total_charge.m - charges.sum()) / molecule.n_atoms


class SMIRNOFFNAGLMBISElectrostaticsCollection(SMIRNOFFElectrostaticsCollection):
    """
    The standard SMIRNOFF electrostatics collection, extended to assign partial charges from
    a ``NAGLMBISCharges`` section.

    Library charges take precedence over NAGL-MBIS charges, which take precedence over the
    other charge methods. The resulting potentials are labelled as coming from the
    ``NAGLChargesHandler`` so that the rest of Interchange (charge lookup, serialization,
    combining, logging) treats them like any other NAGL charges. The NAGL-MBIS provenance (models
    and alpha) is kept in ``extras["partial_charge_method"]`` of each ``SingleAtomChargeTopologyKey``,
    which is serialized with the key map.
    """

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [*super().allowed_parameter_handlers(), NAGLMBISChargesHandler]

    @classmethod
    def _find_reference_matches(
        cls,
        parameter_handlers,
        unique_molecule: Molecule,
    ) -> tuple[dict[TopologyKey, PotentialKey], dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a particular reference molecule and set of parameter handlers.
        """
        if "NAGLMBISCharges" not in parameter_handlers:
            return super()._find_reference_matches(parameter_handlers, unique_molecule)

        if "LibraryCharges" in parameter_handlers:
            library_matches, library_potentials = cls._find_slot_matches(
                parameter_handlers["LibraryCharges"],
                unique_molecule,
            )

            if {index for key in library_matches for index in key.atom_indices} == set(range(unique_molecule.n_atoms)):
                return library_matches, library_potentials

        parameter_handler = parameter_handlers["NAGLMBISCharges"]
        # Drop any stale atom map (e.g. from `from_smiles` on a mapped SMILES, with string keys after a
        # JSON round trip), which would otherwise make `to_smiles` silently emit an unmapped SMILES.
        # This is a temporary workaround until this is fixed in the OpenFF toolkit.
        molecule = Molecule(unique_molecule)
        molecule._properties.pop("atom_map", None)
        mapped_smiles = molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)

        partial_charge_method = (
            f"NAGL-MBIS (gas_model={parameter_handler.gas_model}, "
            f"water_model={parameter_handler.water_model}, alpha={parameter_handler.alpha})"
        )

        partial_charges = _compute_nagl_mbis_charges(
            mapped_smiles,
            parameter_handler.gas_model,
            parameter_handler.water_model,
            parameter_handler.alpha,
        )

        matches: dict = {}
        potentials: dict[PotentialKey, Potential] = {}

        for atom_index, partial_charge in enumerate(partial_charges):
            # Label the charges as NAGLCharges, as Interchange only accepts known
            # handler names when looking up charges and deserializing
            potential_key = PotentialKey(
                id=mapped_smiles,
                mult=atom_index,
                associated_handler="NAGLChargesHandler",
            )
            potentials[potential_key] = Potential(
                parameters={"charge": Quantity(float(partial_charge), "elementary_charge")},
            )

            matches[
                SingleAtomChargeTopologyKey(
                    this_atom_index=atom_index,
                    extras={
                        "handler": "NAGLChargesHandler",
                        "partial_charge_method": partial_charge_method,
                    },
                )
            ] = potential_key

        return matches, potentials
