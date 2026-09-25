"""Teach Interchange about the ``NAGLMBISCharges`` handler.

Interchange hard codes which handlers can assign partial charges, so this module adds
``NAGLMBISCharges`` to ``_create._SUPPORTED_PARAMETER_HANDLERS`` and wraps ``_create._electrostatics``
to build :class:`~smirnoff_plugins.collections.charges.SMIRNOFFNAGLMBISElectrostaticsCollection` when a
force field has a ``NAGLMBISCharges`` section. ``_create`` is only partly initialised when plugins are
loaded, so it is patched lazily on the first call to ``Interchange.from_smirnoff``.
"""

import functools

_TAGNAME = "NAGLMBISCharges"


@functools.cache
def _patch_create():
    """Patch ``openff.interchange.smirnoff._create`` to support ``NAGLMBISCharges``."""
    from openff.interchange.exceptions import MissingParameterHandlerError
    from openff.interchange.smirnoff import _create

    from smirnoff_plugins.collections.charges import SMIRNOFFNAGLMBISElectrostaticsCollection

    original_electrostatics = _create._electrostatics

    @functools.wraps(original_electrostatics)
    def _electrostatics(
        interchange,
        force_field,
        topology,
        molecules_with_preset_charges=None,
        allow_nonintegral_charges: bool = False,
    ):
        if _TAGNAME not in force_field.registered_parameter_handlers:
            return original_electrostatics(
                interchange,
                force_field,
                topology,
                molecules_with_preset_charges,
                allow_nonintegral_charges,
            )

        if "Electrostatics" not in force_field.registered_parameter_handlers:
            raise MissingParameterHandlerError(
                f"Force field contains a {_TAGNAME} section, which assigns partial charges, "
                "but no ElectrostaticsHandler was found.",
            )

        handlers = force_field._parameter_handlers

        interchange.collections["Electrostatics"] = SMIRNOFFNAGLMBISElectrostaticsCollection.create(
            parameter_handler=[
                handlers[name]
                for name in [
                    "Electrostatics",
                    _TAGNAME,
                    "NAGLCharges",
                    "ChargeIncrementModel",
                    "ToolkitAM1BCC",
                    "LibraryCharges",
                ]
                if name in handlers
            ],
            topology=topology,
            molecules_with_preset_charges=molecules_with_preset_charges,
            allow_nonintegral_charges=allow_nonintegral_charges,
        )

    _create._SUPPORTED_PARAMETER_HANDLERS.add(_TAGNAME)
    _create._electrostatics = _electrostatics


def install():
    """Wrap ``Interchange.from_smirnoff`` so that ``_create`` is patched before it is first used."""
    from openff.interchange import Interchange

    original_from_smirnoff = Interchange.from_smirnoff.__func__

    @functools.wraps(original_from_smirnoff)
    def from_smirnoff(cls, *args, **kwargs):
        _patch_create()
        return original_from_smirnoff(cls, *args, **kwargs)

    Interchange.from_smirnoff = classmethod(from_smirnoff)
