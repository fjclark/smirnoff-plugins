from openff.toolkit import unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ConstraintHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)


class UreyBradleyHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Urey-Bradley interactions."""

    class UreyBradleyType(ParameterType):
        """A custom SMIRNOFF type for Urey-Bradley interactions."""

        _ELEMENT_NAME = "UreyBradley"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        length = ParameterAttribute(default=None, unit=unit.nanometers)

    _TAGNAME = "UreyBradleys"
    _INFOTYPE = UreyBradleyType
    _DEPENDENCIES = [ConstraintHandler]


class EspalomaValenceHandler(ParameterHandler):
    """ParameterHandler for applying valence parameters from the Espaloma 0.3 model."""

    _TAGNAME = "EspalomaValence"
    _DEPENDENCIES = []
    _INFOTYPE = None  # No separate parameter types; just a model path
    # _MAX_SUPPORTED_SECTION_VERSION = Version("0.3")
    # model_file = ParameterAttribute(converter=str)

    def check_handler_compatibility(
        self,
        other_handler: "EspalomaValenceHandler",
        assume_missing_is_default: bool = True,
    ):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.
        Parameters
        ----------
        other_handler
            The handler to compare to.
        assume_missing_is_default
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        # TODO: exclude all other valence handlers
        # if self.model_file != other_handler.model_file:
        #     raise IncompatibleParameterError(
        #         "Attempted to initialize two NAGLCharges sections with different "
        #         "model_files: "
        #         f"{self.model_file=} is not identical to {}"
        #     )


class Egret1ValenceHandler(ParameterHandler):
    """ParameterHandler for applying valence parameters from the Egret-1 model."""

    _TAGNAME = "Egret1Valence"
    _DEPENDENCIES = []
    _INFOTYPE = None  # No separate parameter types; just a model path
    # _MAX_SUPPORTED_SECTION_VERSION = Version("0.3")
    # model_file = ParameterAttribute(converter=str)

    def check_handler_compatibility(
        self,
        other_handler: "Egret1ValenceHandler",
        assume_missing_is_default: bool = True,
    ):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.
        Parameters
        ----------
        other_handler
            The handler to compare to.
        assume_missing_is_default
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        # TODO: exclude all other valence handlers
        # if self.model_file != other_handler.model_file:
        #     raise IncompatibleParameterError(
        #         "Attempted to initialize two NAGLCharges sections with different "
        #         "model_files: "
        #         f"{self.model_file=} is not identical to {}"
        #     )
