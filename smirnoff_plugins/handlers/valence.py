from openff.toolkit import unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ConstraintHandler,
    IncompatibleParameterError,
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

    # The user specifies the Python environment to run Espaloma in
    espaloma_python_path = ParameterAttribute(default=None, converter=str)

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
        if self.espaloma_python_path != other_handler.espaloma_python_path:
            raise IncompatibleParameterError(
                "Attempted to initialize two EspalomaValence sections with different "
                "python paths: "
                f"{self.espaloma_python_path=} is not identical to {other_handler.espaloma_python_path=}"
            )


class OpenmmlValenceHandler(ParameterHandler):
    """ParameterHandler for applying valence parameters from the OpenmmML model."""

    _TAGNAME = "OpenmmlValence"
    _DEPENDENCIES = []
    _INFOTYPE = None  # No separate parameter types; just a model path

    model_name = ParameterAttribute(default="aceff-2.0", converter=str)
    model_path = ParameterAttribute(default=None, converter=str)

    def check_handler_compatibility(
        self,
        other_handler: "OpenmmlValenceHandler",
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
        if (self.model_name != other_handler.model_name) or (
            self.model_path != other_handler.model_path
        ):
            raise IncompatibleParameterError(
                "Attempted to initialize two OpenmmlValence sections with different "
                "models: "
                f"{self.model_name=} is not identical to {other_handler.model_name=} "
                "or "
                f"{self.model_path=} is not identical to {other_handler.model_path=}"
            )
