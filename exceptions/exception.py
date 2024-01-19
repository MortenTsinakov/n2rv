"""Custom exceptions."""


class IncompatibleLayerError(Exception):
    """Exception thrown when layers are not compatible with each other."""
    def __init__(self, message="Layers are incompatible"):
        self.message = message
        super().__init__(self.message)


class DisconnectedLayersError(Exception):
    """Exception thrown when layers in a Model layer graph cannot be connected."""
    def __init__(self, message="No connection between layers") -> None:
        self.message = message
        super().__init__(self.message)


class ShapeMismatchError(Exception):
    """Exception thrown when shapes of layers don't match."""
    def __init__(self, message="Shape mismatched") -> None:
        self.message = message
        super().__init__(self.message)
