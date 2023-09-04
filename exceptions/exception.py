class IncompatibleLayerError(Exception):
    def __init__(self, message="Layers are incompatible"):
        self.message = message
        super().__init__(self.message)


class DisconnectedLayersError(Exception):
    def __init__(self, message="No connection between layers") -> None:
        self.message = message
        super().__init__(self.message)


class ShapeMismatchError(Exception):
    def __init__(self, message="Shape mismatched") -> None:
        self.message = message
        super().__init__(self.message)
