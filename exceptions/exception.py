class IncompatibleLayerError(Exception):
    def __init__(self, message="Layers are incompatible"):
        self.message = message
        super().__init__(self.message)
