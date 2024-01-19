class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
