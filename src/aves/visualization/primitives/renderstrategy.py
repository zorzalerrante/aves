from abc import ABC, abstractmethod


class RenderStrategy(ABC):
    """An interface to render things"""

    def __init__(self, data):
        self.set_data(data)

    def set_data(self, data):
        self.data = data
        self.prepared = False

    def plot(self, ax, *args, **kwargs):
        if not self.prepared:
            self.prepare()

        self.render(ax, *args, **kwargs)

    def prepare(self):
        self.prepare_data()
        self.prepared = True

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def render(self, ax, *args, **kwargs):
        pass

    @abstractmethod
    def name(self):
        pass
