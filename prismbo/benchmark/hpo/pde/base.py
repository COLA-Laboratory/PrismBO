from abc import ABC, abstractmethod

class PDE(ABC):
    def __init__(self):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def pde(self, x, y):
        pass
    
    @abstractmethod
    def bc(self, x, y):
        pass
    
    @abstractmethod
    def ic(self, x, y):
        pass

    @abstractmethod
    def analytic_func(self, x):
        pass