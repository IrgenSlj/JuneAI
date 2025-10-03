from abc import ABC, abstractmethod
from math import pi

class GeometricObjectInterface(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Circle(GeometricObjectInterface):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * pi * self.radius
    
class Resizable(ABC):
    def resize(self, param):
        pass

class CircleResizable(Circle, Resizable):
    def __init__(self, radius):
        Circle.__init__(self, radius)

    def resize(self, param):
        self.radius = self.radius * param

c = CircleResizable(1)
print(c.area(), c.perimeter())
c.resize(2)
print(c.area(), c.perimeter())