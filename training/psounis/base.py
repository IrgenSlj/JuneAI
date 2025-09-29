class Base:
    def __init__(self, b):
        self.b = b
    
    def __str__(self):
        return "SOme info about the base class"
    
class Derived(Base):
    def __init__(self, b, d):
        super().__init__(b)
        self.d = d

    def __str__(self):
        return "Some info  about the derived class"
    
    def info(self):
        return super().__str__() + " and " + self.__str__()
    
d = Derived(4, 5)
print(d.info())