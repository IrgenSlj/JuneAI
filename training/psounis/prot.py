class Base:
    def __init__(self):
        self.__a = 1

class Derived(Base):
    def __init__(self):
        super().__init__()

d = Derived()
print(d._Base__a)