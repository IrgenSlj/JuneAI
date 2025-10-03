class Base1:
    def __init__(self, b1_attr):
        self.b1 = b1_attr

class Base2:
    def __init__(self, b2_attr):
        self.b2 = b2_attr

class Base(Base1,Base2):
    def __init__(self, b1, b2, b):
        Base1.__init__(self, b1)
        Base2.__init__(self, b2)
        self.b = b

d = Base(1, 2, 3)
print(d.b1)
