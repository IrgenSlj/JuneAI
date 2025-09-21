from math import sqrt
from point import Point

class Line:
    def __init__(self, point_A=None, point_B=None):
        if point_A == None:
            self.point_A = Point(0, 0)
        else:
            self.point_A = point_A

        if point_B is None:
            self.point_B = Point(0, 0)
        else:
            self.point_B = point_B

    def set_point_A(self, point_A):
        self.point_A = (point_A)

    def set_point_B(self, point_B):
        self.point_B = point_B

    def length(self):
        return sqrt((self.point_A.x - self.point_B.x)**2 + (self.point_A.y - self.point_B.y)**2)

    def __str__(self):
        return f"{self.point_A}-{self.point_B}"
    
    def __lt__(self, other):
        if isinstance(other, int):
            return self.length() < other
        elif isinstance(other, Line):
            return self.length() < other.length()
    
    def __eq__(self, other):
        if isinstance(other, int):
            return self.length() == other
        elif isinstance(other, Line):
            return self.length() == other.length()

l1 = Line(Point(1, 1))
l2 = Line(Point(1, 1))

print(l1.__eq__(l2))
print(l1.__lt__(l2))