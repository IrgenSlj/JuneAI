from math import sqrt

class Cyrcle:
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return f"c^2 = x^2 + y^2"
    
    def __eq__(self, other):
        return self.c == other.c
    
    def __lt__(self, other):
        return self.c < other.c
    
    def __call__(self, x, y=None):
        if isinstance(x, (int, float)):
            if abs(x) < self.c:
                return (x, sqrt(self.c**2 - x**2)), (x,-sqrt(self.c**2 - x**2))
            
            elif abs(x) == self.c:
                return (x, 0)
            
            else:
                return None