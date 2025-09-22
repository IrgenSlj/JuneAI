class Length:
    def __init__(self, val, m):
        if m == "m":
            self.val = val
        elif m == "cm":
            self.val = val / 100
        elif m == "in":
            self.val = val * 0.0254
        else:
            self.val = 0

    def __str__(self):
        return str(self.val) +"m"
    
    def __add__(self, other):
        return Length(self.val + other.val, "m")
    
    def __round__(self, n=None):
        return round(self.val, n)
    
    def __int__(self):
        return int(self.val)

    
l1 = Length(1, "m")
l2 = Length(500, "cm")
l3 = Length(30, "in")
ad = (l1 + l2 + l3).__round__(3)
print(ad)

print(ad.__int__().__str__() + " m")
print(l3.__round__(2))