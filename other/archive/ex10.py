class MyClass:
    def __init__(self, x):
        self.x = x

o1 = MyClass(2)
o2 = MyClass(2)

o3 = o2
print(o2 is o3)
print(id(o1), id(o2), id(o3))

print("_-----_")

print(isinstance([1, 2], (list, int)))