class A:
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"{self.n}"

    def __del__(self):
        print(self.n + " destroyed")
        del self

def f():
    t = A("t")

f()

x = A("x")
z = x
del x
#print(z.__str__())