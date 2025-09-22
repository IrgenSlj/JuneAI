class Byte:
    def __init__(self, s=""):
        if s == "":
            self.array = [0 for i in range(8)]
        else:
            self.array = [int(c) for c in s]

    def __str__(self):
        st = [str(c) for c in self.array]
        return "".join(st)
    
    def __lshift__(self, other):
        for i in range(other):
            self.array.pop(0)
            self.array.append(0)

    def __rshift__(self, other):
        for i in range(other):
            self.array.pop()
            self.array.insert(0, 0)

    def __and__(self, other):
        new_byte = Byte("")
        for i in range(8):
            new_byte.array[i] = self.array[i] & other.array[i]
        return new_byte
    
    def __or__(self, other):
        new_byte = Byte("")
        for i in range(8):
            new_byte.array[i] = self.array[i] | other.array[i]
        return new_byte
    
    def __invert__(self):
        new_byte = Byte("")
        for i in range(8):
            new_byte.array[i] = 1 if self.array[i] == 0 else 0
        return new_byte

b = Byte("00110101")
b1 = Byte("00000000")
print(~b)


'''b = Byte("00110101")
b2 = Byte()
print(b2, b)
print()
print(b.__str__())
print()
b << 2
print(b)
b >> 2
print(b)
print()

b3 = Byte("01001010")
b4 = Byte("01000000")
x = b3|b4
print(x)'''