class Cow:
    def __init__(self, weight, hunger):
        self.weight = weight
        self.hunger = hunger

    def express(self):
        if self.hunger > 5:
            print("wwwoooooooowww")
        else:
            print("wow")

class TexasLonghorn(Cow):
    def __init__(self, weight, hunger, horn_length):
        super().__init__(weight, hunger)
        self.horn_length = horn_length

molly = Cow(500, 4)
molly.express()
print()

bob = TexasLonghorn(400, 20, 0.50)
bob.express()
print()

print(f"Molly {molly.weight}, {molly.hunger}")
print()

print(f"Bob {bob.horn_length}")