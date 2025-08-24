class Cow:
    def __init__(self, weight, hunger=5):
        self.weight = weight
        self.__hunger = hunger

    def express(self):
        if self.__hunger > 5:
            print("Moooooooooooooooooooooooooo")
        else:
            print("Mooo")

molly = Cow(235, 6)
molly2 = Cow(160, 4)
molly2.express()
molly.express()