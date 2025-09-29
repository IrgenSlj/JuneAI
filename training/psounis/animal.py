class Animal:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height

class Horse(Animal):
    def __init__(self, weight, height, color, tail):
        super().__init__(weight, height)
        self.color = color
        self.tail = tail

class Dog(Animal):
    def __init__(self, weight, height, bark_db):
        super().__init__(weight, height)
        self.bark_db = bark_db

    def bark(self):
        print("ggab" + str(self.bark_db))

class Doberman(Dog):
    def __init__(self, weight, height, bark_db):
        super().__init__(weight, height, bark_db)

    def run(self):
        print("Doberman runs")

class Bulldog(Dog):
    def __init__(self, weight, height, bark_db, ears_size):
        super().__init__(weight, height, bark_db)
        self.ears_size = ears_size

    def sleep(self):
        print("Bulldog sleeps")

horse = Horse(250, 1.60, "black", "long")
print(f"Horse color is {horse.color}")
print()

dober = Doberman(70, 0.5, -10)
dober.bark()
dober.run()
print()

bull = Bulldog(67, 0.4, -15, 0.5)
bull.bark()
bull.sleep()
bull.sleep()