class Flat:
    def __init__(self):
        self.people = 0

class Storey:
    def __init__(self, number_flats):
        self.flats = [Flat() for _ in range(number_flats)]

class Building:
    def __init__(self, num_storeys, number_flats):
        self.storeys = [Storey(number_flats) for _ in range(num_storeys)]

    def add_people(self, storey, flat, people):
        self.storeys[storey].flats[flat].people += people

    def print_people(self):
        total = 0
        for i in range(len(self.storeys)):
            for y in range(len(self.storeys[i].flats)):
                print(f"Storey {i}, flat {y} has {self.storeys[i].flats[y].people} people")
                total += self.storeys[i].flats[y].people
        print(f"total amount of people is: {total}")

e = Building(5, 5)
e.add_people(1, 3, 4)
e.add_people(2, 1, 3)
e.print_people()