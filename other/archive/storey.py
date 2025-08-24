class Storey:
    def __init__(self):
        self.people = 0

class Building:
    def __init__(self, num_storeys):
        self.storeys = [Storey() for _ in range(num_storeys)]

    def add_people(self, storey, people):
        self.storeys[storey].people += people

    def print_people(self):
        total = 0
        for i in range(len(self.storeys)):
            print(f"Storey {i} has {self.storeys[i].people} people")
            total += self.storeys[i].people
        print(f"total amount of people is: {total}")

e = Building(5)
e.add_people(0, 3)
e.add_people(1, 2)
e.add_people(4, 5)

e.print_people()