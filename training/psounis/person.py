from random import randrange
from abc import ABC

class Person(ABC):
    def __init__(self, name, wage):
        self.name = name
        self.wage = wage

class Waiter(Person):
    def __init__(self, name, wage):
        super().__init__(name, wage)
        pass

    def serve(self, customers, barista):
        print(f"\nWaiter {self.name} brought {customers} customers to barista {barista.name}")
        barista.prepare(customers)

class Barista(Person):
    def __init__(self, name, wage):
        Person.__init__(self, name, wage)
        pass

    def prepare(self, customers):
        print(f"Barista {self.name} prepared {customers} customers.")

class Owner(Waiter, Barista):
    def __init__(self, name, wage):
        Barista.__init__(self, name, wage)
        pass

def main():
    owner = Owner("Mike", 5000)
    w1 = Waiter("Kim", 2000)
    w2 = Waiter("Stef", 2300)
    barista1 = Barista("Panos", 2500)
    barista2 = Barista("Nikos", 2400)
    waiters = [w1, w2]
    baristas = [barista1, barista2]

    for i in range(10):
        people = randrange(1, 5)
        serving_waiter = waiters[randrange(len(waiters))]
        serving_barista = baristas[randrange(len(baristas))]
        serving_waiter.serve(people, serving_barista)


main()