class Account:
    def __init__(self, id, name, amount=0):
        self.id = id
        self.name = name
        self.amount = amount

    def transfer_to(self, account, amount=0):
        if amount <= self.amount:
            self.amount -= amount
            account.amount += amount
            print(
                f"{self.name} transfered {amount} Euto to {account.name}. \nRemaining balance: {self.amount} Euro")
        else:
            print("Transfered failed! Not enough credit")

    def print_balance(self):
        print(f"\nWelcome {self.name}. Your balance is {self.amount} Euro")

class Person:
    def __init__(self, name, age, id):
        self.name = name
        self.age = age
        self.id = id


person_1 = Person("Maria Gonzalez", 34, "345354343435")
person_2 = Person("John Blade", 45, "345354343434")

client_1 = Account(person_1.id, person_1.name, 1590920.23)
client_2 = Account(person_2.id, person_2.name, 55448392.43)

client_1.transfer_to(client_2, 1000)
client_2.print_balance()