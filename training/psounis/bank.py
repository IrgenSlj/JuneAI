from queue import Queue
from random import randrange

class Bank:
    def __init__(self, N):
        self.N = N
        self.cash_desks = [Queue() for i in range(N)]

    def customer_enters(self, full_name):
        cash_desk = randrange(self.N)
        self.cash_desks[cash_desk] += full_name
        print(f"{full_name} to be served by cash desk {cash_desk}")


    def customer_serverd(self):
        not_empty_cash_desks = [i for i in range(self.N) if len(self.cash_desks[i]) > 0]

        if len(not_empty_cash_desks) > 0:
            cash_desk = not_empty_cash_desks[randrange(len(not_empty_cash_desks))]
            customer = - self.cash_desks[cash_desk]
            print(f"{customer} served by cash desk {cash_desk}")
        else:
            print("No customers")
    
    def __str__(self):
        return ", ".join(self.n)