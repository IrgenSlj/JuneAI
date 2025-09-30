class Customer:
    def __init__(self, name, surname, address, orders=[]):
        self.name = name
        self.surname = surname
        self.address = address
        self.orders = orders

    def place_order(self, order):
        return  self.orders.append(order)
    
    def __str__(self):
        st = f"{self.name}, {self.surname}, {self.address}\n"
        st += "\nORDERS:\n"
        total = 0
        for order in self.orders:
            st += f"\nOrder date: {order.date}, sum {str(order.payment.sum)} Euro"
            total += order.payment.sum

        st += f"\n===============\nTotal payments: {total} Euro"

        print(st)
    
class Payment:
    def __init__(self, sum):
        self.sum = sum

class Order:
    def __init__(self, date, payment):
        self.date = date
        self.payment = payment

def main():
    customer = Customer("Mike", "Gibson", "California Str")
    customer.place_order(Order("20251007", Payment(5.6)))
    customer.place_order(Order("20251010", Payment(12.3)))
    customer.place_order(Order("20251027", Payment(23.0)))

    customer.__str__()

main()