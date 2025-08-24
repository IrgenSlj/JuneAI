my_list = []

for number in range(11):
    if number % 2 == 0:
        my_list.append(number**2)
    else:
        my_list.append(0)

print(my_list)

my_list = [number**2 if number % 2 == 0 else 0 for number in range(11)]

print(my_list)