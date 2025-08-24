from random import randrange

array = []

for row in range(3):
    new_row = []
    for item in range(3):
        new_row.append(randrange(0, 1000))
    array.append(new_row)

for i in range(len(array)):
    print(array[i])

for row in array:
    for i in range(3):
        print("+----", end="")
    print("+")

    for element in row:
        print(("|" + str(element) + "\t").expandtabs(5), end="")
    print("|")

for i in range(3):
    print("+----", end="")
print("+")
