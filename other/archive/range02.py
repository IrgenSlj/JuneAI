inpt = int(input("Type amount of number (3 - 20): "))

while inpt < 3 or inpt > 20:
    inpt = int(input("Type number 3 - 20: "))

numbers = []

for i in range(1, inpt+1):
    numbers.append(float(input(f"Type number in position {i}: ")))

print(f"\nNumbers entered are: {numbers}")

numbers.sort()

print(f"Sorted you get these numbers: {numbers}")