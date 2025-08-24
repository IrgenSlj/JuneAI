x = int(input("Enter first number: "))
y = int(input("Enter second number: "))

if x > y:
    print(f"{x} is bigger than {y}")
elif x < y:
    print(f"{y} is bigger than {x}")
else:
    print(f"{x} and {y} are equal")

print("-----------------")

a = int(input("Now enter first of three numbers: "))
b = int(input("Now enter second of three numbers: "))
c = int(input("Now enter third of three numbers: "))

if a > b and a > c:
    print(f"{a} is the biggest number")
elif b > a and b > c:
    print(f"{b} is the biggest number")
elif c > a and c > b:
    print(f"{c} is the biggest number")
else:
    print("All three numbers are equal")

print("-----------------")

x = int(input("Type x value: "))

max_value = x

y = int(input("Type y value: "))

if y > max_value:
    max_value = y

z = int(input("Type z value: "))

if z > max_value:
    max_value = z

print(f"The maximum value is {max_value}")
