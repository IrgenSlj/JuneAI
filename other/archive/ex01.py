import math

radius = float(input("Type the cycle radius: "))

perimeter = radius * 2 * math.pi
area = radius * math.pi

print(f"\nThe perimeter of the cycle is: {perimeter}")
print(f"The area of the cycle is: {area}")

print()
print(type(radius), type(area))
print(type(type(radius)))