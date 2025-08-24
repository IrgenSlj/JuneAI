def factory_power(power):
    def nth_power(number):
        return number ** power
    return nth_power

square = factory_power(2)
print(square(3))


cube = factory_power(2)
print(cube(4))