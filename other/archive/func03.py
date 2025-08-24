def is_odd(x):
    return x % 2 == 1
    
def is_even(x):
    return x % 2 == 0
    
def is_prime(x):
    if x == 0 or x == 1:
        return False
    for i in range(2, x - 1):
        if x % i == 0:
            return False
    return True
    
def is_square(x):
    i = 0
    sq = 0
    while sq < x:
        i +=1
        sq = i * i
    return sq == x

def is_cube(x):
    i = 0
    cu = 0
    while cu < x:
        i +=1
        cu = i ** 3
    return cu == x

for i in range(1, 101):
    print(i, " odd:  ", is_odd(i))
    print("    even:  ", is_even(i))
    print("   prime:  ", is_prime(i))
    print("  square:  ", is_square(i))
    print("    cube:  ", is_cube(i))
    print()