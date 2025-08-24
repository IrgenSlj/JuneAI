list_even = [n for n in range(0, 101, 2)]
set_even = set()
set_even.update(list_even)
list_odd = [n for n in range(1, 101, 2)]
set_odd = set()
set_odd.update(list_odd)

list_mult3 = [n for n in range(0, 101) if n % 3 == 0]
set_mult3 = set()
set_mult3.update(list_mult3)

primes = {2}
for i in range(2, 101):
    for j in range(2, i):
        if i % j == 0:
            break
        else:
            primes.add(i)

print(set_even)
print()
print(set_odd)
print()
print(set_mult3)
print()
print(primes)
print("-------------------")

even_mult3 = set_even | set_mult3
print(even_mult3)
print()

odd_prime = set_odd & primes
print(odd_prime)
print()

set3 = primes - set_odd
print(set3)
print()
