prime_list = []

for i in range(2, 101):
    for j in range(2, i - 1):
        if i % j == 0:
            break
    else:
        prime_list.append(i)

prime_tuple = tuple(prime_list)
print(prime_tuple)