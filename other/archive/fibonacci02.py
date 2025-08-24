def fib02(n):
    fib = [0, 1]

    for i in range(2, n + 1):
        fib.append(fib[i -1] + fib[i - 2])
    return fib[n]

for i in range(101):
    print(fib02(i))
