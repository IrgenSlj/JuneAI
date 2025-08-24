N = 9
n = 1
for i in range(1, N + 1):
    print(" " * (N - i) + "*" * (i * 2 - 1))


for i in range(0, N):
    for j in range(0, N - i -1):
        print(" ", end="")
    for j in range(0, 2 * i + 1):
        print("*", end="")
    print("")

