subsets = set()
N = 10

for i in range(1, N + 1):
    for j in range( + 1, N + 1):
        subsets.add((i,j))

for elements in subsets:
    print(elements)