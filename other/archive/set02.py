N = 5
l = set()
t = set()
for i in range(1, N + 1):
    t = (i, i**2)
    l.add(t)
print(l)
for element in l:
    print(element)