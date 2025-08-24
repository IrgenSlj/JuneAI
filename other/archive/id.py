import copy

'''
x = 5
y = 5
y += 1
print(id(x), id(5), id(y), id(6))
'''

l1 = [1, 2]
l2 = [1, 2]

print(id(l1))
print(id(l2))
print()

print(id(l1[1]))
print(id(l2[1]))
print()

t = [1, [1, 2]]
t[1].append("loly")
t2 = t.copy()
print(id(t2))
print()

t2 = copy.deepcopy(t)
print(id(t2))