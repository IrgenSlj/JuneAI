my_set = {1, 2, 3}
set2 = my_set.copy()
set2.add(4)
print(set2)

set2.remove(4)
set2.discard(44)
print(set2)

set2.clear()
print(set2)