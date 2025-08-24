my_list = [1, 2, 3, 5, 7, 2, 3, 4, 5]

print("Length: " + str(len(my_list)))
print("min: " + str(min(my_list)))
print("max: " + str(max(my_list)))
print("Count 3s: " + str(my_list.count(3)))
print("Position of a 3: " + str(my_list.index(3)))
print("Position of a 3: " + str(my_list.index(3, 3)))


my_list = [1, 2, 3]

new_list = ((my_list * 2)[1:5] + list((7, 8)))*4
print(new_list)

print(str((my_list + new_list).count(2)))
print(str((my_list + new_list)))