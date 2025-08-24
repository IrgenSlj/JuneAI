list_int = [1, 2, 3]
list_float = [3.12, 4.5, 1.4]
list_string = ["Hello", "World"]

list_misc = [1, "hello", [1, 2]]

print(list_float[1])
print(f"{list_string[0]} {list_string[1]}")

list_misc[2] = "world"
print(list_misc[:3])


friends = ["John", "Mike", "Max"]

print(type(friends))

friends[0] = "Maria"

print(friends[0])
print(friends[1])
print(friends[2])

x = 0
list = [5, 25, 9]

if list[0] >= list[1] and list[0] >= list[2]:
    x = list[0]
elif list[1] >= list[2]:
    x = list[1]
else:
    x = list[2]

print(x)  

# or better

x = list[0]

if list[1] > x:
    x = list[1]

if list[2] > x:
    x = list[2]

print(x)