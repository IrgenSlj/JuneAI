my_list = []

for i in range(10):
    x = int(input("Type a number from 10 to 20: "))

    while x > 20 or x < 10:
        x = int(input("Type a number from 10 to 20, not smaller or bigger: "))
    my_list.append(x)

print(f"List length is: {len(my_list)}")
my_tuple = tuple(my_list)
print(f"List turned tuple is: {my_tuple}")

new_list = []
for element in my_list:
    new_list.append(pow(element, 2))

new_list.sort()
new_tuple = tuple(new_list)
print(f"Sorted new tuple in the power of 2 is: {new_tuple}")