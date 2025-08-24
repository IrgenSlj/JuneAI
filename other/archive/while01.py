import math

cnt = 1
value = 0
lists = []

while cnt <= 10:
    usr_inp = int(input(f"Type number number {cnt}: "))
    lists.append(usr_inp)
    value += usr_inp
    print(f"Numbers sum is {value}, summed {cnt}/10")
    cnt += 1

print(f"\nNumbers entered are: {lists}")
print(f"Total sum is : {value}")

def list_sort(lists):
    lists.sort()
    return lists

lists_sort = list_sort(lists)
print(f"\nSorted numbers are: {lists_sort}")

n_max = lists[0]
for i in range(len(lists)):
    if lists[i] > n_max:
        n_max = lists[i]

print(f"The max number in the list is: {n_max}")