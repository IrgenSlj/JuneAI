
list = [1, 2, 3, 4, 5, 56]

def print_list(list):
    n = len(list)
    if ln > 0:
        print(list[0], end=",")
        print_list(list[1:])

print_list(list)