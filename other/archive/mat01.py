array = []

rows = int(input("Type number of rows: "))
cols = int(input("Type number of columns: "))

for i in range(rows):
    array.append([])
    for y in range(cols):
        array[i].append(int(input(f"Type value on column {y} of row {i}: ")))


for row in array:
    for col in row:
        print(col, end=" ")
    print("")