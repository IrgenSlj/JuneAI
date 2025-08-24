array = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

array.insert(3, [0, 0, 0, 0])

for rows in array:
    for cols in rows: 
        if cols < 10:
            print(" " + str(cols), end=" ")
        else:
            print(cols, end=" ")
    print("")
print("")

for row in array:
    row.append(1)

for rows in array:
    for cols in rows: 
        if cols < 10:
            print(" " + str(cols), end=" ")
        else:
            print(cols, end=" ")
    print("")
