numbers = [5, 8, 45, 3, 6, 4, 2, 8]
search = 8

for number in numbers:
    print(number)
    if number == search:
        print(f"Found: {number}")
        break
else:
    print("Not found")


'''
if 2 in numbers:
    print("Found 2")

for i in range(0, 11):
    if i % 2 != 0:
        continue

    print(i)

print("--------------")

for number in  range(1, 11):
   
    if number == 7:
        continue
    print(number)
else:
    print("Done")
'''