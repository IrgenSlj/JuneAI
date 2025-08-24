from random import seed, randrange
from datetime import datetime

seed(datetime.now().timestamp())

i = 1
columns = []
while True:
    column = set()

    # 10-19
    rand_number = randrange(10, 20)

    column.add(rand_number)

    while True:
        rand_number = randrange(10, 20)
        if rand_number not in column:
            column.add(rand_number)
            break

    # 20-29
    rand_number = randrange(20, 30)

    column.add(rand_number)

    while True:
        rand_number = randrange(20, 30)
        if rand_number not in column:
            column.add(rand_number)
            break

    # even 1-9
    rand_number = randrange(2, 10, 2)
    column.add(rand_number)

    # odd 40-49
    rand_number = randrange(41, 50, 2)
    column.add(rand_number)

    if column not in columns:
        columns.append(column)
        i += 1

    if i > 10:
        break


for column in columns:
    print(column)