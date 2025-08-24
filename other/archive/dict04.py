from random import randrange
from random import seed
from datetime import datetime

seed(datetime.now().timestamp())
N = 1000000
numbers = {}

for i in range(1, 7):
    numbers[i] = 0

print(numbers)

for i in range(N):
    x = randrange(1, 7)
    numbers[x] += 1

print(str(numbers) + "\n")

cnt = 0
for i in range(1, 7):
    print(f"probability of {i} is {numbers[i] / N}%")
    cnt += (numbers[i] / N)

