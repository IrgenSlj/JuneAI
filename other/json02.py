import json
from random import randrange

with open("numb.json", "w") as f:
    for _ in range(4):
        if randrange(3) == 0:
            json.dump([1, 2, 3], f)
        elif randrange(3) == 1:
            json.dump({"a":1, "b":2, "c":3}, f)
        else:
            json.dump(4, f)
        f.write("\n")

with open("numb.json", "r") as f:
    for line in f:
        print(json.loads(line))