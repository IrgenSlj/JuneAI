import json

n = [1, "patata", 3, "hi"]

with open("numb.json", "w") as f:
    json.dump(n, f)


with open("numb.json", "r") as f:
    numbs = json.load(f)

print(numbs)  

