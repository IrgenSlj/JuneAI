
string = "Hello malaka mou"
array = []
array.extend(string)
print(array)
print()

d = {}
for element in array:
    if element not in d:
        d[element] = 1
    else:
        d[element] += 1

ds = {}

for key in d:
    if key not in ds.values():
        ds[d[key]] = key
    else:
        ds[d[key]] = str(ds[key]) + str(key)

for key in sorted(ds.keys(), reverse=True):
    print(str(key) + " : " + str(ds[key]))


