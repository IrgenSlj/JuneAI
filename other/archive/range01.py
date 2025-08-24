cities = ["London", "New York", "Athens", "Adam", "Rdam"]

for city in cities:
    if cities.index(city) % 2 == 0:
        print(city)

print("============")

for i in range(0, len(cities), 2):
    print(cities[i])