list = [1, 2, 3]

print(list)

list.append(5)

print(list)

list.insert(2, 4)

print(list)

last = list.pop()

print(list)
print(last)

first = list.pop(0)
print(list)
print(first)

list2 = ["a", "b", "c"]

list.extend(list2)
print(list)

list.remove("a")
print(list)

list2.clear()
print(list2)


cash_desk = []
cash_desk.append("Tom")
cash_desk.append("Bob")

print(cash_desk)

client_ready = cash_desk.pop(0)
print(f"{client_ready} is served and out of the que")
print(cash_desk)

cash_desk.append("Pam")
cash_desk.append("Jim")

print(cash_desk)

client_ready = cash_desk.pop(0)
print(f"{client_ready} served and out of the que")
print(f"People in que are {cash_desk}")


list = [1, 2, 3]
list.append("hello")
if "hello" in list:
    print("exists")
elif "world" not in list:
    print("nop")
else:
    print("not in list")

print(list)


list = [1, 2, 3, 4, 5]

print(len(list))
list.reverse()

print(list)

list.sort()
print(list)



list = []

list.append(float(input("Type first number: ")))

print(list)

list.append(float(input("Type second number: ")))

print(list)

list.append(float(input("Type third number: ")))

list.sort(reverse=True)
print(list)


movies = ["titanic", "blade runner", "forest gumb", "matrix"]

new_movie = input("Type a favorite movie: ")

if new_movie in movies:
    print("Movie is in the list already.")
else:
    movies.append(new_movie)
    print("Movie added to the list.")

movies.sort()
print(f"Current {len(movies)} movies in the list are: {movies}")