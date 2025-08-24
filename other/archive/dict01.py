empty = {}
person = {
    "name":"Nick",
    "age": 29,
    "grade": 10
}
person["hair"] = "long"

print(person["name"])
print(person["age"])
print(person["grade"])
print(person["hair"])
print(type(person))
print()

person["name"] = "John"

copy = person.copy()

print(copy)
print(copy is person)
print(copy == person)
print()

copy.pop("hair")
print(copy)
print(copy == person)
print()

a_list = [("name", "Giorgis"), ("prof", "pianist")]
a_dict = dict(a_list)
prof = a_dict.pop("prof")

print(a_dict)
print(a_dict["name"])
print(prof)


