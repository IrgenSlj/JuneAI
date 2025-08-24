my_dict = {
    "itamos":"proklitikos, authadis, anaidis",
    "onidos":"ntropi, kataisxuni",
    "pamfolyges":"aerologies, anohsies",
}

print(my_dict)
print()

my_dict["flinafhmata"] = "Anohsies, saxlamares"

print(my_dict)
print()

def add_word(dict):
    temp_d = {}
    temp_d.update(dict)
    key = input("Give a dict key: ")
    value = input(f"Give a definition for {key}: ")
    temp_d[key] = value
    return temp_d

my_dict = add_word(my_dict)
print(my_dict)
# print(my_dict["hi"])
print()