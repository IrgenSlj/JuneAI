heros_weapons = {
    "Black Panther":"Anti-metal claws",
    "Wolverine":"Claws",
    "Ultron":"Plasma Weapons",
    "Spider-Man":"Web-shooters",
    "Beast":"Claws",
    "Venom":"Web-shooters"
}

print("Key-value loop:")
print("----------------")
for key, value in heros_weapons.items():
    print(key + " has " + value)

print("\nOrdered key loop: ")
print("----------------")
for key in sorted(heros_weapons.keys()):
    print(key + " has " + heros_weapons[key])

print("\nWeapons Gallery: ")
print("-----------")
for value in set(heros_weapons.values()):
    print(value, end=", ")