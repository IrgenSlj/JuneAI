pupils = [
    {
       "name": "John",
       "surname": "Cons",
       "age": 15,
       "father": "Mike",
       "class": 1,
       "id_number": "AN445F"
    },
    {
       "name": "Stam",
       "surname": "Pele",
       "age": 16,
       "father": "Craig",
       "class": 3,
       "id_number": "AN563F"
    },
    {
       "name": "Maria",
       "surname": "Kalas",
       "age": 13,
       "father": "Kostas",
       "class": 5,
       "id_number": "AH112F"
    },
]

while True:
    print("1 - Create \n2 - Read \n3 - Update \n4 - Delete \n5 - Exit")
    choice = input("Type number corresponding to action: ")

    if choice == 1:
        pupil = input("Type name of new pupil: ")
        pupils[pupil] = pupil
        surname = input("Type surname of pupil: ")
        pupils[surname] = surname
    elif choice == 2:
        pass
    elif choice == 3:
        pass
    elif choice == 4:
        pass
    else:
        print("Good bye!")