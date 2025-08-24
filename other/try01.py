import json

try:
    with open("numb.json") as f: ##
        reminders = json.load(f)
except FileNotFoundError:
    reminders =[1, 2, 3, 4, 5]

def print_rem(rems):
    print(f"Here are your current reminders: ")
    for i in range(len(reminders)):
        print(f"no {i}: {rems[i]}", end=" ")
    print("\n")

print_rem(reminders)

while True:
    choice = input("Type 1 to add reminder, 2 to remove, 3 to print and 4 to exit: ")
    while True:
        try:
            choice = int(choice)
            break
        except ValueError:
            choice = input("Error! Type 1 to add reminder, 2 to remove, 3 to print and 4 to exit: ")


    if choice == 1:
        add = input("Type a reminder to add: ")
        reminders.append(add)
        with open("numb.json", "w") as f: ##
            json.dump(reminders, f)
        print(f"Reminder {add} added to list\n")

    elif choice == 2:
        rem = int(input(f"Type index of reminder to remove: "))
        len = int(len(reminders))
        for i in range(len):
            if i == rem:
                len -= 1
                reminders.pop(i)
                print(f"Reminder no {rem} was removed\n")
                with open("numb.json", "w") as f: ##
                    json.dump(reminders, f)
                continue


    elif choice == 3:
        print_rem(reminders)

    elif choice == 4:
        print("Exiting now..\n")
        break
    else:
        print("wrong input")
        continue