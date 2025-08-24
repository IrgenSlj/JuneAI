def TextEdit(string):
    return string.strip().capitalize()

while True:
    string = input("Type your name: ")
    if string.isalpha():
        string = TextEdit(string)
        print(f"Your name is {string}\n")
        break
    else:
        string = input("Type only chars pls: ")

while True:
    string2 = input("Type your surname: ")
    if string2.isalpha():
        string2 = TextEdit(string2)
        print(f"Your surname is {string2}\n")
        break
    else:
        string2 = input("Type only chars pls: ")

print(f"\nWelcome {string} {string2}".strip().center(50))