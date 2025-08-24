x = input("Type and integer: ")
while True:
    if x.isdigit():
        print(f"Number entered is: {x}")
        break
    else:
        x = input("Type and integer: ")