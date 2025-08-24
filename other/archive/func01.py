def input_integer():
    while True:
        data = input("Type an integer: ").strip()
        if data.isdigit():
            return data
        else:
            data = input("Type only an integer: ")

int_my = input_integer()

print(int_my)